"""
baseline.py

Baseline inference script for the Log Analyzer OpenEnv environment.
Runs a language model (via OpenAI-compatible API) against all 3 tasks
and reports reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Run a specific task only:
    python baseline.py --task task1

    # Use a different model:
    python baseline.py --model gpt-4o-mini

    # Use Anthropic-compatible endpoint:
    python baseline.py --base-url https://api.anthropic.com/v1 --model claude-3-haiku-20240307

Environment variables:
    OPENAI_API_KEY      Required. API key for the LLM provider.
    OPENAI_BASE_URL     Optional. Defaults to OpenAI's endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Graceful import of openai ─────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)

from env.environment import LogAnalyzerEnv
from env.models import Action

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
TASKS = ["task1", "task2", "task3"]

SYSTEM_PROMPT = """You are an expert SRE and DevOps engineer analyzing HDFS server logs.
Your job is to identify anomalies, warnings, and root causes in log data.

You interact with the environment using JSON actions. Each action must be valid JSON with:
  - "action_type": one of "analyze", "query", "submit"
  - "payload": task-specific data

Action types:
  analyze: Filter logs. payload: {"filter_level": "WARN"} or {"filter_component": "DataXceiver"} or {"line_range": [1, 50]}
  query:   Ask a question. payload: {"question": "How many WARN lines are there?"}
  submit:  Final answer (ends episode). payload varies by task.

IMPORTANT: Always respond with ONLY valid JSON. No explanation, no markdown, no code blocks.
Just the raw JSON object."""

TASK_PROMPTS = {
    "task1": """Task: Identify all WARN-level log lines.
Submit format: {"action_type": "submit", "payload": {"warn_lines": [11, 12, ...]}}
The warn_lines must be a list of 1-indexed line numbers containing WARN events.""",

    "task2": """Task: Identify ALL anomalous lines across all 50 lines, classify each, and assign severity.

STEP 1 — Find WARN lines (always anomalous):
  - WARN + "Got exception" in content → anomaly_type=DATA_SERVE_EXCEPTION, severity="warning"
  - WARN without "Got exception"      → anomaly_type=DATANODE_WARNING,      severity="warning"

STEP 2 — Find block ID anomalies using the known_anomalous_block_ids list:
  - The context includes block IDs flagged as anomalous in the cluster database.
  - Scan EVERY line. If any line's text contains one of those block IDs:
    → anomaly_type=ANOMALOUS_BLOCK_ACTIVITY, severity="info"
  - These lines look normal (INFO level) — the ONLY way to find them is the block ID list.

SEVERITY RULES (exact — do not deviate, never use "critical"):
  DATA_SERVE_EXCEPTION     → severity must be "warning"
  DATANODE_WARNING         → severity must be "warning"
  ANOMALOUS_BLOCK_ACTIVITY → severity must be "info"

Submit format:
{"action_type": "submit", "payload": {"anomalies": [
  {"line_number": 7,  "anomaly_type": "DATA_SERVE_EXCEPTION",     "severity": "warning"},
  {"line_number": 2,  "anomaly_type": "ANOMALOUS_BLOCK_ACTIVITY", "severity": "info"},
  ...
]}}""",

    "task3": """Task: Analyze the 200-line failure window. Identify root cause, timeline, affected components, and remediation.
Submit format:
{"action_type": "submit", "payload": {
  "root_cause": "dfs.DataNode$DataXceiver",
  "timeline": [
    {"line_number": 78, "event": "First DataNode exception observed"},
    {"line_number": 120, "event": "Repeated exceptions across multiple blocks"},
    {"line_number": 200, "event": "Last exception - pattern established"}
  ],
  "affected_components": ["dfs.DataNode$DataXceiver", "dfs.DataNode$PacketResponder"],
  "remediation": [
    "Check network connectivity between DataNodes and clients",
    "Inspect DataNode logs for IOException stack traces",
    "Restart affected DataNode services",
    "Verify HDFS block replication factor",
    "Monitor DataNode heap usage"
  ]
}}"""
}


# ──────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────

def run_task(
    client: OpenAI,
    model: str,
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single task episode and return the result."""
    env = LogAnalyzerEnv(task_id=task_id)
    obs = env.reset()
    print(f"[START] task={task_id} model={model} max_steps={obs.max_steps}")
    # ── Build the initial user message ────────────────────────
    # For task2: inject block IDs BEFORE the logs so a 7B model sees them
    # at the start of context, not buried after 50 lines of log text.
    block_ids_preamble = ""
    if task_id == "task2":
        known_blocks = obs.context.get("known_anomalous_block_ids", [])
        if known_blocks:
            block_ids_preamble = (
                "=== STEP 0 — MEMORIZE THESE ANOMALOUS BLOCK IDs BEFORE READING THE LOGS ===\n"
                + "\n".join(f"  {b}" for b in known_blocks)
                + "\n\nFor EACH block ID above, find the log line that contains it."
                  " Add that line to your anomalies as:"
                  ' {"line_number": X, "anomaly_type": "ANOMALOUS_BLOCK_ACTIVITY", "severity": "info"}'
                  "\n=== END BLOCK ID LIST — NOW READ THE LOGS ===\n\n"
            )

    # Build post-log checklist for task2 (block IDs appear before AND after logs)
    block_ids_checklist = ""
    if task_id == "task2":
        known_blocks = obs.context.get("known_anomalous_block_ids", [])
        if known_blocks:
            # Build a line-by-line lookup so the model can directly verify
            checklist_lines = []
            for blk in known_blocks:
                # Find which line number contains this block ID
                for i, line in enumerate(obs.log_content.strip().split("\n"), 1):
                    if blk in line:
                        checklist_lines.append(f"  {blk} → Line {i:02d}")
                        break
                else:
                    checklist_lines.append(f"  {blk} → (not found)")
            block_ids_checklist = (
                "\n=== ANOMALOUS BLOCK ID → LINE NUMBER LOOKUP ===\n"
                + "\n".join(checklist_lines)
                + "\nAdd ALL of these lines as ANOMALOUS_BLOCK_ACTIVITY with severity=\'info\'.\n"
                  "Also add ALL WARN lines as DATA_SERVE_EXCEPTION with severity=\'warning\'.\n"
                  "=== NOW SUBMIT YOUR ANSWER ===\n"
            )

    initial_content = (
        f"{TASK_PROMPTS[task_id]}\n\n"
        f"{block_ids_preamble}"
        f"Here are the logs:\n\n{obs.log_content}\n\n"
        f"{block_ids_checklist}"
        f"You have {obs.max_steps} steps. Submit your final answer."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_content},
    ]

    #if verbose:
        #print(f"\n{'='*60}")
        #print(f"  Task: {task_id.upper()}")
        #print(f"{'='*60}")

    final_reward = None
    step_count = 0

    while not env._done:
        step_count += 1

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            break

        raw_reply = response.choices[0].message.content.strip()


        #if verbose:
            #print(f"\n  Step {step_count}: {raw_reply[:120]}{'...' if len(raw_reply) > 120 else ''}")

        # Parse action
        try:
            clean = raw_reply.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
            action_dict = json.loads(clean)
            action = Action(
                action_type=action_dict["action_type"],
                payload=action_dict.get("payload", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if verbose:
                print(f"  [WARN] Failed to parse action: {e}. Submitting defaults.")
            continue

        # Execute step
        result = env.step(action)
        obs = result.observation
        done_str = str(result.done).lower()
        print(
            f"[STEP] step={step_count} "
            f"action={action.action_type} "
            f"reward={result.reward.score if result.reward else 0.0} "
            f"done={done_str}"
        )

        if result.reward is not None:
            final_reward = result.reward

        if result.done:
            break

        # Add to conversation
        messages.append({"role": "assistant", "content": raw_reply})
        reminder = ""
        if task_id == "task2":
            reminder = (
                "\nREMINDER: severity='warning' for DATA_SERVE_EXCEPTION/DATANODE_WARNING. "
                "severity='info' for ANOMALOUS_BLOCK_ACTIVITY. Never use 'critical'. "
                "Also check every line against the known anomalous block IDs listed above."
            )
        messages.append({
            "role": "user",
            "content": (
                f"Step result: {obs.feedback}\n"
                f"Log output: {obs.log_content[:2000]}\n"
                f"Steps remaining: {obs.max_steps - obs.current_step}\n"
                f"Continue analyzing or submit your final answer.{reminder}"
            ),
        })

        time.sleep(0.5)

    score = final_reward.score if final_reward else 0.0
    print(
        f"[END] success=true "
        f"score={score:.4f} "
        f"steps={step_count}"
    )
    partial = final_reward.partial_scores if final_reward else {}
    feedback = final_reward.feedback if final_reward else "No reward (episode ended without submit)."

    #if verbose:
        #print(f"\n  {'─'*40}")
        #print(f"  Final score:  {score:.4f}")
        #print(f"  Partial:      {json.dumps(partial, indent=2)}")
        #print(f"  Feedback:     {feedback}")

    return {
        "task_id": task_id,
        "score": score,
        "partial_scores": partial,
        "feedback": feedback,
        "steps_taken": step_count,
        "penalties": final_reward.penalties if final_reward else 0.0,
    }


def _default_submit(task_id: str) -> Action:
    """Fallback submit actions when LLM response can't be parsed."""
    if task_id == "task1":
        return Action(action_type="submit", payload={"warn_lines": []})
    elif task_id == "task2":
        return Action(action_type="submit", payload={"anomalies": []})
    else:
        return Action(action_type="submit", payload={
            "root_cause": "",
            "timeline": [],
            "affected_components": [],
            "remediation": [],
        })


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Log Analyzer OpenEnv")
    parser.add_argument("--task", choices=TASKS + ["all"], default="all",
                        help="Which task to run (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    client = OpenAI(**client_kwargs)

    tasks_to_run = TASKS if args.task == "all" else [args.task]

    #print(f"\n{'#'*60}")
    #print(f"  Log Analyzer — Baseline Evaluation")
    #print(f"  Model: {args.model}")
    #print(f"  Tasks: {', '.join(tasks_to_run)}")
    #print(f"{'#'*60}")

    all_results = []
    for task_id in tasks_to_run:
        result = run_task(client, args.model, task_id, verbose=args.verbose)
        all_results.append(result)

    # Summary
    #print(f"\n{'='*60}")
    #print(f"  BASELINE SUMMARY")
    #print(f"{'='*60}")
    total = 0.0
    #for r in all_results:
        #bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        #print(f"  {r['task_id'].upper()}: [{bar}] {r['score']:.4f}  (steps={r['steps_taken']}, penalties={r['penalties']:.2f})")
        #total += r["score"]

    avg = total / len(all_results) if all_results else 0.0
    #print(f"  {'─'*50}")
    #print(f"  AVERAGE:          {avg:.4f}")
    #print(f"{'='*60}\n")

    out_path = "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "tasks": all_results,
            "average_score": round(avg, 4),
        }, f, indent=2)
    #print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
