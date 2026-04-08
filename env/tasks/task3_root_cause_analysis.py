"""
env/tasks/task3_root_cause_analysis.py

Task 3 — Root Cause Analysis (Hard)

The agent sees 200 log lines covering a failure window and must:
  1. Identify the root cause component
  2. Reconstruct the failure timeline (key events in order)
  3. List affected components
  4. Propose remediation steps

Grader uses a rubric across 4 dimensions:
  root_cause_component  → 0.35
  timeline_key_events   → 0.25
  affected_components   → 0.20
  remediation_steps     → 0.20
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from env.models import Action, Observation, Reward

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

TASK_ID = "task3"
MAX_STEPS = 10

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "task3")

ROOT_CAUSE_KEYWORDS = [
    "dataxceiver", "datanode$dataxceiver", "datanode",
    "data xceiver", "data_xceiver",
]

VALID_REMEDIATION_STEPS = [
    "check network",
    "network connectivity",
    "inspect datanode logs",
    "datanode logs",
    "restart datanode",
    "restart affected datanode",
    "replication factor",
    "block replication",
    "heap usage",
    "memory",
    "oom",
    "out of memory",
    "ioexception",
    "stack trace",
    "circuit breaker",
    "connection pool",
]

VALID_COMPONENTS = {
    "dfs.datanode$dataxceiver": 1.0,
    "dfs.datanode$packetresponder": 1.0,
    "dataxceiver": 1.0,
    "packetresponder": 1.0,
    "datanode": 0.5,  # partial credit — too generic
    "dfs.datanode": 0.5,
}


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

def load_task() -> Dict[str, Any]:
    logs_path = os.path.join(DATA_DIR, "logs.txt")
    gt_path = os.path.join(DATA_DIR, "ground_truth.json")

    with open(logs_path, "r") as f:
        log_content = f.read()

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    return {
        "log_content": log_content,
        "ground_truth": ground_truth,
    }


# ──────────────────────────────────────────────────────────────
# Initial Observation
# ──────────────────────────────────────────────────────────────

def make_initial_observation(log_content: str) -> Observation:
    return Observation(
        task_id=TASK_ID,
        log_content=log_content,
        current_step=0,
        max_steps=MAX_STEPS,
        context={
            "total_lines": 200,
            "components_present": [
                "dfs.DataNode$DataXceiver",
                "dfs.DataNode$PacketResponder",
                "dfs.FSNamesystem",
                "dfs.DataBlockScanner",
            ],
            "hint": (
                "Analyze the 200 log lines to identify: "
                "(1) the root cause component, "
                "(2) a timeline of key failure events, "
                "(3) affected components, "
                "(4) remediation steps. "
                "Look for WARN lines and patterns across components."
            ),
        },
        feedback=(
            "Episode started. You have 10 steps. "
            "Analyze the 200-line log window and submit your root cause analysis."
        ),
        done=False,
    )


# ──────────────────────────────────────────────────────────────
# Action Handlers
# ──────────────────────────────────────────────────────────────

def handle_analyze(
    payload: Dict[str, Any],
    log_content: str,
    current_step: int,
) -> Observation:
    """
    Filter logs by level, component, keyword, block ID, or line range.
    payload examples:
      {"filter_level": "WARN"}
      {"filter_component": "DataXceiver"}
      {"filter_keyword": "exception"}
      {"filter_block": "blk_-2918118818249673980"}
      {"line_range": [1, 50]}
    """
    lines = log_content.strip().split("\n")
    filtered = []

    filter_level = payload.get("filter_level", "").upper()
    filter_component = payload.get("filter_component", "").lower()
    filter_keyword = payload.get("filter_keyword", "").lower()
    filter_block = payload.get("filter_block", "")
    line_range = payload.get("line_range", None)

    if filter_level:
        filtered = [l for l in lines if filter_level in l]
        feedback = f"Filtered to {len(filtered)} lines with level '{filter_level}'."

    elif filter_component:
        filtered = [l for l in lines if filter_component in l.lower()]
        feedback = f"Filtered to {len(filtered)} lines from component '{filter_component}'."

    elif filter_keyword:
        filtered = [l for l in lines if filter_keyword in l.lower()]
        feedback = f"Filtered to {len(filtered)} lines containing '{filter_keyword}'."

    elif filter_block:
        filtered = [l for l in lines if filter_block in l]
        feedback = f"Filtered to {len(filtered)} lines referencing block '{filter_block}'."

    elif line_range and len(line_range) == 2:
        start, end = max(1, line_range[0]), min(len(lines), line_range[1])
        filtered = lines[start - 1: end]
        feedback = f"Showing lines {start}–{end} ({len(filtered)} lines)."

    else:
        filtered = lines
        feedback = "No valid filter. Showing all lines."

    return Observation(
        task_id=TASK_ID,
        log_content="\n".join(filtered),
        current_step=current_step,
        max_steps=MAX_STEPS,
        context={
            "hint": (
                "Use filtered results to build your root cause hypothesis. "
                "Try filtering by 'WARN', 'DataXceiver', or 'exception'."
            )
        },
        feedback=feedback,
        done=False,
    )


def handle_query(
    payload: Dict[str, Any],
    log_content: str,
    current_step: int,
) -> Observation:
    """
    Answer analytical questions about the 200-line log window.
    payload: {"question": "What components appear most often?"}
    """
    question = payload.get("question", "").lower()
    lines = log_content.strip().split("\n")

    warn_lines = [l for l in lines if "WARN" in l]
    info_lines = [l for l in lines if "INFO" in l]
    exception_lines = [l for l in lines if "exception" in l.lower()]

    # Extract components
    component_counts: Dict[str, int] = {}
    for line in lines:
        match = re.search(r"dfs\.\S+", line)
        if match:
            comp = match.group(0).rstrip(",:")
            component_counts[comp] = component_counts.get(comp, 0) + 1

    # Extract block IDs
    all_blocks = []
    for line in lines:
        all_blocks.extend(re.findall(r"blk_-?\d+", line))
    block_freq: Dict[str, int] = {}
    for b in all_blocks:
        block_freq[b] = block_freq.get(b, 0) + 1

    if "warn" in question and ("how many" in question or "count" in question):
        answer = f"There are {len(warn_lines)} WARN lines in this log window."

    elif "component" in question:
        sorted_comps = sorted(component_counts.items(), key=lambda x: -x[1])[:5]
        summary = ", ".join(f"{c} ({n})" for c, n in sorted_comps)
        answer = f"Top components by frequency: {summary}."

    elif "exception" in question:
        answer = (
            f"There are {len(exception_lines)} lines containing exceptions. "
            f"All exceptions are from DataXceiver threads serving blocks to clients."
        )

    elif "block" in question or "blk" in question:
        top_blocks = sorted(block_freq.items(), key=lambda x: -x[1])[:5]
        summary = ", ".join(f"{b} ({n}x)" for b, n in top_blocks)
        answer = f"Most-referenced blocks: {summary}."

    elif "timeline" in question or "when" in question or "time" in question:
        if warn_lines:
            times = []
            for wl in warn_lines[:3]:
                m = re.search(r"\[(\d+) (\d+)\]", wl)
                if m:
                    times.append(f"{m.group(1)} {m.group(2)}")
            answer = (
                f"WARN events span across the log window. "
                f"First exceptions at: {', '.join(times) if times else 'unknown timestamps'}. "
                f"Total WARN events: {len(warn_lines)}."
            )
        else:
            answer = "No WARN events found in this window."

    elif "root cause" in question or "cause" in question:
        answer = (
            f"The log shows {len(warn_lines)} WARN events from DataXceiver threads, "
            f"all indicating failures while serving blocks to clients. "
            f"This pattern suggests DataXceiver instability as a primary failure point."
        )

    elif "remediation" in question or "fix" in question or "solution" in question:
        answer = (
            "Potential remediation areas based on logs: "
            "network connectivity between DataNodes and clients, "
            "DataNode service health, block replication integrity, "
            "DataNode heap/memory, and IOException root causes."
        )

    else:
        answer = (
            f"Log window summary: {len(lines)} total lines, "
            f"{len(warn_lines)} WARN, {len(info_lines)} INFO, "
            f"{len(exception_lines)} exception lines. "
            f"Components present: {', '.join(sorted(component_counts.keys()))}."
        )

    return Observation(
        task_id=TASK_ID,
        log_content=answer,
        current_step=current_step,
        max_steps=MAX_STEPS,
        context={},
        feedback=f"Query answered: '{payload.get('question', '')}'",
        done=False,
    )


# ──────────────────────────────────────────────────────────────
# Grader
# ──────────────────────────────────────────────────────────────

def _score_root_cause(
    predicted_root_cause: str,
    ground_truth: Dict[str, Any],
) -> float:
    """
    Score the root cause identification.
    Full credit for DataXceiver; partial for generic DataNode.
    """
    if not predicted_root_cause:
        return 0.0

    pred_lower = predicted_root_cause.lower()

    for keyword, credit in VALID_COMPONENTS.items():
        if keyword in pred_lower:
            return credit

    return 0.0


def _score_timeline(
    predicted_timeline: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Score the failure timeline.
    3 key events expected; partial credit per event captured.
    """
    gt_timeline = ground_truth.get("timeline", [])
    if not gt_timeline or not predicted_timeline:
        return 0.0

    # Key concepts that must appear in the predicted timeline
    key_concepts = [
        ["exception", "error", "warn", "fail", "first"],
        ["repeated", "multiple", "recur", "pattern", "again"],
        ["last", "final", "established", "persist", "continued"],
    ]

    score = 0.0
    per_event_credit = 1.0 / len(key_concepts)

    pred_text = " ".join(
        str(e.get("event", "")) + " " + str(e.get("time", ""))
        for e in predicted_timeline
    ).lower()

    for concepts in key_concepts:
        if any(c in pred_text for c in concepts):
            score += per_event_credit

    # Bonus for having chronological ordering with line numbers
    has_line_numbers = any("line_number" in e or "line" in str(e) for e in predicted_timeline)
    if has_line_numbers and len(predicted_timeline) >= 2:
        score = min(1.0, score + 0.1)

    return round(score, 4)


def _score_affected_components(
    predicted_components: List[str],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Score identification of affected components.
    Expected: DataXceiver and PacketResponder.
    """
    expected = [c.lower() for c in ground_truth.get("affected_components", [])]
    if not expected or not predicted_components:
        return 0.0

    predicted_lower = [c.lower() for c in predicted_components]
    pred_text = " ".join(predicted_lower)

    correct = 0
    for exp in expected:
        # Extract key part of component name
        key = exp.split("$")[-1].lower() if "$" in exp else exp.lower()
        if any(key in p or exp in p for p in predicted_lower) or key in pred_text:
            correct += 1

    return round(correct / len(expected), 4)


def _score_remediation(
    predicted_steps: List[str],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Score remediation steps.
    0.04 per valid step (max 5 valid steps = 0.20 weight in final).
    We score this sub-dimension as fraction of max_steps covered.
    """
    if not predicted_steps:
        return 0.0

    max_steps = 5
    valid_count = 0

    pred_text = " ".join(str(s).lower() for s in predicted_steps)

    for keyword in VALID_REMEDIATION_STEPS:
        if keyword in pred_text:
            valid_count += 1
            if valid_count >= max_steps:
                break

    # Deduplicate by concept — rough dedup
    return round(min(1.0, valid_count / max_steps), 4)


def grade(payload: Dict[str, Any], ground_truth: Dict[str, Any]) -> Reward:
    """
    Grade the agent's submitted root cause analysis.

    Expected payload:
    {
      "root_cause": "dfs.DataNode$DataXceiver",
      "timeline": [
        {"line_number": 78, "event": "First exception observed"},
        ...
      ],
      "affected_components": ["dfs.DataNode$DataXceiver", ...],
      "remediation": ["Check network", "Restart DataNode", ...]
    }
    """
    root_cause = payload.get("root_cause", "")
    timeline = payload.get("timeline", [])
    affected_components = payload.get("affected_components", [])
    remediation = payload.get("remediation", [])

    # Individual dimension scores (each 0.0–1.0)
    rc_score = _score_root_cause(root_cause, ground_truth)
    tl_score = _score_timeline(timeline, ground_truth)
    comp_score = _score_affected_components(affected_components, ground_truth)
    rem_score = _score_remediation(remediation, ground_truth)

    # Weighted final score
    weights = {
        "root_cause": 0.35,
        "timeline": 0.25,
        "affected_components": 0.20,
        "remediation": 0.20,
    }
    final_score = (
        rc_score * weights["root_cause"]
        + tl_score * weights["timeline"]
        + comp_score * weights["affected_components"]
        + rem_score * weights["remediation"]
    )

    # Feedback
    feedback_parts = [f"Final score: {final_score:.2f}."]
    feedback_parts.append(
        f"Root cause: {rc_score:.2f}/1.0 "
        f"({'✓' if rc_score >= 0.9 else '~' if rc_score > 0 else '✗'} "
        f"predicted='{root_cause}', "
        f"expected='dfs.DataNode$DataXceiver')."
    )
    feedback_parts.append(
        f"Timeline: {tl_score:.2f}/1.0 "
        f"({len(timeline)} events provided, "
        f"{len(ground_truth.get('timeline', []))} key events expected)."
    )
    feedback_parts.append(
        f"Affected components: {comp_score:.2f}/1.0 "
        f"({len(affected_components)} provided)."
    )
    feedback_parts.append(
        f"Remediation: {rem_score:.2f}/1.0 "
        f"({len(remediation)} steps provided)."
    )

    return Reward(
        score=round(final_score, 4),
        partial_scores={
            "root_cause_component": round(rc_score * weights["root_cause"], 4),
            "timeline_key_events": round(tl_score * weights["timeline"], 4),
            "affected_components": round(comp_score * weights["affected_components"], 4),
            "remediation_valid_steps": round(rem_score * weights["remediation"], 4),
            "root_cause_raw": rc_score,
            "timeline_raw": tl_score,
            "components_raw": comp_score,
            "remediation_raw": rem_score,
        },
        feedback=" ".join(feedback_parts),
        penalties=0.0,
    )


# ──────────────────────────────────────────────────────────────
# Step handler (called by environment.py)
# ──────────────────────────────────────────────────────────────

def step(
    action: Action,
    log_content: str,
    ground_truth: Dict[str, Any],
    current_step: int,
) -> Dict[str, Any]:
    if action.action_type == "analyze":
        obs = handle_analyze(action.payload, log_content, current_step)
        return {"observation": obs, "reward": None, "done": False}

    elif action.action_type == "query":
        obs = handle_query(action.payload, log_content, current_step)
        return {"observation": obs, "reward": None, "done": False}

    elif action.action_type == "submit":
        reward = grade(action.payload, ground_truth)
        obs = Observation(
            task_id=TASK_ID,
            log_content="",
            current_step=current_step,
            max_steps=MAX_STEPS,
            context={},
            feedback=f"Episode complete. Final score: {reward.score:.4f}",
            done=True,
        )
        return {"observation": obs, "reward": reward, "done": True}

    else:
        obs = Observation(
            task_id=TASK_ID,
            log_content=log_content,
            current_step=current_step,
            max_steps=MAX_STEPS,
            context={},
            feedback=(
                f"Unknown action_type '{action.action_type}'. "
                "Use: analyze, query, submit."
            ),
            done=False,
        )
        return {"observation": obs, "reward": None, "done": False, "penalty": 0.1}