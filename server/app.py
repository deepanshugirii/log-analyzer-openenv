"""
app.py

Gradio web interface for the Log Analyzer OpenEnv environment.
Runs on Hugging Face Spaces.

Features:
  - Interactive task runner (try all 3 tasks)
  - Step-by-step action submission
  - Live reward display
  - Environment state inspector
"""
from __future__ import annotations
from gradio import mount_gradio_app


import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import gradio as gr

from env.environment import LogAnalyzerEnv
from env.models import Action
from fastapi import FastAPI
app = FastAPI()
# ──────────────────────────────────────────────────────────────
# Global state (simple; single-user HF Space)
# ──────────────────────────────────────────────────────────────

_env: Optional[LogAnalyzerEnv] = None
_log_history: list = []


def _get_env() -> Optional[LogAnalyzerEnv]:
    return _env


def _format_obs(obs) -> str:
    if obs is None:
        return ""
    lines = [
        f"📋 Task: {obs.task_id}  |  Step: {obs.current_step}/{obs.max_steps}  |  Done: {obs.done}",
        f"💬 Feedback: {obs.feedback}",
        "",
    ]
    if obs.log_content:
        lines.append("─── Log Output ─────────────────────────────────────────")
        lines.append(obs.log_content[:4000])
        if len(obs.log_content) > 4000:
            lines.append(f"... [{len(obs.log_content) - 4000} more chars truncated]")
    return "\n".join(lines)


def _format_reward(reward) -> str:
    if reward is None:
        return ""
    #score_bar = "█" * int(reward.score * 20) + "░" * (20 - int(reward.score * 20))
    lines = [
       # f"🏆 FINAL SCORE: {reward.score:.4f}  [{score_bar}]",
        "",
        "Partial scores:",
        json.dumps(reward.partial_scores, indent=2),
        "",
        f"Feedback: {reward.feedback}",
    ]
    if reward.penalties > 0:
        lines.append(f"⚠️  Penalties: -{reward.penalties:.2f}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────────────────────

def reset_env(task_id: str) -> Tuple[str, str, str, str]:
    global _env, _log_history
    _env = LogAnalyzerEnv(task_id=task_id)
    obs = _env.reset()
    _log_history = []
    _log_history.append(f"[RESET] {task_id} — episode started.")

    action_hint = {
        "task1": '{"action_type": "analyze", "payload": {"filter_level": "WARN"}}',
        "task2": '{"action_type": "analyze", "payload": {"filter_keyword": "exception"}}',
        "task3": '{"action_type": "query", "payload": {"question": "How many WARN lines are there?"}}',
    }.get(task_id, "")

    return (
        _format_obs(obs),
        "",  # clear reward panel
        json.dumps(_env.state(), indent=2),
        action_hint,
    )


def submit_action(action_json: str) -> Tuple[str, str, str]:
    global _env, _log_history

    if _env is None:
        return "⚠️ No active episode. Click 'Start Episode' first.", "", ""

    if _env._done:
        return "⚠️ Episode is over. Click 'Start Episode' to restart.", "", ""

    # Parse action
    try:
        clean = action_json.strip()
        if clean.startswith("```"):
            lines_raw = clean.split("\n")
            clean = "\n".join(lines_raw[1:-1])
        d = json.loads(clean)
        action = Action(action_type=d["action_type"], payload=d.get("payload", {}))
    except Exception as e:
        return f"❌ Invalid JSON: {e}\n\nInput was:\n{action_json}", "", json.dumps(_env.state(), indent=2)

    result = _env.step(action)
    obs = result.observation
    _log_history.append(
        f"[Step {obs.current_step}] {action.action_type} → {obs.feedback}"
    )

    reward_text = _format_reward(result.reward) if result.reward else ""
    return _format_obs(obs), reward_text, json.dumps(_env.state(), indent=2)


#def get_action_templates(task_id: str, template_name: str) -> str:
#    templates = {
#        "task1": {
#            "Filter WARN lines": '{"action_type": "analyze", "payload": {"filter_level": "WARN"}}',
#            "Count WARN lines": '{"action_type": "query", "payload": {"question": "How many WARN lines are there?"}}',
#            "Which WARN lines?": '{"action_type": "query", "payload": {"question": "Which lines contain WARN events?"}}',
#            "Submit (example)": '{"action_type": "submit", "payload": {"warn_lines": [11, 12, 13]}}',
#        },
#        "task2": {
#            "Filter WARN lines": '{"action_type": "analyze", "payload": {"filter_level": "WARN"}}',
#            "Filter exceptions": '{"action_type": "analyze", "payload": {"filter_keyword": "exception"}}',
#            "Query components": '{"action_type": "query", "payload": {"question": "What components are present?"}}',
#            "Submit (example)": '{"action_type": "submit", "payload": {"anomalies": [{"line_number": 7, "anomaly_type": "DATA_SERVE_EXCEPTION", "severity": "warning"}]}}',
#        },
#        "task3": {
#            "Filter WARN lines": '{"action_type": "analyze", "payload": {"filter_level": "WARN"}}',
#            "Filter DataXceiver": '{"action_type": "analyze", "payload": {"filter_component": "DataXceiver"}}',
#            "Query root cause": '{"action_type": "query", "payload": {"question": "What is the likely root cause?"}}',
#            "Query timeline": '{"action_type": "query", "payload": {"question": "What is the timeline of events?"}}',
#            "Lines 1-50": '{"action_type": "analyze", "payload": {"line_range": [1, 50]}}',
#            "Submit (example)": '{"action_type": "submit", "payload": {"root_cause": "dfs.DataNode$DataXceiver", "timeline": [{"line_number": 78, "event": "First exception"}], "affected_components": ["dfs.DataNode$DataXceiver"], "remediation": ["Check network connectivity", "Restart DataNode"]}}',
#        },
#    }
    #task_templates = templates.get(task_id, {})
    #return task_templates.get(template_name, "")


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

CSS = """
.score-box { font-family: monospace; background: #0d1117; color: #58a6ff; padding: 12px; border-radius: 8px; }
.log-box { font-family: 'Courier New', monospace; font-size: 12px; }
.title-block { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="Log Analyzer — OpenEnv", css=CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
# 🔍 Log Analyzer — OpenEnv Environment

**A real-world HDFS server log analysis environment with 3 progressive tasks.**

An AI agent (or human) analyzes server logs to detect warnings, classify anomalies,
and perform root cause analysis — mimicking the daily work of a DevOps/SRE engineer.

---
        """,
        elem_classes=["title-block"],
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Episode Control")

            task_selector = gr.Radio(
                choices=["task1", "task2", "task3"],
                value="task1",
                label="Select Task",
                info="Easy → Medium → Hard",
            )

            task_descriptions = gr.Markdown(
                """
**Task 1 — Warning Detection** (Easy, 5 steps)
Find all WARN-level lines in 20 log entries. Grader: F1 score.

**Task 2 — Anomaly Classification** (Medium, 7 steps)
Identify anomalies in 50 lines, classify type + severity.
Grader: weighted detection (40%) + type (35%) + severity (25%).

**Task 3 — Root Cause Analysis** (Hard, 10 steps)
Analyze 200 lines for root cause, timeline, affected components, remediation.
Grader: rubric across 4 dimensions.
                """
            )

            start_btn = gr.Button("▶ Start Episode", variant="primary", size="lg")

            #gr.Markdown("### 📝 Action Templates")
            #template_selector = gr.Dropdown(
            #    choices=[
            #        "Filter WARN lines",
            #        "Filter exceptions",
            #        "Filter DataXceiver",
            #        "Count WARN lines",
            #        "Which WARN lines?",
            #        "Query components",
            #        "Query root cause",
            #        "Query timeline",
            #        "Lines 1-50",
            #        "Submit (example)",
            #    ],
            #    label="Load template",
            #    value=None,
            #)

            state_display = gr.Code(
                label="Environment State",
                language="json",
                lines=12,
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Observation")
            obs_display = gr.Textbox(
                label="Current Observation",
                lines=20,
                max_lines=30,
                elem_classes=["log-box"],
                placeholder="Click 'Start Episode' to begin...",
            )

            gr.Markdown("### 🎮 Submit Action")
            action_input = gr.Code(
                label="Action JSON",
                language="json",
                lines=6,
                value='{"action_type": "analyze", "payload": {"filter_level": "WARN"}}',
            )
            step_btn = gr.Button("⚡ Execute Action", variant="secondary", size="lg")

            reward_display = gr.Textbox(
                label="🏆 Reward (shown after submit)",
                lines=10,
                elem_classes=["score-box"],
                placeholder="Submit your final answer to see the reward...",
            )

    # ── Event handlers ─────────────────────────────────────────
    start_btn.click(
        fn=reset_env,
        inputs=[task_selector],
        outputs=[obs_display, reward_display, state_display, action_input],
    )

    step_btn.click(
        fn=submit_action,
        inputs=[action_input],
        outputs=[obs_display, reward_display, state_display],
    )
    app = mount_gradio_app(app, demo, path="/ui")


    @app.post("/reset")
    def reset():
        global _env
        _env = LogAnalyzerEnv(task_id="task1")
        obs = _env.reset()
        return obs.model_dump()


    @app.post("/step")
    def step(action: dict):
        global _env
        act = Action(
            action_type=action["action_type"],
            payload=action.get("payload", {})
        )
        result = _env.step(act)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.score if result.reward else 0.0,
            "done": result.done,
            "info": result.info,
        }





    @app.get("/state")
    def state():
        return {"status": "ok"}

    #template_selector.change(
    #    fn=lambda task, tmpl: get_action_templates(task, tmpl) if tmpl else gr.update(),
    #    inputs=[task_selector, template_selector],
    #    outputs=[action_input],
    #)

    # ── Footer ─────────────────────────────────────────────────
    gr.Markdown(
        """
---
**OpenEnv** · Log Analyzer · HDFS Dataset · 
[GitHub](https://github.com/) · 
Action types: `analyze` | `query` | `submit`
        """
    )
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
