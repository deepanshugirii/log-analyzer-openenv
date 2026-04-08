"""
env/models.py

Typed Pydantic models for the Log Analyzer OpenEnv environment.
All three core interfaces: Observation, Action, Reward.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# OBSERVATION — what the agent sees after every step
# ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    """Which task is running: 'task1', 'task2', or 'task3'"""

    log_content: str
    """
    The log text shown to the agent.
    On reset()      → full logs.txt content
    After 'analyze' → filtered subset of lines
    After 'query'   → answer to the agent's question
    After 'submit'  → empty string (episode over)
    """

    current_step: int
    """How many steps the agent has taken so far (0 on reset)"""

    max_steps: int
    """Episode ends when current_step reaches this value"""

    context: Dict[str, Any]
    """
    Task-specific background info. Examples:
      task1 → {"total_lines": 20, "log_level_hint": "look for WARN"}
      task2 → {"anomaly_types": [...], "severity_levels": [...]}
      task3 → {"components": [...], "time_window": "..."}
    """

    feedback: str
    """
    Human-readable result of the last action.
    On reset() → "Episode started. Analyze the logs carefully."
    After step → e.g. "Filtered to 5 WARN lines." or "Invalid action format."
    """

    done: bool
    """True when episode is over (submit called or max_steps reached)"""


# ──────────────────────────────────────────────────────────────
# ACTION — what the agent submits each step
# ──────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: Literal["analyze", "query", "submit"]
    """
    Three possible actions:

    "analyze" — filter the logs by level, component, or line range.
                costs 1 step. returns filtered log view.
      payload examples:
        {"filter_level": "WARN"}
        {"filter_component": "DataXceiver"}
        {"line_range": [1, 50]}

    "query"   — ask a specific question about the logs.
                costs 1 step. returns a direct answer string.
      payload examples:
        {"question": "How many WARN lines are there?"}
        {"question": "Which block IDs appear most often?"}

    "submit"  — final answer. triggers the grader. ends the episode.
      payload varies by task:

        task1 → {"warn_lines": [11, 12, 13, ...]}

        task2 → {"anomalies": [
                    {"line_number": 7,
                     "anomaly_type": "DATA_SERVE_EXCEPTION",
                     "severity": "warning"},
                    ...
                 ]}

        task3 → {"root_cause": "dfs.DataNode$DataXceiver",
                 "timeline": [
                    {"line_number": 78, "event": "First exception observed"},
                    ...
                 ],
                 "affected_components": ["dfs.DataNode$DataXceiver"],
                 "remediation": ["Check network", "Restart DataNode", ...]}
    """

    payload: Dict[str, Any]
    """Action-specific data. See action_type docstring for schema per action."""


# ──────────────────────────────────────────────────────────────
# REWARD — returned by the grader after "submit"
# ──────────────────────────────────────────────────────────────

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    """
    Final episode score between 0.0 and 1.0.
    This is the number that goes on the leaderboard.
    """

    partial_scores: Dict[str, float]
    """
    Breakdown of how the score was computed. Examples:
      task1 → {"precision": 0.8, "recall": 0.9, "f1": 0.85}
      task2 → {"detection": 0.4, "anomaly_type": 0.3, "severity": 0.2}
      task3 → {"root_cause": 0.35, "timeline": 0.15,
                "affected_components": 0.2, "remediation": 0.16}
    """

    feedback: str
    """
    Human-readable explanation of the score.
    e.g. "Found 9/10 WARN lines. Missed line 14. No false positives."
    """

    penalties: float = 0.0
    """
    Total penalties deducted during the episode (always >= 0).
    Penalties come from: invalid actions, repeated identical actions, 
    submitting before step 1.
    Final score = raw_score - penalties (clamped to 0.0 minimum).
    """


# ──────────────────────────────────────────────────────────────
# STEP RESULT — convenience wrapper returned by env.step()
# ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: Optional[Reward] = None
    """Only populated when action_type == 'submit'"""
    done: bool
    info: Dict[str, Any] = {}
    """Extra debug info: {"penalty_reason": "repeated action", ...}"""