"""
env/tasks/task1_warning_detection.py

Task 1 — Warning Detection (Easy)

The agent sees 20 log lines and must identify which line numbers
(1-indexed) contain WARN-level events.

Grader uses F1 score for partial credit:
  - Perfect score  → F1 = 1.0
  - Partial credit → 0.0 < F1 < 1.0
  - All wrong      → F1 = 0.0
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from env.models import Action, Observation, Reward


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

TASK_ID = "task1"
MAX_STEPS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "task1")


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

def load_task() -> Dict[str, Any]:
    """Load logs and ground truth from disk."""
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

def make_initial_observation(log_content: str, total_lines: int) -> Observation:
    return Observation(
        task_id=TASK_ID,
        log_content=log_content,
        current_step=0,
        max_steps=MAX_STEPS,
        context={
            "total_lines": total_lines,
            "hint": (
                "Identify all line numbers (1-indexed) that contain "
                "WARN-level log events. Submit a list of line numbers."
            ),
        },
        feedback="Episode started. Read the logs and find all WARN lines.",
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
    Filter logs by level or line range.
    payload: {"filter_level": "WARN"} or {"line_range": [1, 10]}
    """
    lines = log_content.strip().split("\n")
    filtered = []

    filter_level = payload.get("filter_level", "").upper()
    line_range = payload.get("line_range", None)

    if filter_level:
        for line in lines:
            if filter_level in line:
                filtered.append(line)
        feedback = f"Filtered to {len(filtered)} lines containing '{filter_level}'."

    elif line_range and len(line_range) == 2:
        start, end = line_range
        filtered = lines[start - 1: end]
        feedback = f"Showing lines {start}–{end} ({len(filtered)} lines)."

    else:
        filtered = lines
        feedback = "No valid filter provided. Showing all lines."

    return Observation(
        task_id=TASK_ID,
        log_content="\n".join(filtered),
        current_step=current_step,
        max_steps=MAX_STEPS,
        context={"hint": "Use this filtered view to identify WARN line numbers."},
        feedback=feedback,
        done=False,
    )


def handle_query(
    payload: Dict[str, Any],
    log_content: str,
    current_step: int,
) -> Observation:
    """
    Answer a simple question about the logs.
    payload: {"question": "How many WARN lines are there?"}
    """
    question = payload.get("question", "").lower()
    lines = log_content.strip().split("\n")

    warn_lines = [l for l in lines if "WARN" in l]
    info_lines = [l for l in lines if "INFO" in l]

    if "how many" in question and "warn" in question:
        answer = f"There are {len(warn_lines)} WARN lines in the log."
    elif "how many" in question and "info" in question:
        answer = f"There are {len(info_lines)} INFO lines in the log."
    elif "how many" in question and ("line" in question or "total" in question):
        answer = f"There are {len(lines)} total lines in the log."
    elif "which" in question and "warn" in question:
        nums = []
        for line in warn_lines:
            try:
                num = int(line.split(":")[0].replace("Line", "").strip())
                nums.append(str(num))
            except ValueError:
                pass
        answer = f"WARN lines appear at: {', '.join(nums) if nums else 'unknown'}."
    else:
        answer = (
            f"Log summary: {len(lines)} total lines, "
            f"{len(warn_lines)} WARN, {len(info_lines)} INFO."
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

def _f1_score(predicted: List[int], actual: List[int]) -> Dict[str, float]:
    """Compute precision, recall, F1 between two sets of line numbers."""
    predicted_set = set(predicted)
    actual_set = set(actual)

    true_positives = len(predicted_set & actual_set)

    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(actual_set) if actual_set else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def grade(payload: Dict[str, Any], ground_truth: Dict[str, Any]) -> Reward:
    """
    Grade the agent's submitted answer for task 1.

    Expected payload: {"warn_lines": [11, 12, 13, ...]}
    """
    predicted = payload.get("warn_lines", [])
    actual = ground_truth.get("warn_lines", [])

    # Validate types
    if not isinstance(predicted, list):
        return Reward(
            score=0.0,
            partial_scores={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            feedback="Invalid submission: 'warn_lines' must be a list of integers.",
            penalties=0.1,
        )

    # Convert to int safely
    try:
        predicted = [int(x) for x in predicted]
    except (ValueError, TypeError):
        return Reward(
            score=0.0,
            partial_scores={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            feedback="Invalid submission: line numbers must be integers.",
            penalties=0.1,
        )

    scores = _f1_score(predicted, actual)
    f1 = scores["f1"]

    # Build human-readable feedback
    predicted_set = set(predicted)
    actual_set = set(actual)
    missed = sorted(actual_set - predicted_set)
    false_pos = sorted(predicted_set - actual_set)

    feedback_parts = [f"Score: {f1:.2f} (F1)."]
    if missed:
        feedback_parts.append(f"Missed WARN lines: {missed}.")
    if false_pos:
        feedback_parts.append(f"False positives (not WARN): {false_pos}.")
    if not missed and not false_pos:
        feedback_parts.append("Perfect! All WARN lines correctly identified.")

    return Reward(
        score=round(f1, 4),
        partial_scores=scores,
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
    """
    Process one action and return observation + optional reward.
    Called by LogAnalyzerEnv.step().
    """
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
        # Unknown action type — penalize
        obs = Observation(
            task_id=TASK_ID,
            log_content=log_content,
            current_step=current_step,
            max_steps=MAX_STEPS,
            context={},
            feedback=f"Unknown action_type '{action.action_type}'. Use: analyze, query, submit.",
            done=False,
        )
        return {"observation": obs, "reward": None, "done": False, "penalty": 0.1}