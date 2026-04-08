"""
env/tasks/task2_anomaly_classification.py

Task 2 — Anomaly Classification (Medium)

The agent sees 50 log lines and must:
  1. Identify which line numbers contain anomalies
  2. Classify each anomaly type
  3. Assign correct severity level

Grader uses weighted partial credit:
  - Detection (found the right lines)  → 40%
  - Anomaly type correct               → 35%
  - Severity correct                   → 25%
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from env.models import Action, Observation, Reward


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

TASK_ID = "task2"
MAX_STEPS = 7

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "task2")

ANOMALY_TYPES = [
    "DATA_SERVE_EXCEPTION",
    "DATANODE_WARNING",
    "ANOMALOUS_BLOCK_ACTIVITY",
]

SEVERITY_LEVELS = ["critical", "warning", "info"]

VALID_SEVERITY_MAP = {
    "DATA_SERVE_EXCEPTION": "warning",
    "DATANODE_WARNING": "warning",
    "ANOMALOUS_BLOCK_ACTIVITY": "info",
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

def make_initial_observation(log_content: str, ground_truth: dict = None) -> Observation:
    # Extract known anomalous block IDs from ground truth so the model
    # can identify ANOMALOUS_BLOCK_ACTIVITY lines (INFO lines that reference
    # blocks flagged in the cluster anomaly database).
    known_anomalous_blocks = []
    if ground_truth:
        for entry in ground_truth.get("anomalies", []):
            if entry.get("anomaly_type") == "ANOMALOUS_BLOCK_ACTIVITY":
                known_anomalous_blocks.extend(entry.get("block_ids", []))

    return Observation(
        task_id=TASK_ID,
        log_content=log_content,
        current_step=0,
        max_steps=MAX_STEPS,
        context={
            "total_lines": 50,
            "anomaly_types": ANOMALY_TYPES,
            "severity_levels": SEVERITY_LEVELS,
            "known_anomalous_block_ids": known_anomalous_blocks,
            "hint": (
                "Identify anomalous lines. For each, provide: "
                "line_number, anomaly_type, and severity. "
                "Anomaly types: DATA_SERVE_EXCEPTION, DATANODE_WARNING, "
                "ANOMALOUS_BLOCK_ACTIVITY. "
                "Severity levels: critical, warning, info. "
                "IMPORTANT: Any line referencing a block ID from "
                "'known_anomalous_block_ids' is anomalous "
                "(type=ANOMALOUS_BLOCK_ACTIVITY, severity=info), "
                "even if the line level is INFO."
            ),
        },
        feedback="Episode started. Analyze the 50 log lines and identify anomalies.",
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
    Filter logs by level, component, keyword, or line range.
    payload examples:
      {"filter_level": "WARN"}
      {"filter_component": "DataXceiver"}
      {"filter_keyword": "exception"}
      {"line_range": [1, 25]}
    """
    lines = log_content.strip().split("\n")
    filtered = []

    filter_level = payload.get("filter_level", "").upper()
    filter_component = payload.get("filter_component", "").lower()
    filter_keyword = payload.get("filter_keyword", "").lower()
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

    elif line_range and len(line_range) == 2:
        start, end = line_range
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
        context={"anomaly_types": ANOMALY_TYPES, "severity_levels": SEVERITY_LEVELS},
        feedback=feedback,
        done=False,
    )


def handle_query(
    payload: Dict[str, Any],
    log_content: str,
    current_step: int,
) -> Observation:
    """
    Answer questions about the logs.
    payload: {"question": "How many WARN lines are there?"}
    """
    question = payload.get("question", "").lower()
    lines = log_content.strip().split("\n")

    warn_lines = [l for l in lines if "WARN" in l]
    info_lines = [l for l in lines if "INFO" in l]
    exception_lines = [l for l in lines if "exception" in l.lower()]
    block_lines = [l for l in lines if "blk_" in l]

    if "warn" in question and "how many" in question:
        answer = f"There are {len(warn_lines)} WARN lines in the log."
    elif "info" in question and "how many" in question:
        answer = f"There are {len(info_lines)} INFO lines in the log."
    elif "exception" in question:
        answer = f"There are {len(exception_lines)} lines containing exceptions."
    elif "block" in question or "blk" in question:
        answer = f"There are {len(block_lines)} lines referencing block IDs."
    elif "component" in question or "components" in question:
        components = set()
        for line in lines:
            if "dfs." in line:
                parts = line.split("dfs.")
                if len(parts) > 1:
                    comp = "dfs." + parts[1].split(":")[0].strip()
                    components.add(comp)
        answer = f"Components seen: {', '.join(sorted(components))}."
    else:
        answer = (
            f"Log summary: {len(lines)} total lines, "
            f"{len(warn_lines)} WARN, {len(info_lines)} INFO, "
            f"{len(exception_lines)} exception lines."
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

def _detection_score(
    predicted_lines: List[int],
    actual_lines: List[int],
) -> Dict[str, float]:
    """F1 score on detected line numbers."""
    predicted_set = set(predicted_lines)
    actual_set = set(actual_lines)

    tp = len(predicted_set & actual_set)
    precision = tp / len(predicted_set) if predicted_set else 0.0
    recall = tp / len(actual_set) if actual_set else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _classification_score(
    predicted: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
) -> float:
    """
    For each correctly detected line, check if anomaly_type matches.
    Score = correct_types / total_actual_anomalies
    """
    actual_map = {a["line_number"]: a["anomaly_type"] for a in actual}
    predicted_map = {p["line_number"]: p.get("anomaly_type", "") for p in predicted}

    correct = 0
    for line_num, actual_type in actual_map.items():
        predicted_type = predicted_map.get(line_num, "")
        if predicted_type.upper() == actual_type.upper():
            correct += 1

    return round(correct / len(actual_map), 4) if actual_map else 0.0


def _severity_score(
    predicted: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
) -> float:
    """
    For each correctly detected line, check if severity matches.
    Score = correct_severities / total_actual_anomalies
    """
    actual_map = {a["line_number"]: a["severity"] for a in actual}
    predicted_map = {p["line_number"]: p.get("severity", "") for p in predicted}

    correct = 0
    for line_num, actual_severity in actual_map.items():
        predicted_severity = predicted_map.get(line_num, "")
        if predicted_severity.lower() == actual_severity.lower():
            correct += 1

    return round(correct / len(actual_map), 4) if actual_map else 0.0


def grade(payload: Dict[str, Any], ground_truth: Dict[str, Any]) -> Reward:
    """
    Grade the agent's submitted answer for task 2.

    Expected payload:
    {
      "anomalies": [
        {"line_number": 7, "anomaly_type": "DATA_SERVE_EXCEPTION", "severity": "warning"},
        ...
      ]
    }
    """
    predicted = payload.get("anomalies", [])
    actual = ground_truth.get("anomalies", [])

    # Validate
    if not isinstance(predicted, list):
        return Reward(
            score=0.0,
            partial_scores={"detection": 0.0, "anomaly_type": 0.0, "severity": 0.0},
            feedback="Invalid submission: 'anomalies' must be a list.",
            penalties=0.1,
        )

    if len(predicted) == 0:
        return Reward(
            score=0.0,
            partial_scores={"detection": 0.0, "anomaly_type": 0.0, "severity": 0.0},
            feedback="Empty submission. No anomalies provided.",
            penalties=0.0,
        )

    # Compute scores
    actual_lines = [a["line_number"] for a in actual]
    predicted_lines = [p.get("line_number", -1) for p in predicted]

    detection = _detection_score(predicted_lines, actual_lines)
    classification = _classification_score(predicted, actual)
    severity = _severity_score(predicted, actual)

    # Weighted final score
    weights = {"detection": 0.40, "anomaly_type": 0.35, "severity": 0.25}
    final_score = (
        detection["f1"] * weights["detection"]
        + classification * weights["anomaly_type"]
        + severity * weights["severity"]
    )

    # Feedback
    predicted_set = set(predicted_lines)
    actual_set = set(actual_lines)
    missed = sorted(actual_set - predicted_set)
    false_pos = sorted(predicted_set - actual_set)

    feedback_parts = [f"Final score: {final_score:.2f}."]
    feedback_parts.append(
        f"Detection F1: {detection['f1']:.2f} "
        f"(precision={detection['precision']:.2f}, recall={detection['recall']:.2f})."
    )
    feedback_parts.append(f"Anomaly type accuracy: {classification:.2f}.")
    feedback_parts.append(f"Severity accuracy: {severity:.2f}.")
    if missed:
        feedback_parts.append(f"Missed anomaly lines: {missed}.")
    if false_pos:
        feedback_parts.append(f"False positive lines: {false_pos}.")

    return Reward(
        score=round(final_score, 4),
        partial_scores={
            "detection_f1": detection["f1"],
            "detection_precision": detection["precision"],
            "detection_recall": detection["recall"],
            "anomaly_type_accuracy": classification,
            "severity_accuracy": severity,
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