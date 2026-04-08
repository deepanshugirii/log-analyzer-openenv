"""
prepare_data.py

Run this ONCE to slice the raw HDFS logs into task-specific
data files + ground truth JSONs.

Usage:
    python prepare_data.py \
        --log      data/raw/HDFS_2k.log \
        --struct   data/raw/HDFS_2k_log_structured.csv \
        --labels   data/raw/anomaly_label.csv \
        --out      env/data
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def extract_block_ids(content: str) -> list:
    return re.findall(r"blk_-?\d+", str(content))


def format_log_line(row) -> str:
    """Reconstruct a human-readable log line from structured CSV row."""
    return (
        f"[{row['Date']} {str(row['Time']).zfill(6)}] "
        f"{row['Level']:<5} "
        f"{row['Component']}: "
        f"{row['Content']}"
    )


def save(out_dir: str, logs: list, ground_truth: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "logs.txt"), "w") as f:
        f.write("\n".join(logs))
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  Saved {len(logs)} lines → {out_dir}/logs.txt")
    print(f"  Ground truth       → {out_dir}/ground_truth.json")


# ──────────────────────────────────────────────────────────────
# Task builders
# ──────────────────────────────────────────────────────────────

def build_task1(df: pd.DataFrame, out_dir: str) -> None:
    """
    Task 1 — Warning Detection (Easy)

    Given 20 log lines, identify every line (1-indexed) that
    contains a WARN-level event. WARN = DataNode serving exception.

    Grader: F1-score on predicted vs actual warn line numbers.
    """
    print("\n[Task 1] Building warning detection slice …")

    # Use lines 60-100 where WARN lines cluster naturally
    slice_df = (
        df[(df["LineId"] >= 60) & (df["LineId"] <= 100)]
        .sample(n=20, random_state=42)
        .sort_values("LineId")
        .reset_index(drop=True)
    )

    logs = []
    warn_positions = []  # 1-indexed positions within this slice

    for i, row in slice_df.iterrows():
        line_num = i + 1  # 1-indexed
        logs.append(f"Line {line_num:02d}: {format_log_line(row)}")
        if row["Level"] == "WARN":
            warn_positions.append(line_num)

    ground_truth = {
        "task": "warning_detection",
        "description": (
            "Identify all line numbers (1-indexed) that contain "
            "WARN-level events indicating DataNode serving exceptions."
        ),
        "warn_lines": warn_positions,
        "total_lines": len(logs),
        "total_warns": len(warn_positions),
        "scoring": {
            "method": "f1_score",
            "partial_credit": True,
            "note": "score = F1(predicted_set, actual_set)",
        },
    }

    save(os.path.join(out_dir, "task1"), logs, ground_truth)
    print(f"  WARN lines at positions: {warn_positions}")


def build_task2(df: pd.DataFrame, anomaly_blocks: set, out_dir: str) -> None:
    """
    Task 2 — Anomaly Classification (Medium)

    Given 50 log lines, the agent must:
      1. Identify which lines belong to anomalous block IDs
      2. Classify the anomaly type for each flagged line
      3. Assign a severity level

    Grader: weighted partial credit across detection + classification + severity.
    """
    print("\n[Task 2] Building anomaly classification slice …")

    slice_df = (
        df[(df["LineId"] >= 60) & (df["LineId"] <= 200)]
        .sample(n=50, random_state=42)
        .sort_values("LineId")
        .reset_index(drop=True)
    )

    logs = []
    anomaly_entries = []

    for i, row in slice_df.iterrows():
        line_num = i + 1
        logs.append(f"Line {line_num:02d}: {format_log_line(row)}")

        block_ids = extract_block_ids(row["Content"])
        is_anomaly = any(b in anomaly_blocks for b in block_ids)

        if is_anomaly:
            # Classify based on component + content pattern
            if row["Level"] == "WARN" and "Got exception" in str(row["Content"]):
                anomaly_type = "DATA_SERVE_EXCEPTION"
                severity = "warning"
            elif row["Level"] == "WARN":
                anomaly_type = "DATANODE_WARNING"
                severity = "warning"
            else:
                anomaly_type = "ANOMALOUS_BLOCK_ACTIVITY"
                severity = "info"

            anomaly_entries.append({
                "line_number": line_num,
                "block_ids": block_ids,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "component": row["Component"],
            })

    # Also flag all WARN lines as anomalies (separate from block-based)
    warn_entries = []
    for i, row in slice_df.iterrows():
        line_num = i + 1
        if row["Level"] == "WARN":
            block_ids = extract_block_ids(row["Content"])
            already = any(e["line_number"] == line_num for e in anomaly_entries)
            if not already:
                warn_entries.append({
                    "line_number": line_num,
                    "block_ids": block_ids,
                    "anomaly_type": "DATA_SERVE_EXCEPTION",
                    "severity": "warning",
                    "component": row["Component"],
                })

    all_anomalies = anomaly_entries + warn_entries
    # deduplicate by line_number
    seen = set()
    deduped = []
    for e in sorted(all_anomalies, key=lambda x: x["line_number"]):
        if e["line_number"] not in seen:
            seen.add(e["line_number"])
            deduped.append(e)

    ground_truth = {
        "task": "anomaly_classification",
        "description": (
            "Identify anomalous lines, classify their anomaly type, "
            "and assign severity (critical / warning / info)."
        ),
        "anomalies": deduped,
        "total_lines": len(logs),
        "anomaly_types_reference": {
            "DATA_SERVE_EXCEPTION": "DataNode threw exception while serving a block to a client",
            "DATANODE_WARNING": "DataNode issued a non-exception warning",
            "ANOMALOUS_BLOCK_ACTIVITY": "Block ID linked to a known anomaly trace",
        },
        "severity_levels": ["critical", "warning", "info"],
        "scoring": {
            "method": "weighted_partial_credit",
            "weights": {
                "detection": 0.40,
                "anomaly_type": 0.35,
                "severity": 0.25,
            },
        },
    }

    save(os.path.join(out_dir, "task2"), logs, ground_truth)
    print(f"  Anomalies found: {len(deduped)} (across {len(logs)} lines)")


def build_task3(df: pd.DataFrame, anomaly_blocks: set, out_dir: str) -> None:
    """
    Task 3 — Root Cause Analysis (Hard)

    Given 200 log lines covering a failure window, the agent must:
      1. Identify the root cause component
      2. Reconstruct the failure timeline (key events in order)
      3. List affected block IDs
      4. Propose remediation steps

    Grader: rubric across 4 dimensions (0.35 / 0.25 / 0.20 / 0.20).
    """
    print("\n[Task 3] Building root cause analysis slice …")

    slice_df = (
        df[(df["LineId"] >= 1) & (df["LineId"] <= 250)]
        .head(200)
        .reset_index(drop=True)
    )

    logs = []
    warn_lines = []
    anomaly_block_ids_seen = set()

    for i, row in slice_df.iterrows():
        line_num = i + 1
        logs.append(f"Line {line_num:03d}: {format_log_line(row)}")

        if row["Level"] == "WARN":
            warn_lines.append({
                "line_number": line_num,
                "time": str(row["Time"]).zfill(6),
                "content": row["Content"],
            })

        for b in extract_block_ids(row["Content"]):
            if b in anomaly_blocks:
                anomaly_block_ids_seen.add(b)

    # Build timeline from WARN events
    timeline = []
    if warn_lines:
        timeline.append({
            "line_number": warn_lines[0]["line_number"],
            "time": warn_lines[0]["time"],
            "event": "First DataNode serving exception observed",
        })
        if len(warn_lines) >= 3:
            mid = warn_lines[len(warn_lines) // 2]
            timeline.append({
                "line_number": mid["line_number"],
                "time": mid["time"],
                "event": "Repeated serving exceptions — multiple blocks affected",
            })
        timeline.append({
            "line_number": warn_lines[-1]["line_number"],
            "time": warn_lines[-1]["time"],
            "event": "Last exception in window — failure pattern established",
        })

    ground_truth = {
        "task": "root_cause_analysis",
        "description": (
            "Analyze the 200-line log window and identify the root cause, "
            "timeline of key failure events, affected components, and remediation."
        ),
        "root_cause": {
            "component": "dfs.DataNode$DataXceiver",
            "description": (
                "DataXceiver threads are failing to serve blocks to clients. "
                "Multiple DataNodes are throwing exceptions during block transfer, "
                "indicating network issues or DataNode instability."
            ),
            "evidence_lines": [w["line_number"] for w in warn_lines],
        },
        "timeline": timeline,
        "affected_components": [
            "dfs.DataNode$DataXceiver",
            "dfs.DataNode$PacketResponder",
        ],
        "affected_block_ids": list(anomaly_block_ids_seen)[:10],
        "remediation_steps": [
            "Check network connectivity between DataNodes and clients",
            "Inspect DataNode logs for underlying IOException stack traces",
            "Restart affected DataNode services if exceptions persist",
            "Verify HDFS block replication factor is maintained",
            "Monitor DataNode heap usage — OOM can cause serving failures",
        ],
        "scoring": {
            "method": "rubric",
            "weights": {
                "root_cause_component": 0.35,
                "timeline_key_events": 0.25,
                "affected_components": 0.20,
                "remediation_valid_steps": 0.20,
            },
            "rubric_notes": {
                "root_cause_component": "Must identify DataXceiver or DataNode as root cause",
                "timeline_key_events": "Partial credit for each key event identified",
                "affected_components": "Partial credit per correct component",
                "remediation_valid_steps": "0.04 per valid step (max 5 steps)",
            },
        },
    }

    save(os.path.join(out_dir, "task3"), logs, ground_truth)
    print(f"  Timeline events: {len(timeline)}")
    print(f"  Anomalous blocks seen: {len(anomaly_block_ids_seen)}")
    print(f"  WARN lines in window: {len(warn_lines)}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare HDFS log data for OpenEnv tasks")
    parser.add_argument("--log",    required=True, help="Path to HDFS_2k.log")
    parser.add_argument("--struct", required=True, help="Path to HDFS_2k_log_structured.csv")
    parser.add_argument("--labels", required=True, help="Path to anomaly_label.csv")
    parser.add_argument("--out",    default="env/data", help="Output directory (default: env/data)")
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(args.struct)
    labels = pd.read_csv(args.labels)

    anomaly_blocks = set(labels[labels["Label"] == "Anomaly"]["BlockId"].tolist())
    print(f"  Structured CSV: {len(df)} rows")
    print(f"  Anomaly blocks: {len(anomaly_blocks):,}")

    build_task1(df, args.out)
    build_task2(df, anomaly_blocks, args.out)
    build_task3(df, anomaly_blocks, args.out)

    print("\n✅ All tasks prepared successfully.")
    print(f"   Output: {args.out}/task1/, task2/, task3/")
    print("   Each folder contains: logs.txt + ground_truth.json")


if __name__ == "__main__":
    main()