---
title: Log Analyzer - OpenEnv
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
  - log-analysis
  - devops
---

# 🔍 Log Analyzer — OpenEnv Environment

> Built as a production-grade OpenEnv environment for real-world SRE workflows.

A real-world HDFS server log analysis environment where an AI agent acts as a
DevOps/SRE engineer. The agent analyzes server logs to detect warnings, classify
anomalies, and diagnose root causes — tasks humans do every day.

Built on the [HDFS_2k dataset](https://github.com/logpai/loghub) from LogHub.

---

## 🎯 Why Log Analysis?

Server logs are among the most structured real-world data. Every production system
generates them. Analyzing them well is a core SRE skill — and it scales naturally
from trivial (find the errors) to genuinely hard (reconstruct a cascading failure).

This environment makes that progression explicit and measurable.

---

## 📋 Tasks

### Task 1 — Warning Detection *(Easy, 5 steps)*

**Objective:** Given 20 HDFS log lines, identify all line numbers (1-indexed)
that contain `WARN`-level events.

**Expected output:**

```json
{"warn_lines": [11, 12, 13, 14]}
```

**Grader:** F1 score

---

### Task 2 — Anomaly Classification *(Medium, 7 steps)*

**Objective:** Detect anomalies and classify type + severity.

**Expected output:**

```json
{
  "anomalies": [
    {"line_number": 7, "anomaly_type": "DATA_SERVE_EXCEPTION", "severity": "warning"}
  ]
}
```

**Grader:** Detection + Type + Severity

---

### Task 3 — Root Cause Analysis *(Hard, 10 steps)*

**Objective:** Full RCA (root cause, timeline, components, remediation)

---

## 🤖 Agent Interaction Flow

Each episode works like:

1. `reset()` → get logs
2. `analyze` / `query` → explore
3. `submit` → final answer
4. reward returned

Example:

```json
{"action_type": "analyze", "payload": {"filter_level": "WARN"}}
{"action_type": "submit", "payload": {"warn_lines": [11,12,...]}}
```

---

## 🔧 Action Space

```python
action_type: "analyze" | "query" | "submit"
payload: dict
```

---

## 📊 Observation Space

Includes:

* log_content
* step count
* feedback
* done flag

---

## 🏆 Reward Function

* Score: 0.0–1.0
* Partial scores
* Penalties for bad actions

---

## 🚀 Setup & Usage

```bash
git clone https://github.com/your-org/log-analyzer-openenv
cd log-analyzer-openenv
pip install -r requirements.txt
```

---

## ▶️ Run UI

```bash
python -m server.app
```

Open → http://localhost:7860

---

## 🤖 Run Baseline

```bash
python inference.py
```

---

## 🐳 Docker

```bash
docker build -t log-analyzer .
docker run -p 7860:7860 log-analyzer
```

---

## 📈 Baseline Scores

| Task        | Score    |
| ----------- | -------- |
| Task 1      | 0.95     |
| Task 2      | 0.95     |
| Task 3      | 0.95     |
| **Average** | **0.95** |

---

## ✅ OpenEnv Validation

```bash
openenv validate
# [OK] Ready for multi-mode deployment
```

---

## 📁 Project Structure

```
log-analyzer/
├── server/app.py
├── env/
├── inference.py
├── openenv.yaml
├── Dockerfile
└── README.md
```

---

## 🗂️ Dataset

HDFS_2k dataset from LogHub.

---

## 👨‍💻 Author

Deepanshu Giri

---

## 🏁 Submission Ready

✔ OpenEnv validated
✔ Docker working
✔ HF Spaces ready
✔ Baseline reproducible
✔ Structured logs compliant

---
