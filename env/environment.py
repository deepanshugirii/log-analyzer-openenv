"""
env/environment.py

LogAnalyzerEnv — OpenEnv-compliant environment for HDFS log analysis.

Three tasks:
  task1 — Warning Detection   (Easy,   5 steps)
  task2 — Anomaly Classification (Medium, 7 steps)
  task3 — Root Cause Analysis (Hard,  10 steps)

API:
  env = LogAnalyzerEnv(task_id="task1")
  obs = env.reset()
  result = env.step(action)
  state = env.state()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from env.models import Action, Observation, Reward, StepResult

# ── Task modules ──────────────────────────────────────────────
from env.tasks import task1_warning_detection as task1
from env.tasks import task2_anomaly_classification as task2
from env.tasks import task3_root_cause_analysis as task3

TASK_MODULES = {
    "task1": task1,
    "task2": task2,
    "task3": task3,
}

TASK_MAX_STEPS = {
    "task1": 5,
    "task2": 7,
    "task3": 10,
}


class LogAnalyzerEnv:
    """
    OpenEnv-compliant environment for HDFS server log analysis.

    Parameters
    ----------
    task_id : str
        One of "task1", "task2", "task3".
    """

    def __init__(self, task_id: str = "task1") -> None:
        if task_id not in TASK_MODULES:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {list(TASK_MODULES.keys())}"
            )
        self.task_id = task_id
        self._module = TASK_MODULES[task_id]
        self._max_steps = TASK_MAX_STEPS[task_id]

        # Episode state
        self._log_content: str = ""
        self._ground_truth: Dict[str, Any] = {}
        self._current_step: int = 0
        self._done: bool = False
        self._penalties: float = 0.0
        self._action_history: List[str] = []
        self._last_observation: Optional[Observation] = None

    # ──────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment and return the initial observation.
        Loads logs and ground truth from disk.
        """
        data = self._module.load_task()
        self._log_content = data["log_content"]
        self._ground_truth = data["ground_truth"]
        self._current_step = 0
        self._done = False
        self._penalties = 0.0
        self._action_history = []

        if self.task_id == "task1":
            total = self._ground_truth.get("total_lines", 20)
            obs = task1.make_initial_observation(self._log_content, total)
        elif self.task_id == "task2":
            obs = task2.make_initial_observation(self._log_content, self._ground_truth)
        else:
            obs = task3.make_initial_observation(self._log_content)

        self._last_observation = obs
        return obs

    # ──────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return (observation, reward, done, info).

        Raises
        ------
        RuntimeError if reset() has not been called yet.
        """
        if self._last_observation is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            return StepResult(
                observation=self._last_observation,
                reward=None,
                done=True,
                info={"warning": "Episode already finished. Call reset()."},
            )

        self._current_step += 1
        info: Dict[str, Any] = {}
        penalty = 0.0

        # ── Penalty: repeated identical action ────────────────
        action_sig = f"{action.action_type}:{json.dumps(action.payload, sort_keys=True)}"
        if action_sig in self._action_history:
            penalty += 0.15
            info["penalty_reason"] = "Repeated identical action."
        self._action_history.append(action_sig)

        # ── Penalty: submit on step 0 (no exploration) ────────
        if action.action_type == "submit" and self._current_step == 1:
            penalty += 0.05
            info["penalty_reason"] = info.get("penalty_reason", "") + " Submitted without exploring."

        self._penalties += penalty

        # ── Dispatch to task module ───────────────────────────
        result = self._module.step(
            action=action,
            log_content=self._log_content,
            ground_truth=self._ground_truth,
            current_step=self._current_step,
        )

        obs: Observation = result["observation"]
        reward: Optional[Reward] = result.get("reward")
        done: bool = result.get("done", False)

        # Apply any per-action penalty from the task module
        module_penalty = result.get("penalty", 0.0)
        self._penalties += module_penalty
        if module_penalty > 0:
            info["penalty_reason"] = info.get("penalty_reason", "") + f" Module penalty: {module_penalty}."

        # ── Max steps enforcement ─────────────────────────────
        if self._current_step >= self._max_steps and not done:
            done = True
            obs = Observation(
                task_id=self.task_id,
                log_content="",
                current_step=self._current_step,
                max_steps=self._max_steps,
                context={},
                feedback=(
                    f"Max steps ({self._max_steps}) reached without submitting. "
                    f"Score: 0.0"
                ),
                done=True,
            )
            reward = Reward(
                score=0.0,
                partial_scores={},
                feedback="Episode ended at max steps without a submission.",
                penalties=self._penalties,
            )

        # ── Apply penalties to final reward ───────────────────
        if reward is not None:
            adjusted = max(0.0, reward.score - self._penalties)
            reward = Reward(
                score=round(adjusted, 4),
                partial_scores=reward.partial_scores,
                feedback=reward.feedback + (
                    f" (Penalties deducted: {self._penalties:.2f}.)"
                    if self._penalties > 0 else ""
                ),
                penalties=self._penalties,
            )

        self._done = done
        self._last_observation = obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # ──────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """
        Return current internal state snapshot.
        Useful for debugging, checkpointing, or inspection.
        """
        return {
            "task_id": self.task_id,
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "done": self._done,
            "penalties_accumulated": self._penalties,
            "actions_taken": len(self._action_history),
            "action_history": self._action_history,
            "log_length_chars": len(self._log_content),
            "ground_truth_keys": list(self._ground_truth.keys()),
        }

    # ──────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LogAnalyzerEnv(task_id='{self.task_id}', "
            f"step={self._current_step}/{self._max_steps}, "
            f"done={self._done})"
        )