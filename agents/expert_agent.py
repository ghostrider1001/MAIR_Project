"""
Expert Agent
============
A thin agent wrapper around a single restoration expert function.

Each ExpertAgent:
  - observe()  — loads and validates the input image
  - plan()     — decides the output path for its restoration
  - act()      — calls the underlying expert function and returns the result

This enables each expert to be treated as an autonomous agent with its own
identity, history, and capability metadata, rather than a bare function.
"""

import os
from typing import Callable, Optional, Any

from agents.base_agent import BaseAgent


class ExpertAgent(BaseAgent):
    """
    Wraps a single restoration expert function as an autonomous agent.

    Args:
        key         : registry key for this expert (e.g. 'restormer_deblur')
        name        : human-readable name (e.g. 'Restormer Motion Deblurring')
        fn          : the callable that performs restoration
                      signature: fn(input_path: str) -> str | None
        task        : primary degradation type this expert targets
        handles     : list of degradation types this expert improves
        speed       : 'very_fast' | 'fast' | 'medium' | 'slow'
        quality     : 'low' | 'medium' | 'high' | 'very_high'
        description : one-line capability summary
    """

    def __init__(
        self,
        key:         str,
        name:        str,
        fn:          Callable,
        task:        str,
        handles:     list,
        speed:       str       = "medium",
        quality:     str       = "high",
        description: str       = "",
    ):
        super().__init__(name=name)
        self.key         = key
        self.fn          = fn
        self.task        = task
        self.handles     = handles
        self.speed       = speed
        self.quality     = quality
        self.description = description

    # ─────────────────────────────────────────────────────────
    # AGENT INTERFACE
    # ─────────────────────────────────────────────────────────

    def observe(self, state: Any) -> dict:
        """
        Validate and record the input image path.

        Returns a dict with 'input_path' and 'valid' flag.
        """
        input_path = state if isinstance(state, str) else state.get("input_path", "")
        valid      = os.path.exists(input_path)

        if not valid:
            print(f"[{self.name}] WARNING: Input not found — {input_path}")

        return {"input_path": input_path, "valid": valid}

    def plan(self, observation: dict) -> dict:
        """
        Decide whether to run the expert based on input validity.

        Returns a dict with 'should_run' flag and the 'input_path'.
        """
        return {
            "should_run": observation["valid"],
            "input_path": observation["input_path"],
        }

    def act(self, action: dict) -> Optional[str]:
        """
        Call the expert function and return the output path (or None).
        """
        if not action["should_run"]:
            print(f"[{self.name}] Skipping — invalid input.")
            return None

        try:
            return self.fn(action["input_path"])
        except Exception as e:
            print(f"[{self.name}] Exception during restoration: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # CONVENIENCE
    # ─────────────────────────────────────────────────────────

    def restore(self, input_path: str) -> Optional[str]:
        """Shortcut: run the full observe→plan→act cycle."""
        return self.run(input_path)

    def can_handle(self, degradation: str) -> bool:
        """Return True if this agent handles the given degradation type."""
        return degradation in self.handles

    def capability_summary(self) -> str:
        return (
            f"  [{self.key}]  {self.name}\n"
            f"    task={self.task}  handles={self.handles}\n"
            f"    speed={self.speed}  quality={self.quality}\n"
            f"    {self.description}"
        )

    def __repr__(self) -> str:
        return (
            f"ExpertAgent(key={self.key!r}, task={self.task!r}, "
            f"quality={self.quality!r}, speed={self.speed!r})"
        )
