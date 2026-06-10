"""
Base Agent
==========
Abstract base class for all MAIR+ agents.

Every agent follows a perceive → plan → act loop:
    observe(state)   — perceive the current environment/context
    plan(observation) — decide what to do next
    act(plan)        — execute the chosen action and return result

This interface is intentionally minimal so concrete agents can
extend it with domain-specific logic without breaking the contract.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract agent base class.

    Subclasses must implement observe(), plan(), and act().
    The run() method wires them together into a single call.
    """

    def __init__(self, name: str):
        self.name    = name
        self._log    = []      # action history for reflection / debugging

    # ─────────────────────────────────────────────────────────
    # CORE INTERFACE  (must be implemented by subclasses)
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    def observe(self, state: Any) -> Any:
        """
        Perceive the current environment or context.

        Args:
            state: Raw input (e.g. image path, sensor reading, dict).

        Returns:
            Processed observation the agent can reason about.
        """

    @abstractmethod
    def plan(self, observation: Any) -> Any:
        """
        Decide what action to take given an observation.

        Args:
            observation: Output of observe().

        Returns:
            An action descriptor (string key, callable, plan dict, etc.)
        """

    @abstractmethod
    def act(self, action: Any) -> Any:
        """
        Execute the chosen action and return its result.

        Args:
            action: Output of plan().

        Returns:
            Result of the action (e.g. restored image path, metric dict).
        """

    # ─────────────────────────────────────────────────────────
    # CONVENIENCE — full perceive→plan→act cycle
    # ─────────────────────────────────────────────────────────

    def run(self, state: Any) -> Any:
        """
        Execute a full observe → plan → act cycle.

        Logs each step for post-hoc inspection via self.history().

        Args:
            state: Raw input to the agent.

        Returns:
            Result from act().
        """
        observation = self.observe(state)
        action      = self.plan(observation)
        result      = self.act(action)

        self._log.append({
            "agent":       self.name,
            "state":       state,
            "observation": observation,
            "action":      action,
            "result":      result,
        })

        return result

    # ─────────────────────────────────────────────────────────
    # INTROSPECTION
    # ─────────────────────────────────────────────────────────

    def history(self) -> list:
        """Return the full list of logged action records."""
        return self._log

    def last_result(self) -> Any:
        """Return the result of the most recent act() call, or None."""
        return self._log[-1]["result"] if self._log else None

    def reset(self) -> None:
        """Clear the action history."""
        self._log.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
