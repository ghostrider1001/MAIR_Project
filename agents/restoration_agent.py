"""
Restoration Agent
=================
The top-level orchestrating agent for MAIR+.

This agent coordinates the full pipeline:
  observe()  — analyse image degradation (runs the detector)
  plan()     — select and rank expert agents from the registry
  act()      — execute the restoration loop with reflection

The RestorationAgent wraps the scheduler logic in an agent interface,
giving it a persistent identity, action history, and introspection API.

Usage:
    agent = RestorationAgent()
    output_path = agent.restore("path/to/degraded.jpg")
    print(agent.last_report())
"""

import os
from typing import Any, Optional

from agents.base_agent    import BaseAgent
from agents.expert_agent  import ExpertAgent
from core.degradation_detector  import detect_degradation
from core.restoration_context   import RestorationContext
from core.tool_registry         import REGISTRY
from scheduler.expert_selector  import select_experts, print_ranking
from scheduler.reflection_engine import reflect, explain, ACCEPT, ESCALATE
from evaluation.quality_evaluator import evaluate_quality


class RestorationAgent(BaseAgent):
    """
    Top-level MAIR+ orchestrating agent.

    Internally builds ExpertAgent instances from the Tool Registry,
    then runs the agentic select → execute → reflect loop.

    Args:
        max_attempts  : max expert invocations per image (default 2)
        verbose       : print full reasoning trace (default True)
        chain_mode    : if True, chain multiple experts for mixed degradations
    """

    def __init__(
        self,
        max_attempts: int  = 2,
        verbose:      bool = True,
        chain_mode:   bool = False,
    ):
        super().__init__(name="RestorationAgent")
        self.max_attempts = max_attempts
        self.verbose      = verbose
        self.chain_mode   = chain_mode

        # Build ExpertAgent instances from the registry
        self._experts: dict[str, ExpertAgent] = {
            key: ExpertAgent(
                key=key,
                name=entry["name"],
                fn=entry["fn"],
                task=entry["task"],
                handles=entry["handles"],
                speed=entry["speed"],
                quality=entry["quality"],
                description=entry["description"],
            )
            for key, entry in REGISTRY.items()
        }

        self._last_context: Optional[RestorationContext] = None

    # ─────────────────────────────────────────────────────────
    # AGENT INTERFACE
    # ─────────────────────────────────────────────────────────

    def observe(self, state: Any) -> dict:
        """
        Perceive the environment: detect degradation in the input image.

        Args:
            state: path to the degraded input image (str)

        Returns:
            degradation_result dict from detect_degradation()
        """
        input_path = state if isinstance(state, str) else state.get("input_path", "")

        if not os.path.exists(input_path):
            print(f"[RestorationAgent] ERROR: File not found — {input_path}")
            return {"input_path": input_path, "valid": False, "degradation": None}

        degradation = detect_degradation(input_path)
        return {
            "input_path":  input_path,
            "valid":       True,
            "degradation": degradation,
        }

    def plan(self, observation: dict) -> dict:
        """
        Reason about the observation and select the expert plan.

        Args:
            observation: output of observe()

        Returns:
            dict with 'input_path', 'ranked' expert list, 'context'
        """
        if not observation.get("valid"):
            return {"valid": False, "ranked": [], "input_path": ""}

        input_path  = observation["input_path"]
        degradation = observation["degradation"]
        ranked      = select_experts(degradation)

        if self.verbose:
            print_ranking(degradation)

        ctx = RestorationContext(
            original_path=input_path,
            degradation_result=degradation,
            expert_plan=ranked,
            max_attempts=self.max_attempts,
        )

        return {
            "valid":      True,
            "input_path": input_path,
            "ranked":     ranked,
            "context":    ctx,
        }

    def act(self, action: dict) -> Optional[str]:
        """
        Execute the restoration loop: run experts, evaluate, reflect.

        Args:
            action: output of plan()

        Returns:
            Path to the best restored image, or None if all experts failed.
        """
        if not action.get("valid"):
            return None

        from scheduler.scheduler import run_scheduler
        
        # We delegate to the unified scheduler to prevent logic drift
        # between the agent path and the run_pipeline path.
        output_path = run_scheduler(
            input_path=action["input_path"],
            max_attempts=self.max_attempts,
            verbose=self.verbose,
            three_stage=self.chain_mode, # Currently chain_mode maps to three_stage
        )
        
        return output_path

    # ─────────────────────────────────────────────────────────
    # CONVENIENCE
    # ─────────────────────────────────────────────────────────

    def restore(self, input_path: str) -> Optional[str]:
        """Shortcut: full observe → plan → act cycle."""
        return self.run(input_path)

    def last_report(self) -> str:
        """Return a human-readable summary of the last restoration run."""
        if self._last_context is None:
            return "No restoration run recorded yet."
        ctx = self._last_context
        lines = [
            "=" * 50,
            "  MAIR+ RestorationAgent — Last Run Report",
            "=" * 50,
            f"  Input      : {ctx.original_path}",
            f"  Primary    : {ctx.degradation_result.get('primary', 'N/A')}",
            f"  Confidence : {ctx.degradation_result.get('confidence', 0):.3f}",
            ctx.summary(),
            f"  Best Output: {ctx.best_output_path or 'None'}",
            f"  Best Score : {ctx.best_quality_score:.4f}" if ctx.best_quality_score else "  Best Score : N/A",
            "=" * 50,
        ]
        return "\n".join(lines)

    def list_experts(self) -> None:
        """Print a summary of all registered expert agents."""
        print("\n  Registered Expert Agents:")
        print("  " + "─" * 56)
        for agent in self._experts.values():
            print(agent.capability_summary())
        print()

    def __repr__(self) -> str:
        return (
            f"RestorationAgent(max_attempts={self.max_attempts}, "
            f"chain_mode={self.chain_mode}, experts={len(self._experts)})"
        )
