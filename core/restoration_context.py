"""
Restoration Context
===================
Dataclass that carries the full pipeline state through the
scheduler, expert selector, and reflection engine.

Passed by reference — each stage updates it rather than
creating new objects. This enables the reflection engine
to see the full history of what was tried and what worked.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttemptRecord:
    """Captures the outcome of one expert invocation."""
    attempt_num:    int
    expert_key:     str
    expert_name:    str
    output_path:    Optional[str]
    quality_score:  Optional[float]
    success:        bool


@dataclass
class RestorationContext:
    """
    Full state of one restoration pipeline run.

    Created once at the start of run_scheduler() and
    updated after each expert attempt.
    """
    # ── Input ─────────────────────────────────────────────────
    original_path:      str

    # ── Degradation analysis ──────────────────────────────────
    degradation_result: dict = field(default_factory=dict)
    # e.g. {"primary": "blur", "confidence": 0.83,
    #        "scores": {"blur": 0.83, "sr": 0.41, ...}}

    # ── Ranked expert plan ────────────────────────────────────
    expert_plan:        list = field(default_factory=list)
    # list of (expert_key, expert_fn) in priority order

    # ── Attempt history ───────────────────────────────────────
    attempts:           list = field(default_factory=list)
    # list of AttemptRecord

    # ── Current best ─────────────────────────────────────────
    best_output_path:   Optional[str]  = None
    best_quality_score: Optional[float] = None
    best_expert_key:    Optional[str]  = None

    # ── Pipeline state ────────────────────────────────────────
    current_attempt:    int  = 0
    max_attempts:       int  = 2
    pipeline_complete:  bool = False

    def record_attempt(
        self,
        expert_key:    str,
        expert_name:   str,
        output_path:   Optional[str],
        quality_score: Optional[float],
    ) -> None:
        """Record the result of one expert attempt and update best."""
        success = output_path is not None and quality_score is not None

        record = AttemptRecord(
            attempt_num=self.current_attempt,
            expert_key=expert_key,
            expert_name=expert_name,
            output_path=output_path,
            quality_score=quality_score,
            success=success,
        )
        self.attempts.append(record)

        # Update best if this attempt was better
        if (
            success and (
                self.best_quality_score is None
                or quality_score > self.best_quality_score
            )
        ):
            self.best_output_path   = output_path
            self.best_quality_score = quality_score
            self.best_expert_key    = expert_key

        self.current_attempt += 1

    def summary(self) -> str:
        """Return a formatted summary of all attempts."""
        lines = [f"  Attempts: {len(self.attempts)}"]
        for r in self.attempts:
            status = "OK" if r.success else "FAIL"
            score  = f"{r.quality_score:.4f}" if r.quality_score else "N/A"
            lines.append(
                f"    [{r.attempt_num}] {r.expert_key:<22}  "
                f"score={score:<8}  [{status}]"
            )
        if self.best_expert_key:
            lines.append(
                f"  Best: {self.best_expert_key}  "
                f"(score={self.best_quality_score:.4f})"
            )
        return "\n".join(lines)
