"""
Iterative Detection Context (C2)
==================================
MAIR+ Contribution C2: Iterative Re-Detection Between Stages.

Stores a record of degradation scores computed at each stage boundary,
enabling post-stage score updates to influence remaining stages.

The scheduler runs detect_degradation() ONCE at pipeline entry. After
each ACCEPTED stage completes, it re-runs detect_degradation() on the
updated image and records the change. The new scores feed the next stage's
expert selection — so if JPEG removal reveals blur, Stage 2 sees the
updated blur signal.

Usage:
    from core.iterative_context import IterativeContext

    ctx = IterativeContext()
    ctx.add("compression", before_scores, after_scores)
    change = ctx.score_change("blur")     # how much blur signal changed
    print(ctx.summary())
"""

from dataclasses import dataclass, field


@dataclass
class IterativeDetectionRecord:
    """
    One re-detection event recorded after a stage completes.

    Attributes:
        stage          : the stage that just completed
        scores_before  : degradation scores used to PLAN this stage
        scores_after   : degradation scores measured on the stage OUTPUT
    """
    stage:         str
    scores_before: dict
    scores_after:  dict


class IterativeContext:
    """
    Tracks degradation score evolution across pipeline stages.

    When Stage 1 (compression) removes JPEG artifacts, the blur or noise
    signals that were masked by compression can become more visible.
    This context records that evolution so Stage 2 plans with updated scores.
    """

    def __init__(self):
        self.records: list[IterativeDetectionRecord] = []

    def add(self, stage: str, before_scores: dict, after_scores: dict):
        """Record a re-detection event after a stage completes."""
        self.records.append(IterativeDetectionRecord(
            stage=stage,
            scores_before=dict(before_scores),
            scores_after=dict(after_scores),
        ))

    def score_change(self, degradation_type: str) -> float:
        """
        Compute total change in a degradation signal across all recorded stages.

        Returns:
            float — positive means signal increased (degradation more visible),
                    negative means decreased (degradation resolved).
        """
        if not self.records:
            return 0.0

        first = self.records[0].scores_before.get(degradation_type, 0.0)
        last  = self.records[-1].scores_after.get(degradation_type, 0.0)
        return round(last - first, 4)

    def latest_scores(self) -> dict | None:
        """Return the most recently measured degradation scores."""
        if not self.records:
            return None
        return self.records[-1].scores_after

    def summary(self) -> str:
        """Human-readable summary of all score changes."""
        if not self.records:
            return "IterativeContext: no re-detection records yet."

        lines = ["Iterative Re-Detection Summary:"]
        all_keys = set()
        for r in self.records:
            all_keys |= set(r.scores_before.keys()) | set(r.scores_after.keys())

        for key in sorted(all_keys):
            if key in ("image_size", "primary"):
                continue
            first = self.records[0].scores_before.get(key, 0.0)
            last  = self.records[-1].scores_after.get(key, 0.0)
            delta = last - first
            sign  = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "≈")
            lines.append(f"  {key:<12} {first:.3f} → {last:.3f}  ({sign}{abs(delta):.3f})")

        return "\n".join(lines)

    def __repr__(self):
        return f"IterativeContext(stages_recorded={len(self.records)})"
