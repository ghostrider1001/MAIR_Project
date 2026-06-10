"""
Quality Gate — Progressive Quality Gating with Rollback
========================================================
MAIR+ Contribution C4.

Prevents stage-level quality regression by comparing the quality score
of an image BEFORE a stage runs vs AFTER. If the stage worsened the
image beyond a tolerance threshold, the stage output is rolled back
(discarded) and the pre-stage image is passed to the next stage.

Design choices:
  - Uses a RELATIVE threshold (ratio), not absolute. This makes the gate
    metric-agnostic: it works correctly whether the score comes from
    SSIM-only, composite SSIM+LPIPS, or any future formula.
  - Default threshold: 0.97 → accepts up to a 3% relative quality drop.
  - prev_score is clamped to a minimum of 0.01 to avoid division by zero
    on images where the quality score is near zero.

Usage:
    gate = QualityGate()

    pre_score  = evaluate_quality(original_path, current_path)
    post_score = evaluate_quality(original_path, stage_output_path)

    if gate.should_accept(pre_score, post_score):
        current_path = stage_output_path   # accept stage result
    else:
        pass   # rollback — keep current_path unchanged
"""


class QualityGate:
    """
    Stage-level quality gate with relative threshold comparison.

    Args:
        relative_threshold:  Minimum ratio (post_score / pre_score) to
                             accept a stage result. Default 0.97 means
                             "allow up to 3% relative quality drop."
    """

    # ── Class-level default ──────────────────────────────────
    RELATIVE_THRESHOLD = 0.97

    def __init__(self, relative_threshold: float = None):
        self.threshold = (
            relative_threshold
            if relative_threshold is not None
            else self.RELATIVE_THRESHOLD
        )

    def should_accept(self, prev_score: float, new_score: float) -> bool:
        """
        Determine whether a stage result should be accepted.

        Returns True if the quality did not regress beyond the threshold.
        Returns False (rollback) if quality dropped too much.

        Args:
            prev_score:  quality score BEFORE the stage ran
            new_score:   quality score AFTER the stage ran

        Returns:
            True  → accept the stage output
            False → rollback (keep pre-stage image)
        """
        # Guard: if either score is None/NaN, accept (can't compare)
        if prev_score is None or new_score is None:
            return True

        # Guard: if pre-stage quality is near zero, accept anything positive
        safe_prev = max(prev_score, 0.01)

        ratio = new_score / safe_prev
        return ratio >= self.threshold

    def format_decision(
        self,
        prev_score: float,
        new_score:  float,
        accepted:   bool,
    ) -> str:
        """
        Return a human-readable explanation of the gate decision.
        """
        if prev_score is None or new_score is None:
            return "Quality gate: ACCEPT (scores unavailable for comparison)"

        safe_prev = max(prev_score, 0.01)
        ratio     = new_score / safe_prev
        delta     = new_score - prev_score
        pct       = (ratio - 1.0) * 100.0

        action = "ACCEPT" if accepted else "ROLLBACK"

        return (
            f"Quality gate: {action}  |  "
            f"pre={prev_score:.4f}  post={new_score:.4f}  "
            f"Δ={delta:+.4f}  ratio={ratio:.4f} ({pct:+.1f}%)  "
            f"threshold={self.threshold:.2f}"
        )
