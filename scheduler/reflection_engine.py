"""
Reflection Engine
=================
Decides what action to take after each expert attempt.

Actions:
    ACCEPT   — quality is sufficient, stop trying
    RETRY    — quality is poor, try next expert
    ESCALATE — all attempts used, accept best seen

Decision logic:
    1. If the expert failed (no output):            RETRY
    2. If quality improved significantly (>= +0.05): ACCEPT
    3. If quality improved slightly (>= +0.01)
       and we are on attempt >= 1:                  ACCEPT
    4. If quality did not improve:                  RETRY
    5. If max attempts reached:                     ESCALATE → ACCEPT best
"""

from core.restoration_context import RestorationContext


# ─────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────
EXCELLENT_THRESHOLD     = 0.85   # score above this → always ACCEPT
GOOD_THRESHOLD          = 0.70   # score above this on attempt ≥ 1 → ACCEPT
SIGNIFICANT_IMPROVEMENT = 0.05   # absolute SSIM gain that always triggers ACCEPT
MINOR_IMPROVEMENT       = 0.01   # minimum gain to accept on late attempts


# ─────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────
ACCEPT   = "ACCEPT"
RETRY    = "RETRY"
ESCALATE = "ESCALATE"


def reflect(ctx: RestorationContext) -> str:
    """
    Evaluate the last attempt and return the next action.

    Args:
        ctx: RestorationContext with at least one recorded attempt.

    Returns:
        One of: ACCEPT | RETRY | ESCALATE
    """
    if not ctx.attempts:
        return RETRY

    last = ctx.attempts[-1]
    attempt_num   = last.attempt_num
    max_attempts  = ctx.max_attempts
    current_score = last.quality_score
    best_score    = ctx.best_quality_score

    # ── Rule 0: Expert produced no output ──────────────────────
    if not last.success:
        if attempt_num >= max_attempts - 1:
            return ESCALATE
        return RETRY

    # ── Rule 1: Excellent quality — always accept ───────────────
    if current_score is not None and current_score >= EXCELLENT_THRESHOLD:
        return ACCEPT

    # ── Rule 2: Good quality on attempt ≥ 1 ────────────────────
    if (
        current_score is not None
        and current_score >= GOOD_THRESHOLD
        and attempt_num >= 1
    ):
        return ACCEPT

    # ── Rule 3: Significant improvement over previous ──────────
    if len(ctx.attempts) >= 2:
        prev_score = ctx.attempts[-2].quality_score
        if (
            prev_score is not None
            and current_score is not None
            and (current_score - prev_score) >= SIGNIFICANT_IMPROVEMENT
        ):
            return ACCEPT

        if (
            prev_score is not None
            and current_score is not None
            and (current_score - prev_score) >= MINOR_IMPROVEMENT
            and attempt_num >= max_attempts - 1
        ):
            return ACCEPT

    # ── Rule 4: Max attempts reached ───────────────────────────
    if attempt_num >= max_attempts - 1:
        return ESCALATE

    # ── Rule 5: Keep trying ─────────────────────────────────────
    return RETRY


def explain(ctx: RestorationContext, action: str) -> str:
    """Return a human-readable explanation of the reflection decision."""
    if not ctx.attempts:
        return "No attempts recorded."

    last  = ctx.attempts[-1]
    score = last.quality_score
    score_str = f"{score:.4f}" if score else "N/A"

    if action == ACCEPT:
        if score and score >= EXCELLENT_THRESHOLD:
            return f"Quality excellent ({score_str} ≥ {EXCELLENT_THRESHOLD}) — accepting."
        elif score and score >= GOOD_THRESHOLD:
            return f"Quality good ({score_str} ≥ {GOOD_THRESHOLD}) — accepting on attempt {last.attempt_num}."
        else:
            return f"Significant improvement detected — accepting ({score_str})."

    elif action == RETRY:
        return f"Quality insufficient ({score_str}) — retrying with next expert."

    elif action == ESCALATE:
        best = ctx.best_quality_score
        return (
            f"Max attempts reached. "
            f"Accepting best result: {ctx.best_expert_key} "
            f"(score={best:.4f})" if best else "No successful output found."
        )

    return f"Unknown action: {action}"
