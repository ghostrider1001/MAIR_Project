"""
Confidence-Tiered Scheduling (C10)
====================================
MAIR+ Contribution C10.

Implements a three-tier dispatch policy that matches expert complexity to
degradation confidence — avoiding expensive transformer models when the
system is uncertain about the degradation type.

Tiers:
    HIGH   (≥ 0.60): Use full ranked list — all quality levels allowed
    MEDIUM (≥ 0.35): Use only fast experts (speed in "very_fast" | "fast")
                     Falls back to top-1 if no fast expert exists for this task
    LOW    (< 0.35): Skip stage entirely (= existing STAGE_THRESHOLD = 0.20 behavior,
                     but extended to a middle tier rather than binary)

The STAGE_THRESHOLD (0.20) still governs stage activation — a stage doesn't
even reach apply_policy() if its signal is below 0.20.
The policy only operates on signals between 0.20 and 1.0.

Usage:
    from scheduler.confidence_policy import apply_policy

    filtered = apply_policy(stage_confidence=0.45, ranked_experts=ranked)
    # Returns only speed="very_fast"/"fast" experts for confidence 0.45
"""

TIER_HIGH   = 0.60   # Full expert list (all quality levels)
TIER_MEDIUM = 0.35   # Fast experts only
TIER_LOW    = 0.20   # Stage activation threshold (handled by scheduler)

FAST_SPEEDS = {"very_fast", "fast"}


def get_tier(confidence: float) -> str:
    """
    Classify a degradation confidence score into a tier label.

    Returns:
        "high"   → full expert list
        "medium" → fast experts only
        "low"    → borderline (scheduler should skip if below STAGE_THRESHOLD)
    """
    if confidence >= TIER_HIGH:
        return "high"
    elif confidence >= TIER_MEDIUM:
        return "medium"
    else:
        return "low"


def apply_policy(
    stage_confidence: float,
    ranked_experts:   list[tuple],
) -> list[tuple]:
    """
    Filter the ranked expert list according to the confidence tier.

    Args:
        stage_confidence : confidence score for the dominant degradation in this stage
        ranked_experts   : list of (expert_key, expert_entry_dict) tuples,
                           pre-ranked by the expert selector

    Returns:
        Filtered list of (expert_key, expert_entry_dict) tuples.
        For "high" tier: unchanged ranked list.
        For "medium" tier: only fast experts; if none, top-1 from ranked list.
        For "low" tier: empty list (stage should be skipped or already filtered).
    """
    tier = get_tier(stage_confidence)

    if tier == "high":
        return ranked_experts

    if tier == "medium":
        fast_experts = [
            (key, entry) for key, entry in ranked_experts
            if entry.get("speed", "medium") in FAST_SPEEDS
        ]
        if fast_experts:
            return fast_experts
        # No fast expert available — fall through to top-1 from full list
        return ranked_experts[:1]

    # tier == "low" — signal below TIER_MEDIUM, skip
    return []
