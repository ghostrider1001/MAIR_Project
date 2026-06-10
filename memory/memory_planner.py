"""
Memory Planner — C9
====================
Converts retrieved similar cases from CaseStore into an expert ranking
bias dict that the expert selector can apply.

Design:
    bias = BASE_BONUS × cosine_similarity

where BASE_BONUS = 0.15 caps the maximum memory influence at 15% of the
ranking score. This prevents memory from overriding the primary degradation
signal — it gently nudges expert selection toward historically effective choices.

Usage:
    from memory.case_store import CaseStore
    from memory.memory_planner import get_bias

    store  = CaseStore()
    bias   = get_bias(current_scores, stage="imaging", store=store)
    # bias → {"restormer_deblur": 0.13, ...}

    # Then in expert_selector.select_experts():
    #   final_score = base_score + bias.get(expert_key, 0.0)
"""

from memory.case_store import CaseStore


BASE_BONUS = 0.15   # maximum rank bonus from memory retrieval


def get_bias(
    degradation_scores: dict,
    stage:              str,
    store:              CaseStore,
    top_k:             int   = 3,
    min_similarity:    float = 0.80,
) -> dict[str, float]:
    """
    Compute memory bias dict for expert selection.

    For each retrieved similar case, adds a bonus proportional to its
    cosine similarity. If multiple cases recommend the same expert, bonuses
    are averaged (not summed) to prevent gaming.

    Args:
        degradation_scores : current image degradation score dict
        stage              : pipeline stage to query
        store              : CaseStore instance
        top_k              : number of cases to retrieve
        min_similarity     : minimum similarity threshold (from CaseStore)

    Returns:
        Dict mapping expert_key → float bonus (in [0, BASE_BONUS]).
        Empty dict if no similar cases found.
    """
    cases = store.retrieve(
        degradation_scores,
        stage=stage,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    if not cases:
        return {}

    # Aggregate bonuses per expert (average when multiple cases agree)
    expert_bonuses: dict[str, list[float]] = {}
    for case in cases:
        key   = case["expert_key"]
        bonus = BASE_BONUS * case["similarity"]
        if key not in expert_bonuses:
            expert_bonuses[key] = []
        expert_bonuses[key].append(bonus)

    bias = {
        key: round(sum(vals) / len(vals), 4)
        for key, vals in expert_bonuses.items()
    }

    return bias
