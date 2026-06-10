"""
Expert Selector
===============
Selects and ranks experts from the Tool Registry based on
the degradation confidence scores from the detector.

Input:  degradation_result dict
        {"primary": "blur", "confidence": 0.83,
         "scores": {"blur": 0.83, "sr": 0.41, "jpeg": 0.18, ...}}

Output: ranked list of (expert_key, expert_fn) pairs,
        best expert first.

Ranking formula (per expert):
    score = quality_weight × task_confidence
            + speed_weight × (1 - task_confidence)

At high confidence  → prioritize quality (use best model)
At low confidence   → prioritize speed   (try fast model first)
"""

from core.tool_registry import REGISTRY, QUALITY_WEIGHT, SPEED_WEIGHT, expert_score

# Only consider degradations with score above this threshold
CONFIDENCE_THRESHOLD = 0.20


def select_experts(
    degradation_result: dict,
    memory_bias:        dict | None = None,   # C9: {expert_key: float bonus}
    image_size:         dict | None = None,   # C11: {width, height, pixels}
) -> list:
    """
    Rank experts based on degradation confidence scores.

    C9: memory_bias optionally adds a small bonus (max 0.15) to expert scores
    for experts that have historically performed well on similar images.

    C11: image_size is passed to expert_score() so slow models are penalized
    on large images.

    Returns:
        List of (expert_key, expert_entry) tuples, best first.
        Empty list only if no degradation exceeds threshold.
    """
    scores      = degradation_result.get("scores", {})
    pixel_count = image_size["pixels"] if image_size else None
    bias        = memory_bias or {}

    candidate_scores = []   # (ranking_score, expert_key, expert_entry)

    for expert_key, entry in REGISTRY.items():
        # Find the highest-confidence degradation this expert handles
        best_task_conf = 0.0
        for task in entry["handles"]:
            task_conf = scores.get(task, 0.0)
            if task_conf > best_task_conf:
                best_task_conf = task_conf

        # Skip experts whose degradation is below threshold
        if best_task_conf < CONFIDENCE_THRESHOLD:
            continue

        # Base score: quality × conf + speed × (1-conf), with resolution penalty (C11)
        ranking = expert_score(entry, best_task_conf, pixel_count=pixel_count)

        # C9: apply memory bias (bonus from similar past cases)
        memory_bonus = bias.get(expert_key, 0.0)
        ranking = round(ranking + memory_bonus, 4)

        candidate_scores.append((ranking, expert_key, entry, best_task_conf))

    # Sort descending by ranking score
    candidate_scores.sort(key=lambda x: x[0], reverse=True)

    result = [(key, entry) for _, key, entry, _ in candidate_scores]
    return result


def print_ranking(degradation_result: dict, memory_bias: dict | None = None) -> None:
    """Print the expert ranking table for debugging."""
    scores  = degradation_result.get("scores", {})
    ranked  = select_experts(degradation_result, memory_bias=memory_bias)

    print("\n  Expert Ranking:")
    print(f"  {'RANK':<5}  {'EXPERT':<26}  {'TASK':<10}  {'CONF':<8}  {'SCORE':<8}  {'BIAS'}")
    print("  " + "─" * 68)

    for i, (key, entry) in enumerate(ranked, 1):
        best_conf = max(scores.get(t, 0.0) for t in entry["handles"])
        base_s    = expert_score(entry, best_conf)
        bonus     = (memory_bias or {}).get(key, 0.0)
        final_s   = round(base_s + bonus, 4)
        bias_str  = f"+{bonus:.4f}" if bonus > 0 else "—"
        print(
            f"  #{i:<4}  {key:<26}  {entry['task']:<10}  "
            f"{best_conf:<8.3f}  {final_s:<8.4f}  {bias_str}"
        )

    if not ranked:
        print("  (no experts meet the confidence threshold)")
    print()
