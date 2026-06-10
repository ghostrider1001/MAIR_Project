"""
Expert Voting Scheduler (C12)
==============================
MAIR+ Contribution C12: Expert Voting Ensemble.

Instead of running experts sequentially (stopping at first ACCEPT),
voting mode runs the top-N experts and keeps the one with the highest
quality score.

This increases robustness at the cost of ~N× expert invocations per stage.
Voting is OFF by default — enable with --voting flag.

Design note:
  - max_voters=2 is the default (run top-2, keep best).
  - Both experts write output files; the winner's path is returned.
  - Losing output files are deleted by default (cleanup_losers=True).
  - Quality is evaluated using evaluate_quality_full() to get the composite score.

Usage:
    from scheduler.voting_scheduler import run_voting_stage

    result = run_voting_stage(
        ranked_experts=ranked[:2],
        current_path="input.png",
        max_voters=2,
    )
    # result → {output_path, quality_score, winning_expert, all_results}
"""

import os
import time
from evaluation.quality_evaluator import evaluate_quality_full


def run_voting_stage(
    ranked_experts: list[tuple],
    current_path:   str,
    max_voters:     int  = 2,
    cleanup_losers: bool = True,   # delete non-winning output files
) -> dict:
    """
    Run top-N experts on the same input and return the best result.

    Args:
        ranked_experts  : list of (expert_key, expert_entry_dict) pre-ranked by selector
        current_path    : path to the input image for this stage
        max_voters      : maximum number of experts to run (default: 2)
        cleanup_losers  : if True (default), delete non-winning output files after voting

    Returns:
        {
            "output_path":     str | None   — path to winning expert's output
            "quality_score":   float | None — composite quality score of winner
            "winning_expert":  str | None   — expert key that won
            "all_results":     list[dict]   — all expert run details
            "n_voters":        int          — number of experts that ran
        }
    """
    voters      = ranked_experts[:max_voters]
    all_results = []
    best_path   = None
    best_score  = -1.0
    best_expert = None

    for expert_key, expert_entry in voters:
        expert_fn = expert_entry.get("fn")
        if expert_fn is None:
            continue

        t0 = time.time()
        try:
            output_path = expert_fn(current_path)
        except Exception as e:
            print(f"[VotingScheduler] {expert_key} raised exception: {e}")
            output_path = None
        elapsed = round(time.time() - t0, 2)

        if output_path is None:
            all_results.append({
                "expert_key": expert_key,
                "output_path": None,
                "quality_score": None,
                "time_s": elapsed,
                "status": "failed",
            })
            continue

        # Evaluate quality using composite score
        try:
            q_dict = evaluate_quality_full(current_path, output_path)
            score  = q_dict["composite_score"]
        except Exception:
            score = None

        all_results.append({
            "expert_key":   expert_key,
            "output_path":  output_path,
            "quality_score": score,
            "time_s":       elapsed,
            "status":       "ok",
        })

        if score is not None and score > best_score:
            best_score  = score
            best_path   = output_path
            best_expert = expert_key

    # Clean up losing output files to prevent orphan accumulation
    if cleanup_losers:
        for r in all_results:
            loser_path = r.get("output_path")
            if loser_path and loser_path != best_path:
                try:
                    os.remove(loser_path)
                except OSError:
                    pass  # already deleted or on a different fs

    return {
        "output_path":    best_path,
        "quality_score":  best_score if best_path else None,
        "winning_expert": best_expert,
        "all_results":    all_results,
        "n_voters":       len(voters),
    }
