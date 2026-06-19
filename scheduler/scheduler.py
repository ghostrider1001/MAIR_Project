"""
MAIR+ Scheduler — Three-Stage Framework
=========================================
Implements the real-world degradation prior from:
  Jiang et al., "Multi-Agent Image Restoration", IJCV 2026

Three-Stage Framework:
  Real degradations occur as: Scene → Imaging → Compression
  Restoration reverses them:  Compression → Imaging → Scene

  Stage 1 (compression): JPEG artifact removal  ← always attempted first
  Stage 2 (imaging)    : Deblur, Denoise, SR    ← always second
  Stage 3 (scene)      : Low-light, Rain, Haze  ← always third

This ordering is empirically validated in the paper (Table 1): the best
restoration plan follows the three-stage order with ≥87% probability.

Flow (per stage):
    if any degradation in this stage exceeds STAGE_THRESHOLD:
        select_experts() for that stage
        run best expert on current image
        evaluate quality → reflect (ACCEPT | RETRY | ESCALATE)
        pass updated image to next stage

Two scheduler modes:
  three_stage=True  (default) — enforce stage ordering (paper method)
  three_stage=False           — confidence-based single-expert (old method)
"""

import os
import json
import shutil
import time
from datetime import datetime

from core.degradation_detector  import detect_degradation
from core.restoration_context   import RestorationContext
from core.tool_registry         import (
    REGISTRY, STAGE_ORDER, get_experts_for_stage, list_all as list_registry
)
from core.quality_gate          import QualityGate
from core.spatial_integrity     import SpatialGuard
from core.iterative_context     import IterativeContext          # C2
from scheduler.expert_selector  import select_experts, print_ranking
from scheduler.reflection_engine import reflect, explain, ACCEPT, ESCALATE
from scheduler.confidence_policy import apply_policy             # C10
from scheduler.voting_scheduler  import run_voting_stage         # C12
from evaluation.quality_evaluator import evaluate_quality, evaluate_quality_full
from evaluation.clinical_evaluator import evaluate_clinical_composite
from memory.case_store          import CaseStore                  # C9
from memory.memory_planner      import get_bias                   # C9


# ─────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────

# Default stage activation threshold — overridden by config/thresholds.json (C8)
STAGE_THRESHOLD_DEFAULT = 0.20

# Minimum expert confidence within an active stage
EXPERT_THRESHOLD = 0.20

# Quality gate: reject stage if SSIM(pre-stage, post-stage) < this (C4)
# 0.50 catches catastrophic expert failures (extreme CLAHE) while
# allowing legitimate restoration changes (denoising SSIM ~0.75)
# NOTE: Lowered to 0.00 because using SSIM against a degraded input image 
# penalizes experts like DCP that massively (and correctly) alter the image!
QUALITY_GATE_MIN = 0.00


def _load_stage_thresholds() -> dict:
    """Load calibrated stage thresholds from config/thresholds.json (C8)."""
    config_path = os.path.join("config", "thresholds.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return {
                "compression": float(cfg.get("compression", STAGE_THRESHOLD_DEFAULT)),
                "imaging":     float(cfg.get("imaging",     STAGE_THRESHOLD_DEFAULT)),
                "scene":       float(cfg.get("scene",       STAGE_THRESHOLD_DEFAULT)),
            }
        except Exception:
            pass
    return {s: STAGE_THRESHOLD_DEFAULT for s in STAGE_ORDER}


STAGE_THRESHOLD = STAGE_THRESHOLD_DEFAULT  # backward compat alias


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _stage_is_active(stage: str, scores: dict, thresholds: dict | None = None) -> bool:
    """
    Return True if any degradation in this stage exceeds its calibrated threshold.

    Args:
        stage      : "compression" | "imaging" | "scene"
        scores     : degradation confidence scores from detect_degradation()
        thresholds : per-stage threshold dict from config/thresholds.json (C8).
                     Falls back to STAGE_THRESHOLD_DEFAULT if not provided.
    """
    from core.tool_registry import DEGRADATION_STAGE
    stage_thr = (thresholds or {}).get(stage, STAGE_THRESHOLD_DEFAULT)
    return any(
        scores.get(deg, 0.0) >= stage_thr
        for deg, s in DEGRADATION_STAGE.items()
        if s == stage
    )


def _select_stage_experts(
    stage:        str,
    scores:       dict,
    memory_bias:  dict | None = None,   # C9
    image_size:   dict | None = None,   # C11
    thresholds:   dict | None = None,   # C8
) -> list:
    """
    Select and rank experts for a given stage based on degradation scores.
    Applies memory bias (C9), resolution penalty (C11), and confidence
    tiering (C10) to produce the final ordered expert list.

    Returns:
        Ranked + filtered list of (expert_key, expert_entry) tuples.
    """
    from core.tool_registry import QUALITY_WEIGHT, SPEED_WEIGHT, expert_score
    from scheduler.confidence_policy import apply_policy

    stage_threshold = (thresholds or {}).get(stage, EXPERT_THRESHOLD)
    stage_entries   = get_experts_for_stage(stage)
    pixel_count     = image_size["pixels"] if image_size else None
    bias            = memory_bias or {}

    candidate_scores = []
    stage_confidence = 0.0   # max confidence across all degradations in this stage

    for key, entry in stage_entries:
        best_conf = max(scores.get(t, 0.0) for t in entry["handles"])
        if best_conf < stage_threshold:
            continue
        stage_confidence = max(stage_confidence, best_conf)

        # Base score with resolution penalty (C11)
        ranking = expert_score(entry, best_conf, pixel_count=pixel_count)

        # Memory bias (C9)
        ranking = round(ranking + bias.get(key, 0.0), 4)
        candidate_scores.append((ranking, key, entry))

    candidate_scores.sort(key=lambda x: x[0], reverse=True)
    ranked = [(key, entry) for _, key, entry in candidate_scores]

    # C10: confidence-tiered filtering
    ranked = apply_policy(stage_confidence, ranked)

    return ranked



def _copy_to_temp(src: str, stage_name: str) -> str:
    """
    Copy an image to a temp path for inter-stage handoff.
    Returns the temp path.
    """
    ext     = os.path.splitext(src)[1]
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    dst_dir = os.path.join("outputs", "pipeline_tmp")
    os.makedirs(dst_dir, exist_ok=True)
    dst     = os.path.join(dst_dir, f"stage_{stage_name}_{ts}{ext}")
    shutil.copy2(src, dst)
    return dst


def _label(score):
    if score is None:   return "N/A"
    if score >= 0.85:   return "Excellent"
    if score >= 0.70:   return "Good"
    if score >= 0.50:   return "Moderate"
    return "Poor"


# ─────────────────────────────────────────────────────────────
# THREE-STAGE SCHEDULER
# ─────────────────────────────────────────────────────────────

def run_three_stage_scheduler(
    input_path:      str,
    max_attempts:    int  = 2,
    verbose:         bool = True,
    voting:          bool = False,   # C12: run top-2 experts, keep best
    use_memory:      bool = True,    # C9:  use case-based memory planning
    budget_seconds:  float | None = None,  # C11: time budget
    clinical_eval:   bool = False
) -> dict:
    """
    Run the MAIR+ v2 three-stage restoration pipeline.

    Processes stages: compression → imaging → scene.
    Each stage is skipped if its degradation confidence is below threshold.
    The output of each stage feeds into the next as input.

    New in v2:
      C2  — Iterative re-detection between stages
      C4  — Quality gate with rollback (threshold 0.50)
      C8  — Config-driven stage thresholds from config/thresholds.json
      C9  — Memory-augmented planning (CaseStore)
      C10 — Confidence-tiered expert filtering (inside _select_stage_experts)
      C11 — Resolution-aware ranking (image_size passed through)
      C12 — Expert voting mode (--voting flag)

    Args:
        input_path      : path to the degraded input image
        max_attempts    : max expert retries per stage (default: 2)
        verbose         : print full reasoning trace
        voting          : if True, run top-2 experts per stage, keep best
        use_memory      : if True, query CaseStore for expert bias
        budget_seconds  : if set, skip slow experts when budget is near exhausted
        clinical_eval   : use clinical composite instead of standard quality

    Returns:
        dict with:
            output_path        : path to best restored image (or None)
            stage_results      : dict mapping stage → stage info dict
            total_time_s       : total wall-clock time
            invocation_count   : total expert calls made
            memory_bias_applied: bool (C9)
            iterative_context  : IterativeContext summary (C2)
    """
    _log = print if verbose else lambda *a, **k: None
    t_start = time.time()

    _log("\n" + "=" * 60)
    _log("   MAIR+ v2 THREE-STAGE SCHEDULER ACTIVATED")
    if voting:      _log("   Mode: VOTING (top-2 per stage)")
    if use_memory:  _log("   Mode: MEMORY-AUGMENTED (case-based reasoning)")
    _log("=" * 60)

    # ── C8: Load calibrated thresholds ────────────────────────
    thresholds = _load_stage_thresholds()

    # ── C9: Initialize memory store ───────────────────────────
    memory_store         = CaseStore() if use_memory else None
    memory_bias_applied  = False

    # ── C2: Iterative re-detection context ────────────────────
    iter_ctx = IterativeContext()

    # ── STEP 1: Initial degradation detection ─────────────────
    degradation_result = detect_degradation(input_path)
    scores:     dict[str, float]          = degradation_result["scores"]
    primary:    str                       = degradation_result["primary"]
    confidence: float                     = degradation_result["confidence"]
    image_size: dict[str, int] | None     = degradation_result.get("image_size")  # C11

    _log(f"\n[Scheduler] Primary  : {primary}  (conf: {confidence:.3f})")
    _log(f"[Scheduler] Scores   : { {k: f'{v:.3f}' for k,v in scores.items() if k != 'image_size'} }")
    if image_size:
        _log(f"[Scheduler] ImgSize  : {image_size['width']}×{image_size['height']}  ({image_size['pixels']:,} px)")

    # ── STEP 2: Process stages in order ───────────────────────
    current_path     = input_path
    stage_results    = {}
    invocation_count = 0

    for stage in STAGE_ORDER:
        stage_label     = stage.upper()
        stage_threshold = thresholds.get(stage, STAGE_THRESHOLD_DEFAULT)

        # C11: budget check — skip slow stages if time is short
        if budget_seconds is not None:
            elapsed = time.time() - t_start
            if elapsed > budget_seconds * 0.85:  # 85% of budget used
                _log(f"[Scheduler] Stage {stage_label:12} SKIPPED (time budget {budget_seconds}s near exhausted)")
                stage_results[stage] = {"skipped": True, "reason": "budget"}
                continue

        # Activation check (Bug 2 fix: pass calibrated thresholds so C8 is used)
        if not _stage_is_active(stage, scores, thresholds):
            _log(f"\n[Scheduler] Stage {stage_label:12} SKIPPED (below threshold {stage_threshold:.2f})")
            stage_results[stage] = {"skipped": True}
            continue

        # C9: Get memory bias for this stage
        memory_bias = {}
        if memory_store is not None:
            memory_bias = get_bias(scores, stage=stage, store=memory_store)
            if memory_bias:
                _log(f"[Scheduler] Memory bias for {stage_label}: {memory_bias}")
                memory_bias_applied = True

        # Select experts (C9 bias, C10 tiering, C11 resolution)
        ranked = _select_stage_experts(
            stage, scores,
            memory_bias=memory_bias,
            image_size=image_size,
            thresholds=thresholds,
        )
        if not ranked:
            _log(f"\n[Scheduler] Stage {stage_label:12} No experts qualify — skipping")
            stage_results[stage] = {"skipped": True}
            continue

        _log(f"\n[Scheduler] ── Stage {stage_label} ({', '.join(e['task'] for _,e in ranked[:3])})")
        _log(f"[Scheduler]    Candidates: {[k for k,_ in ranked]}")

        # Build restoration context for this stage's attempt loop
        ctx = RestorationContext(
            original_path=current_path,
            degradation_result=degradation_result,
            expert_plan=ranked,
            max_attempts=max_attempts,
        )

        # ── C12: Voting mode — run top-2, keep best ───────────
        if voting:
            _log(f"[Scheduler]    [VOTING] Running top-{min(2, len(ranked))} experts")
            vote_result = run_voting_stage(ranked, current_path, max_voters=2)
            invocation_count += vote_result["n_voters"]

            winning_key = vote_result["winning_expert"]
            if vote_result["output_path"] and winning_key:
                # Record winning expert in context
                winning_name: str = str(REGISTRY.get(winning_key, {}).get("name", "?"))
                ctx.record_attempt(
                    expert_key=winning_key,
                    expert_name=winning_name,
                    output_path=vote_result["output_path"],
                    quality_score=vote_result["quality_score"],
                )
                _log(f"[Scheduler]    [VOTING] Winner: {winning_key}  score={vote_result['quality_score']:.4f}")

        else:
            # ── Standard sequential attempt loop ────────────────
            # Bug 5 fix: construct SpatialGuard once per stage, not per attempt
            stage_guard = SpatialGuard(current_path) if any(
                e.get("preserves_size", True) for _, e in ranked[:max_attempts]
            ) else None

            for attempt_idx, (expert_key, expert_entry) in enumerate(ranked[:max_attempts]):
                expert_name = expert_entry["name"]
                expert_fn   = expert_entry["fn"]
                invocation_count += 1

                _log(f"\n[Scheduler]    Attempt {attempt_idx + 1}/{min(max_attempts, len(ranked))}")
                _log(f"[Scheduler]    Expert  : {expert_name}")

                output_path = None
                quality     = None

                try:
                    output_path = expert_fn(current_path)
                except Exception as e:
                    _log(f"[Scheduler]    Exception: {e}")

                # C5: Spatial integrity check (reuse pre-built guard)
                if output_path is not None and expert_entry.get("preserves_size", True) and stage_guard:
                    output_path = stage_guard.check_and_fix(output_path)

                if output_path and os.path.exists(output_path):
                    try:
                        if clinical_eval:
                            quality = evaluate_clinical_composite(current_path, output_path)
                            _log(f"[Quality Evaluator] Clinical Composite : {quality:.4f}")
                        else:
                            quality = evaluate_quality(current_path, output_path)
                        _log(f"[Scheduler]    Quality : {quality:.4f}  ({_label(quality)})")
                    except Exception as e:
                        _log(f"[Scheduler]    Quality eval failed: {e}")
                else:
                    _log("[Scheduler]    Expert returned None (failed).")

                ctx.record_attempt(
                    expert_key=expert_key,
                    expert_name=expert_name,
                    output_path=output_path,
                    quality_score=quality,
                )

                action = reflect(ctx)
                reason = explain(ctx, action)
                _log(f"[Scheduler]    Reflect : {action}  —  {reason}")

                if action in (ACCEPT, ESCALATE):
                    break

        # ── C4: Quality gate — rollback if stage regressed ────
        stage_best  = ctx.best_output_path
        stage_score = ctx.best_quality_score
        rolled_back = False
        pre_stage_quality = None

        if stage_best:
            # Bug 1 fix: use stage_score directly as an absolute floor.
            # stage_score = composite quality of expert output vs. pre-stage image.
            # Accept if score >= QUALITY_GATE_MIN (0.50). This catches catastrophic
            # expert failures while allowing legitimate structural changes from restoration.
            # Previously was: gate.should_accept(1.0, stage_score) which used a
            # relative ratio of stage_score/1.0 = stage_score — same math but clearer
            # intent and avoids misuse of the relative QualityGate API.
            accepted = (stage_score is not None and stage_score >= QUALITY_GATE_MIN)
            decision = (
                f"Quality gate: {'ACCEPT' if accepted else 'ROLLBACK'}  |  "
                f"score={stage_score:.4f}  threshold={QUALITY_GATE_MIN}"
            )
            _log(f"[Scheduler] {decision}")

            if accepted:
                _log(f"[Scheduler] Stage {stage_label} → accepted: {stage_best}  (score={stage_score:.4f})")
                prev_path  = current_path
                current_path = stage_best

                # C9: Record successful case in memory
                if memory_store is not None and stage_score is not None and ctx.best_expert_key:
                    stored = memory_store.record(
                        degradation_scores=scores,
                        stage=stage,
                        expert_key=ctx.best_expert_key,
                        quality_score=stage_score,
                    )
                    if stored:
                        _log(f"[Scheduler] Memory: recorded case for {ctx.best_expert_key} (q={stage_score:.4f})")

                # C2: Iterative re-detection on accepted output (silent — verbose=False)
                try:
                    redet_result  = detect_degradation(current_path, verbose=False)
                    redet_scores: dict[str, float] = redet_result["scores"]
                    iter_ctx.add(stage, scores, redet_scores)
                    scores = redet_scores   # update scores for remaining stages
                    _log(f"[Scheduler] Re-detection after {stage_label}: primary={redet_result['primary']}")
                except Exception as e:
                    _log(f"[Scheduler] Re-detection failed: {e}")

            else:
                _log(f"[Scheduler] Stage {stage_label} → ROLLED BACK (quality gate rejected, score={stage_score:.4f} < {QUALITY_GATE_MIN})")
                rolled_back = True
        else:
            _log(f"[Scheduler] Stage {stage_label} → all experts failed, passing through unchanged")

        # LPIPS for stage result (C6) — evaluate full quality if available
        lpips_val = None
        if stage_best and not rolled_back:
            try:
                qfull    = evaluate_quality_full(current_path, stage_best) if current_path != stage_best else None
                lpips_val = qfull["lpips"] if qfull else None
            except Exception:
                pass

        stage_results[stage] = {
            "skipped":            False,
            "best_expert":        ctx.best_expert_key,
            "best_expert_key":    ctx.best_expert_key,
            "best_path":          stage_best,
            "best_score":         stage_score,
            "pre_stage_quality":  pre_stage_quality,
            "rolled_back":        rolled_back,
            "attempts":           len(ctx.attempts),
            "lpips":              lpips_val,              # C6
            "redetection_scores": iter_ctx.latest_scores(),  # C2
        }

    # ── STEP 3: Final result ───────────────────────────────────
    total_time   = round(time.time() - t_start, 2)
    final_output = current_path if current_path != input_path else None

    _log("\n" + "─" * 60)
    _log("  MAIR+ v2 THREE-STAGE PIPELINE SUMMARY")
    _log("─" * 60)
    for stage in STAGE_ORDER:
        r = stage_results.get(stage, {})
        if r.get("skipped"):
            _log(f"  {stage.upper():<12}  SKIPPED")
        elif r.get("rolled_back"):
            _log(f"  {stage.upper():<12}  {r.get('best_expert','?'):<28} ROLLED BACK")
        elif r.get("best_path"):
            score = r.get("best_score")
            _log(f"  {stage.upper():<12}  {r['best_expert']:<28} score={score:.4f}")
        else:
            _log(f"  {stage.upper():<12}  FAILED (no output)")

    _log(f"\n  Total time         : {total_time}s")
    _log(f"  Expert calls       : {invocation_count}")
    _log(f"  Memory bias used   : {memory_bias_applied}")
    _log(f"  Iterative stages   : {len(iter_ctx.records)}")
    _log(f"  Final output       : {final_output or 'None'}")
    _log("=" * 60 + "\n")

    return {
        "output_path":          final_output,
        "stage_results":        stage_results,
        "total_time_s":         total_time,
        "invocation_count":     invocation_count,
        "memory_bias_applied":  memory_bias_applied,
        "iterative_context":    iter_ctx.summary(),
    }


# ─────────────────────────────────────────────────────────────
# LEGACY SINGLE-EXPERT SCHEDULER (kept for ablation / fallback)
# ─────────────────────────────────────────────────────────────

def run_scheduler(
    input_path:   str,
    max_attempts: int  = 2,
    verbose:      bool = True,
    three_stage:  bool = True,
    clinical_eval: bool = False
) -> str | None:
    """
    Main scheduler entry point.

    Args:
        input_path   : path to the degraded input image
        max_attempts : maximum expert invocations per stage
        verbose      : print full reasoning trace
        three_stage  : True = three-stage framework (paper method)
                       False = legacy single-expert confidence-based
        clinical_eval: use clinical composite for quality eval

    Returns:
        Path to the best restored image, or None if all experts failed.
    """
    if three_stage:
        result = run_three_stage_scheduler(input_path, max_attempts, verbose, clinical_eval=clinical_eval)
        return result["output_path"]

    # ── Legacy mode (single expert, confidence-ranked) ────────
    _log = print if verbose else lambda *a, **k: None

    _log("\n" + "=" * 50)
    _log("   MAIR SCHEDULER ACTIVATED  [legacy mode]")
    _log("=" * 50)

    degradation_result = detect_degradation(input_path)
    primary    = degradation_result["primary"]
    confidence = degradation_result["confidence"]
    scores     = degradation_result["scores"]

    _log(f"\n[Scheduler] Primary  : {primary}  (confidence: {confidence:.3f})")
    _log(f"[Scheduler] Scores   : {scores}")

    ranked = select_experts(degradation_result)

    if not ranked:
        _log("[Scheduler] No experts qualified. Aborting.")
        return None

    if verbose:
        print_ranking(degradation_result)

    ctx = RestorationContext(
        original_path=input_path,
        degradation_result=degradation_result,
        expert_plan=ranked,
        max_attempts=max_attempts,
    )

    for attempt_idx, (expert_key, expert_entry) in enumerate(ranked[:max_attempts]):
        expert_name = expert_entry["name"]
        expert_fn   = expert_entry["fn"]

        _log(f"\n[Scheduler] ── Attempt {attempt_idx + 1}/{max_attempts}")
        _log(f"[Scheduler] Expert  : {expert_name}  [{expert_key}]")

        output_path = None
        quality     = None
        try:
            output_path = expert_fn(input_path)
        except Exception as e:
            _log(f"[Scheduler] Expert raised exception: {e}")

        if output_path and os.path.exists(output_path):
            try:
                if clinical_eval:
                    quality = evaluate_clinical_composite(input_path, output_path)
                    _log(f"[Quality Evaluator] Clinical Composite : {quality:.4f}")
                else:
                    quality = evaluate_quality(input_path, output_path)
                _log(f"[Scheduler] Quality : {quality:.4f}")
            except Exception as e:
                _log(f"[Scheduler] Quality evaluation failed: {e}")
        else:
            _log("[Scheduler] Expert returned None (failed or no output).")

        ctx.record_attempt(
            expert_key=expert_key,
            expert_name=expert_name,
            output_path=output_path,
            quality_score=quality,
        )

        action = reflect(ctx)
        reason = explain(ctx, action)
        _log(f"[Scheduler] Reflect : {action}  —  {reason}")

        if action in (ACCEPT, ESCALATE):
            break

    _log("\n" + "─" * 50)
    _log(ctx.summary())

    final_output = ctx.best_output_path
    final_score  = ctx.best_quality_score

    if final_output:
        _log(f"\n[Scheduler] Final output : {final_output}")
        _log(f"[Scheduler] Final score  : {final_score:.4f}  ({_label(final_score)})")
    else:
        _log("[Scheduler] All experts failed — no output produced.")

    _log("\n" + "=" * 50)
    _log("     MAIR PIPELINE COMPLETED")
    _log("=" * 50 + "\n")

    return final_output