"""
Case Store — Memory-Augmented MAIR (C9)
========================================
MAIR+ Contribution C9: Online Case-Based Reasoning.

The CaseStore records successful restoration outcomes and retrieves
similar past cases to bias expert selection for future images.

Key Design Decisions (per the critical review):
  - Records ONE case PER STAGE (not per pipeline run) so blur cases
    inform blur-stage queries and lowlight cases inform scene-stage queries.
  - Uses the DEGRADATION SCORES AT THE TIME OF THAT STAGE as the fingerprint
    (not the initial pipeline-entry scores), so iterative re-detection (C2)
    updates the fingerprint correctly.
  - Only records cases with quality_score >= QUALITY_GATE (0.65).
  - Fingerprint is a 6D vector: [blur, sr, jpeg, denoise, lowlight, haze].
  - Retrieval uses cosine similarity; only cases with sim >= min_similarity
    contribute to the bias dict.
  - Stored in outputs/memory/case_memory.json (gitignored).

Usage:
    store = CaseStore()
    store.record(scores, stage="imaging", expert_key="restormer_deblur", quality=0.83)
    bias  = store.retrieve(scores, stage="imaging")  → {expert_key: similarity}
"""

import os
import json
import math
import time
from datetime import datetime


MEMORY_PATH   = os.path.join("outputs", "memory", "case_memory.json")
QUALITY_GATE  = 0.65    # minimum quality_score to store a case
MIN_SIMILARITY = 0.80   # minimum cosine similarity to contribute to bias
TOP_K         = 5       # number of similar cases to retrieve
SCORE_KEYS    = ["blur", "sr", "jpeg", "denoise", "lowlight", "haze", "rain"]  # Phase 3: added rain


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _to_vector(scores: dict) -> list[float]:
    """Convert a degradation scores dict to a fixed-length float vector."""
    return [float(scores.get(k, 0.0)) for k in SCORE_KEYS]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float lists."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    denom = mag_a * mag_b
    if denom < 1e-8:
        return 0.0
    return max(0.0, min(1.0, dot / denom))


# ─────────────────────────────────────────────────────────────
# CASE STORE
# ─────────────────────────────────────────────────────────────

class CaseStore:
    """
    Persistent case memory for MAIR+ restoration outcomes.

    Each case entry:
        {
            "fingerprint":    [float × 6]   — degradation score vector
            "stage":          str            — "compression" | "imaging" | "scene"
            "expert_key":     str            — e.g. "restormer_deblur"
            "quality_score":  float          — composite quality score at accept
            "timestamp":      str            — ISO timestamp
        }
    """

    def __init__(self, memory_path: str = MEMORY_PATH):
        self.memory_path = memory_path
        self._cases: list[dict] = []
        self._load()

    # ── Persistence ─────────────────────────────────────────

    def _load(self):
        """Load cases from disk."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                self._cases = data.get("cases", [])
            except (json.JSONDecodeError, IOError):
                self._cases = []
        else:
            self._cases = []

    def _save(self):
        """Persist cases to disk."""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump({"cases": self._cases, "version": "1.0"}, f, indent=2)

    # ── Recording ────────────────────────────────────────────

    def record(
        self,
        degradation_scores: dict,
        stage:              str,
        expert_key:         str,
        quality_score:      float,
    ) -> bool:
        """
        Record a successful restoration case if quality is above the gate.

        Args:
            degradation_scores : dict of {signal: score} at the time of this stage
            stage              : pipeline stage ("compression" | "imaging" | "scene")
            expert_key         : key of the expert that was accepted
            quality_score      : composite quality score of the accepted output

        Returns:
            True if the case was stored, False if quality was below gate.
        """
        if quality_score < QUALITY_GATE:
            return False

        case = {
            "fingerprint":   _to_vector(degradation_scores),
            "stage":         stage,
            "expert_key":    expert_key,
            "quality_score": round(quality_score, 4),
            "timestamp":     datetime.now().isoformat(),
        }
        self._cases.append(case)
        self._save()
        return True

    # ── Retrieval ────────────────────────────────────────────

    def retrieve(
        self,
        degradation_scores: dict,
        stage:              str,
        top_k:             int   = TOP_K,
        min_similarity:    float = MIN_SIMILARITY,
    ) -> list[dict]:
        """
        Retrieve the top-K most similar past cases for a given stage.

        Args:
            degradation_scores : current image's degradation score dict
            stage              : restrict to cases from this stage only
            top_k              : maximum number of cases to return
            min_similarity     : minimum cosine similarity threshold

        Returns:
            List of dicts: [{expert_key, similarity, quality_score}, ...]
            Sorted by similarity descending. Empty if no match.
        """
        query_vec = _to_vector(degradation_scores)
        stage_cases = [c for c in self._cases if c.get("stage") == stage]

        scored = []
        for case in stage_cases:
            sim = _cosine_similarity(query_vec, case["fingerprint"])
            if sim >= min_similarity:
                scored.append({
                    "expert_key":    case["expert_key"],
                    "similarity":    round(sim, 4),
                    "quality_score": case["quality_score"],
                })

        # Sort by similarity descending
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    # ── Utilities ────────────────────────────────────────────

    def clear(self):
        """Remove all stored cases."""
        self._cases = []
        self._save()

    def stats(self) -> dict:
        """Return memory statistics."""
        if not self._cases:
            return {"count": 0, "avg_quality": None, "most_used_expert": None}

        total_q = sum(c["quality_score"] for c in self._cases)
        expert_counts: dict[str, int] = {}
        for c in self._cases:
            expert_counts[c["expert_key"]] = expert_counts.get(c["expert_key"], 0) + 1
        most_used = max(expert_counts, key=lambda k: expert_counts[k])

        # Stage breakdown
        stages: dict[str, int] = {}
        for c in self._cases:
            stages[c["stage"]] = stages.get(c["stage"], 0) + 1

        return {
            "count":             len(self._cases),
            "avg_quality":       round(total_q / len(self._cases), 4),
            "most_used_expert":  most_used,
            "expert_counts":     expert_counts,
            "stage_counts":      stages,
        }

    def __repr__(self):
        s = self.stats()
        return f"CaseStore(count={s['count']}, avg_q={s['avg_quality']}, path={self.memory_path})"
