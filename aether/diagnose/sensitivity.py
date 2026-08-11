"""Stage 4 -- normalised sensitivity from PGD finite-difference grads.

Per-axis sensitivity:

    s_i = |grad_i| * range_i / (sum_j |grad_j| * range_j)

Returns the 13-D normalised vector, plus topo / photo aggregates and
the dominance ratio r_vt = ||s_topo|| / ||s_photo||.

Confidence proxy:

    confidence = SNR / (SNR + 1)

where SNR is the variance ratio of the per-iteration FD gradient over
the noise floor (mean abs of inactive-axis FD).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .pgd import DualTrackResult, TrackResult
from .semantics import (
    SEMANTIC_VAR_NAMES, SEMANTIC_BOUNDS,
    TOPO_DIMS, PHOTO_DIMS, DIM, per_axis_range,
)


@dataclass
class Sensitivities:
    s: np.ndarray                       # (13,)
    s_named: Dict[str, float]
    topo_norm: float
    photo_norm: float
    r_vt_ratio: float                   # ||s_topo|| / ||s_photo||
    G_topo: float
    G_photo: float
    confidence: float


def _grad_to_weighted(grad: np.ndarray) -> np.ndarray:
    return np.abs(grad) * per_axis_range()


def compute_normalized_sensitivities(dual: DualTrackResult) -> Sensitivities:
    g_topo = _grad_to_weighted(dual.topo.grad_final)
    g_photo = _grad_to_weighted(dual.photo.grad_final)
    s_unnorm = g_topo + g_photo
    s_sum = float(np.sum(s_unnorm))
    s = s_unnorm / s_sum if s_sum > 1e-9 else np.zeros(DIM, dtype=np.float64)

    topo_norm = float(np.linalg.norm(s[list(TOPO_DIMS)]))
    photo_norm = float(np.linalg.norm(s[list(PHOTO_DIMS)]))
    r_vt = float(topo_norm / photo_norm) if photo_norm > 1e-9 else float("inf")

    G_topo = float(np.sum(s[list(TOPO_DIMS)]))
    G_photo = float(np.sum(s[list(PHOTO_DIMS)]))

    confidence = _confidence_score(dual.topo) * _confidence_score(dual.photo)
    confidence = float(confidence ** 0.5)

    return Sensitivities(
        s=s,
        s_named={n: float(s[i]) for i, n in enumerate(SEMANTIC_VAR_NAMES)},
        topo_norm=topo_norm, photo_norm=photo_norm, r_vt_ratio=r_vt,
        G_topo=G_topo, G_photo=G_photo, confidence=confidence,
    )


def _confidence_score(track: TrackResult) -> float:
    if not track.grad_history:
        return 0.0
    grads = np.stack(track.grad_history, axis=0)  # (iters, DIM)
    sig = float(np.mean(np.abs(grads)))
    noise = float(np.std(grads)) + 1e-9
    snr = sig / noise
    return float(snr / (snr + 1.0))
