"""Stage 1.B — Shapley target localization (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


RolloutFn = Callable[[List[int]], List[Dict]]
PenaltyFn = Callable[[List[Dict]], float]


@dataclass
class ShapleyResult:
    target_id: int
    target_class: str
    phi: Dict[int, float]
    samples: int
    target_index_in_pool: int


def shapley_target_localization(actor_pool, rollout, penalty,
                                K: int = 16, rng=None) -> ShapleyResult:
    n = len(actor_pool)
    if n == 0:
        raise ValueError("actor_pool is empty.")
    rng = rng or np.random.default_rng(0)

    phi = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for _ in range(int(K)):
        perm = rng.permutation(n)
        kept = []
        prev = penalty(rollout(kept))
        for idx in perm:
            kept_with = kept + [int(idx)]
            cur = penalty(rollout(kept_with))
            phi[idx] += (cur - prev)
            counts[idx] += 1
            kept = kept_with
            prev = cur

    mean_phi = phi / np.maximum(counts, 1)
    target_idx = int(np.argmax(mean_phi))
    cls, inst = actor_pool[target_idx]
    return ShapleyResult(
        target_id=int(inst), target_class=str(cls),
        phi={int(actor_pool[i][1]): float(mean_phi[i]) for i in range(n)},
        samples=int(K), target_index_in_pool=target_idx,
    )
