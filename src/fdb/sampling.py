"""Seeded stratified sampling: a small subset that keeps the parent's mix (T20a)."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Iterable


def stratified_sample(ids: Iterable[str], stratum: Callable[[str], Hashable], n: int, seed: int) -> list[str]:
    """Pick exactly n IDs so each stratum keeps its share of the parent, seeded and order-independent."""
    strata: dict[Hashable, list[str]] = defaultdict(list)
    for image_id in sorted(ids):
        strata[stratum(image_id)].append(image_id)
    total = sum(len(v) for v in strata.values())
    if not 0 <= n <= total:
        raise ValueError(f"cannot sample {n} from {total} images")

    quota = {k: n * len(v) / total for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    leftover = n - sum(take.values())
    for k in sorted(quota, key=lambda k: (take[k] - quota[k], str(k)))[:leftover]:
        take[k] += 1

    rng = random.Random(seed)
    picked = []
    for k in sorted(strata, key=str):
        picked += rng.sample(strata[k], take[k])
    return sorted(picked)