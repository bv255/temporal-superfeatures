"""
Deterministic seed derivation shared across the GA's per-fold, per-individual/generation
(GBT), and per-baseline seeding (see config.py's GAConfig.random_seed and ga/engine.py's
GeneticAlgorithm1.fold_seed). Uses hashlib rather than Python's built-in hash(), which is
randomized per-process via PYTHONHASHSEED and would silently break reproducibility across
separate runs/processes even when every input is identical.
"""
import hashlib


def derive_seed(*parts) -> int:
    """
    Deterministically derive a 31-bit non-negative int seed from an arbitrary sequence of
    hashable-as-string parts (e.g. a base seed, a fold name, a generation index, an
    individual's repr) - same parts always produce the same seed, different parts are
    well-decorrelated from each other regardless of call order.
    """
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
