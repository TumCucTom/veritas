"""Tests proving the secure-aggregation security + correctness properties.

These tests prove, for the Bonawitz-style pairwise-masking protocol in
``veritas_core.secure_agg``:
  * masks cancel  -> server recovers exactly the true sum;
  * server sees noise -> each individual masked vector is far from the raw
    update, so an honest-but-curious server learns nothing about one client;
  * dropout recovery -> correct sum over survivors after a client drops;
  * unrecovered dropout is garbage -> the recovery path is genuinely working.
"""
import numpy as np
import pytest

from veritas_core.secure_agg import (
    derive_pairwise_mask,
    establish_pairwise_seeds,
    mask_update,
    recover_dropout,
    secure_aggregate,
    secure_sum,
)

DIM = 11  # FEATURE_DIM + 1, matches the Veritas model weight vector.


def _make_updates(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"c{i}" for i in range(n)]
    # Realistic-scale (small) model deltas, the kind ``edge_aggregate`` folds in.
    updates = [rng.normal(0, 0.5, size=dim) for _ in range(n)]
    return ids, updates


def _mask_all(ids, updates, seeds):
    masked = []
    for cid, u in zip(ids, updates):
        peers = [c for c in ids if c != cid]
        masked.append(mask_update(u, cid, peers, seeds))
    return masked


def test_prg_is_deterministic_and_symmetric():
    seed = b"shared-secret-seed-xyz"
    a = derive_pairwise_mask(seed, DIM)
    b = derive_pairwise_mask(seed, DIM)
    assert np.allclose(a, b)  # both clients derive the identical mask
    assert a.shape == (DIM,)
    # Different seeds -> different masks.
    c = derive_pairwise_mask(b"other-seed", DIM)
    assert not np.allclose(a, c)


def test_masks_cancel_secure_sum_equals_plain_sum():
    """n=5: secure_sum over masked updates == plain sum of raw updates."""
    ids, updates = _make_updates(5, seed=1)
    rng = np.random.default_rng(123)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    masked = _mask_all(ids, updates, seeds)

    recovered = secure_sum(masked)
    truth = np.sum(np.stack(updates), axis=0)
    assert np.allclose(recovered, truth, atol=1e-6), (
        "pairwise masks must cancel exactly -> server recovers the true sum"
    )


def test_server_sees_noise_individual_masked_far_from_raw():
    """Each individual masked vector must be dominated by the mask.

    An honest-but-curious server inspecting one client's message must not be
    able to read off its update: the masked vector is far (huge L2 distance)
    from the raw update, looking essentially random.
    """
    ids, updates = _make_updates(5, seed=2)
    rng = np.random.default_rng(7)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    masked = _mask_all(ids, updates, seeds)

    for raw, mv in zip(updates, masked):
        dist = float(np.linalg.norm(mv - raw))
        raw_norm = float(np.linalg.norm(raw))
        # Mask scale ~1e6 vs update scale ~O(1): distance must be enormous.
        assert dist > 1e4, "mask must dominate -> server learns nothing"
        assert dist > 1000 * (raw_norm + 1e-9)


def test_dropout_recovery_yields_survivor_sum():
    """One client drops after masking; recovery yields the survivors' sum."""
    ids, updates = _make_updates(5, seed=3)
    rng = np.random.default_rng(99)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    masked = _mask_all(ids, updates, seeds)

    dropped = ids[2]
    survivor_ids = [c for c in ids if c != dropped]
    survivor_masked = [m for c, m in zip(ids, masked) if c != dropped]
    survivor_updates = [u for c, u in zip(ids, updates) if c != dropped]

    running = secure_sum(survivor_masked)  # carries uncancelled masks
    corrected = recover_dropout(running, [dropped], survivor_ids, seeds, DIM)

    truth = np.sum(np.stack(survivor_updates), axis=0)
    assert np.allclose(corrected, truth, atol=1e-6), (
        "after subtracting the dropped client's leftover masks, the sum over "
        "the survivors must be exact"
    )


def test_unrecovered_dropout_is_garbage():
    """Sanity: without recovery, the survivor sum is corrupted (mask remains)."""
    ids, updates = _make_updates(5, seed=4)
    rng = np.random.default_rng(55)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    masked = _mask_all(ids, updates, seeds)

    dropped = ids[1]
    survivor_masked = [m for c, m in zip(ids, masked) if c != dropped]
    survivor_updates = [u for c, u in zip(ids, updates) if c != dropped]

    running = secure_sum(survivor_masked)  # NOT recovered
    truth = np.sum(np.stack(survivor_updates), axis=0)
    # The leftover mask is ~1e6 scale, so the unrecovered sum is wildly off.
    assert float(np.linalg.norm(running - truth)) > 1e4, (
        "recovery must actually do work: unrecovered dropout is garbage"
    )


def test_secure_aggregate_convenience_matches_truth():
    ids, updates = _make_updates(6, seed=8)
    rng = np.random.default_rng(11)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    out = secure_aggregate(updates, ids, seed_table=seeds)
    truth = np.sum(np.stack(updates), axis=0)
    assert np.allclose(out, truth, atol=1e-6)


def test_multi_dropout_recovery():
    """Two simultaneous dropouts are recoverable."""
    ids, updates = _make_updates(6, seed=12)
    rng = np.random.default_rng(21)
    seeds = establish_pairwise_seeds(ids, master_rng=rng)
    masked = _mask_all(ids, updates, seeds)

    dropped = [ids[0], ids[4]]
    survivor_ids = [c for c in ids if c not in dropped]
    survivor_masked = [m for c, m in zip(ids, masked) if c not in dropped]
    survivor_updates = [u for c, u in zip(ids, updates) if c not in dropped]

    running = secure_sum(survivor_masked)
    corrected = recover_dropout(running, dropped, survivor_ids, seeds, DIM)
    truth = np.sum(np.stack(survivor_updates), axis=0)
    assert np.allclose(corrected, truth, atol=1e-6)
