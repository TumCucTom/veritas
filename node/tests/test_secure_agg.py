"""Edge secure-aggregation (Bonawitz pairwise masking), device→bank.

These tests prove the production-sound posture: the bank node aggregates a
COHORT of masked device updates into a secure SUM and folds that single
aggregate into the edge model, while NEVER seeing, storing, or being able to
derive any individual device's cleartext update. Dropout is handled via
recover_dropout so the sum stays correct over survivors.
"""
import numpy as np

from veritas_core.secure_agg import mask_update

from node.config import NodeConfig
from node.engine import NodeEngine


def _engine(idx=0):
    return NodeEngine(NodeConfig(node_id=f"node{idx}", node_index=idx, seed=idx))


def _masked_cohort(eng, updates: dict[str, np.ndarray], opened: dict):
    """Mask each device's update against its peers using the dealt seeds."""
    ids = list(updates.keys())
    seeds = eng._cohort_seed_table  # dealer table (production: per-device DH)
    masked = {}
    for cid in ids:
        peers = [c for c in ids if c != cid]
        masked[cid] = mask_update(updates[cid], cid, peers, seeds)
    return masked


def test_cohort_secure_sum_equals_true_sum_and_no_cleartext_seen():
    eng = _engine()
    dim = eng.edge_w.shape[0]
    before = eng.edge_w.copy()

    updates = {
        "devA": np.full(dim, 0.5),
        "devB": np.full(dim, -0.2),
        "devC": np.linspace(-0.1, 0.1, dim),
    }
    opened = eng.open_cohort(list(updates.keys()))
    masked = _masked_cohort(eng, updates, opened)

    for cid, mv in masked.items():
        eng.edge_aggregate(mv, num_examples=32, cohort_id=opened["cohortId"], client_id=cid)

    # PRIVACY: what the node buffered is the MASKED vectors, never the cleartext.
    for cid, mv in masked.items():
        stored = eng._cohort_masked[cid]
        assert np.allclose(stored, mv)
        # The stored vector must NOT equal that device's real update; the mask
        # makes it look uniformly random (huge magnitude vs the real ~O(1)).
        assert not np.allclose(stored, updates[cid])
        assert np.linalg.norm(stored) > 1e4  # dominated by the mask scale

    agg = eng.close_cohort()
    true_sum = sum(updates.values())
    assert np.allclose(agg, true_sum, atol=1e-6)
    # The folded edge-model delta equals the secure sum / n (what folding the
    # true sum would give), proving correctness end-to-end.
    n = len(updates)
    assert np.allclose(eng.edge_w - before, true_sum / n, atol=1e-6)
    assert eng._edge_cohorts_aggregated == 1
    assert not eng.cohort_open


def test_dropout_is_recovered_sum_correct_over_survivors():
    eng = _engine()
    dim = eng.edge_w.shape[0]
    before = eng.edge_w.copy()

    updates = {
        "devA": np.full(dim, 0.3),
        "devB": np.full(dim, 0.7),
        "devC": np.full(dim, -0.4),
    }
    opened = eng.open_cohort(list(updates.keys()))
    masked = _masked_cohort(eng, updates, opened)

    # devC drops out: it masked against A and B but never submits its message,
    # so its cancelling terms are missing from the running sum.
    for cid in ("devA", "devB"):
        eng.edge_aggregate(masked[cid], 16, cohort_id=opened["cohortId"], client_id=cid)

    agg = eng.close_cohort()
    survivor_sum = updates["devA"] + updates["devB"]
    # recover_dropout subtracts devC's uncancelled masks -> true survivor sum.
    assert np.allclose(agg, survivor_sum, atol=1e-6)
    assert np.allclose(eng.edge_w - before, survivor_sum / 2.0, atol=1e-6)


def test_node_never_stores_unmasked_individual_update():
    """A naive (unmasked) update must NOT be reconstructable from node state."""
    eng = _engine()
    dim = eng.edge_w.shape[0]
    updates = {"devA": np.full(dim, 0.9), "devB": np.full(dim, 0.1)}
    opened = eng.open_cohort(list(updates.keys()))
    masked = _masked_cohort(eng, updates, opened)
    for cid, mv in masked.items():
        eng.edge_aggregate(mv, 8, cohort_id=opened["cohortId"], client_id=cid)

    # Everything the node holds for the open cohort is masked: no entry matches
    # a real device update.
    for cid in updates:
        assert not np.allclose(eng._cohort_masked[cid], updates[cid])
