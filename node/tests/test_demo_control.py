"""Node reactions to the control plane's demo-control flags.

The plane piggybacks the live RACE state on the EXISTING federation poll
response (``GET /v1/rounds/current``): ``campaignActive`` / ``attackMemberId`` /
``epoch``. The node reads these in its poll loop and reacts truthfully so the
plane's aggregate reflects a REAL federation — no new push channel. These tests
exercise every reaction against the in-memory fake plane (no network).
"""
import numpy as np

from veritas_core.data import make_bank_data

from node.config import NodeConfig
from node.runtime import NodeRuntime

from .fake_plane import FakePlane


# ---- enrol metadata -------------------------------------------------------

def test_enroll_sends_display_name_and_customers():
    """Enrolment carries the REAL bank name + customer count (engine metadata)."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)
    rt.ensure_enrolled()

    body = plane.last_enroll_body
    assert body is not None
    # engine.name / engine.customers (NAMES[0]="Barclays", CUST[0]=2_100_000)
    assert body["displayName"] == rt.engine.name == "Barclays"
    assert body["customers"] == rt.engine.customers == 2_100_000


# ---- campaign flag --------------------------------------------------------

def test_campaign_flag_injects_campaign_locally_once():
    """campaignActive=True → node injects the campaign locally, exactly once."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)

    assert rt.engine.campaign_active is False
    plane.campaign_active = True

    r1 = rt.federate_once()  # noqa: F841 (reaction is on the engine)
    assert rt.engine.campaign_active is True
    assert rt.client._campaign_injected_epoch == plane.epoch

    # Snapshot eval set, run again: must NOT re-inject (idempotent per epoch).
    eval_y_before = rt.engine.eval_y.copy()
    rt.federate_once()
    assert np.array_equal(rt.engine.eval_y, eval_y_before)


def test_blind_node_injects_eval_only_so_silo_stays_blind():
    """The designated blind node injects the campaign EVAL-ONLY (siloed stays blind).

    The load-bearing, DETERMINISTIC difference vs a seeing node: a blind node's
    TRAINING set gains NO campaign examples (so its siloed model can never learn
    the typology), while its EVAL set IS targeted — only the FEDERATED model,
    learning from seeing peers, can flag it. A seeing node injects into BOTH.
    """
    # --- blind node 0: campaign in eval only, NOT in training ---
    blind_plane = FakePlane(min_updates=1, sigma=0.0)
    blind = NodeRuntime(
        NodeConfig(node_id="node0", node_index=0, seed=0,
                   autostart_federation=False, blind_node=0),
        transport=blind_plane,
    )
    train_pos_before = int(blind.engine.train_y.sum())
    eval_pos_before = int(blind.engine.eval_y.sum())
    blind_plane.campaign_active = True
    blind.federate_once()

    assert blind.engine.campaign_active is True
    # Blind: training UNCHANGED (siloed baseline never sees the campaign)...
    assert int(blind.engine.train_y.sum()) == train_pos_before
    # ...but eval IS targeted (the campaign hits this bank's customers).
    assert int(blind.engine.eval_y.sum()) > eval_pos_before

    # --- a SEEING node at the same index DOES inject into training ---
    seeing_plane = FakePlane(min_updates=1, sigma=0.0)
    seeing = NodeRuntime(
        NodeConfig(node_id="node0", node_index=0, seed=0,
                   autostart_federation=False, blind_node=None),
        transport=seeing_plane,
    )
    seeing_train_before = int(seeing.engine.train_y.sum())
    seeing_plane.campaign_active = True
    seeing.federate_once()
    assert int(seeing.engine.train_y.sum()) > seeing_train_before

    # The demo direction holds for the blind node: its federated recall is never
    # below its siloed recall on the campaign-targeted eval set (only the
    # federated model can pick up the typology it was never trained on).
    plane = FakePlane(min_updates=1, sigma=0.0)
    b = NodeRuntime(
        NodeConfig(node_id="node0", node_index=0, seed=0,
                   autostart_federation=False, blind_node=0),
        transport=plane,
    )
    peer = NodeRuntime(
        NodeConfig(node_id="node1", node_index=1, seed=1,
                   autostart_federation=False, blind_node=0),
        transport=plane,
    )
    plane.campaign_active = True
    for _ in range(8):
        peer.federate_once()
        b.federate_once()
    fed, silo = b.engine._det()
    assert fed >= silo


# ---- attack flag ----------------------------------------------------------

def test_attack_flag_poisons_submitted_update():
    """attackMemberId == my id → my submitted update is sign-flipped + amplified."""
    # Honest baseline: same node/plane/seed, NO attack.
    honest_plane = FakePlane(min_updates=1, sigma=0.0)
    honest_cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    honest = NodeRuntime(honest_cfg, transport=honest_plane)
    honest.federate_once()
    honest_update = np.asarray(honest_plane.last_update_body["update"], dtype=np.float64)

    # Attack run: plane designates THIS node as the attacker.
    atk_plane = FakePlane(min_updates=1, sigma=0.0)
    atk_plane.attack_member_id = "node0"
    atk_cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    atk = NodeRuntime(atk_cfg, transport=atk_plane)
    res = atk.federate_once()  # noqa: F841

    poisoned = np.asarray(atk_plane.last_update_body["update"], dtype=np.float64)
    # poisoned_update = -10 * honest: much larger norm AND sign-flipped.
    assert np.linalg.norm(poisoned) > np.linalg.norm(honest_update) * 5
    assert float(np.dot(poisoned, honest_update)) < 0  # opposite direction


def test_attack_flag_targets_only_the_named_member():
    """A node NOT named by attackMemberId submits an honest (un-poisoned) update."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    plane.attack_member_id = "someone-else"
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)

    baseline_plane = FakePlane(min_updates=1, sigma=0.0)
    baseline = NodeRuntime(
        NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False),
        transport=baseline_plane,
    )
    rt.federate_once()
    baseline.federate_once()

    targeted = np.asarray(plane.last_update_body["update"], dtype=np.float64)
    honest = np.asarray(baseline_plane.last_update_body["update"], dtype=np.float64)
    # Not targeted → identical to the honest baseline.
    assert np.allclose(targeted, honest)


# ---- epoch reset ----------------------------------------------------------

def test_epoch_bump_resets_local_state_to_genesis():
    """epoch change → engine resets (campaign off, fresh weights/data, counters 0)."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)

    # Inject a campaign and advance some rounds so state is non-genesis.
    plane.campaign_active = True
    for _ in range(3):
        rt.federate_once()
    assert rt.engine.campaign_active is True
    assert rt.engine.eval_y.sum() > make_bank_data(1000, 0.05, seed=cfg.seed + 100)[1].sum()

    # Plane bumps the epoch (reset beat) and clears the campaign for the re-run.
    plane.epoch = 1
    plane.campaign_active = False
    rt.federate_once()

    assert rt.client.current_epoch == 1
    # Genesis: campaign cleared, injection bookkeeping cleared.
    assert rt.engine.campaign_active is False
    assert rt.client._campaign_injected_epoch is None
    # Eval set is back to the pristine (campaign-free) baseline.
    pristine_X, pristine_y = make_bank_data(1000, 0.05, seed=cfg.seed + 100)
    assert np.array_equal(rt.engine.eval_y, pristine_y)
    # Counters reflect at most the single post-reset round, not the prior run's
    # accumulation (which over 3 campaign rounds was far larger).
    assert rt.engine.cum["silo"] <= 1500.0 * 1.0 + 1e-6


def test_epoch_bump_then_campaign_reinjects_for_new_epoch():
    """After a reset, campaignActive in the NEW epoch injects again (re-run works)."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)

    plane.campaign_active = True
    rt.federate_once()
    assert rt.engine.campaign_active is True

    # Reset to a new epoch, campaign off.
    plane.epoch = 1
    plane.campaign_active = False
    rt.federate_once()
    assert rt.engine.campaign_active is False

    # New campaign in the new epoch → injects again.
    plane.campaign_active = True
    rt.federate_once()
    assert rt.engine.campaign_active is True
    assert rt.client._campaign_injected_epoch == 1
