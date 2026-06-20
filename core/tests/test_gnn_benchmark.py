"""Tests for the reusable federated-GNN benchmark (veritas_core.gnn_benchmark).

These assert the dict carries every field the web's summarizeGnnBenchmark reads,
that the measured metrics show the real federated lift, that benchmark_current
tracks the REAL round trajectory, and that the whole thing is JSON-serializable.
"""
import json

import pytest

from veritas_core.gnn_benchmark import compute_gnn_benchmark, benchmark_current


@pytest.fixture(scope="module")
def benchmark():
    # Smaller/faster run; still real training, still shows the lift.
    return compute_gnn_benchmark(seed=0, rounds=15)


def test_shape_has_every_field_the_web_reads(benchmark):
    b = benchmark
    # source.path, source.seed
    assert b["source"]["path"] == "core/experiments/federated_gnn.py"
    assert b["source"]["seed"] == 0
    assert isinstance(b["source"]["generatedAt"], str)
    # graph.banks / accounts / campaigns (+ crossBankEdges, alertBudgetPct)
    g = b["graph"]
    assert g["banks"] == 4 and g["accounts"] == 700 and g["campaigns"] == 8
    assert g["crossBankEdges"] > 0
    assert g["alertBudgetPct"] == 15
    # training.aggregation / privacy / verification
    t = b["training"]
    assert t["aggregation"] == "Multi-Krum"
    assert t["privacy"] == "DP clip/noise"
    assert t["verification"] == "VSA bounded-norm proof"
    assert t["rounds"] == 15
    # metrics.siloedRecall / federatedRecall / siloedAuc / federatedAuc
    m = b["metrics"]
    for k in ("siloedRecall", "federatedRecall", "siloedAuc", "federatedAuc"):
        assert isinstance(m[k], float)
    # real per-round trajectory
    assert isinstance(b["roundTrajectory"], list)
    assert len(b["roundTrajectory"]) == t["rounds"]


def test_metrics_show_real_federated_lift(benchmark):
    m = benchmark["metrics"]
    assert m["federatedRecall"] > m["siloedRecall"]
    lift = m["federatedRecall"] - m["siloedRecall"]
    assert lift > 0.2, f"federated lift too small: {lift}"
    # AUC should also favour federated.
    assert m["federatedAuc"] > m["siloedAuc"]


def test_benchmark_current_rises_with_round_along_trajectory(benchmark):
    b = benchmark
    # Before campaign / round 0: starts low at trajectory[0].
    c0 = benchmark_current(b, 0, campaignActive=False)
    assert c0["round"] == 0
    assert c0["progress"] == 0.0
    assert abs(c0["federatedRecall"] - b["roundTrajectory"][0]) < 1e-9

    # Rising along the real trajectory as the round advances under campaign.
    # The real (DP-noised) trajectory may have small local dips, so we assert the
    # OVERALL trend rises: early rounds are well below late rounds, and the
    # interpolated value tracks the underlying samples.
    rounds = b["training"]["rounds"]
    early = benchmark_current(b, 1, campaignActive=True)["federatedRecall"]
    late = benchmark_current(b, rounds, campaignActive=True)["federatedRecall"]
    assert late > early + 0.2, f"trajectory did not rise: {early} -> {late}"
    mid = benchmark_current(b, rounds // 2, campaignActive=True)["federatedRecall"]
    assert early <= mid <= late + 1e-9

    # Final round equals the headline federated recall, lift matches.
    cf = benchmark_current(b, rounds, campaignActive=True)
    assert abs(cf["federatedRecall"] - b["metrics"]["federatedRecall"]) < 1e-9
    assert cf["progress"] == 1.0
    assert abs(cf["recallLift"]
               - (b["metrics"]["federatedRecall"] - b["metrics"]["siloedRecall"])) < 1e-9


def test_progress_clamped(benchmark):
    # Round beyond rounds clamps progress to 1.0 and recall to the last sample.
    rounds = benchmark["training"]["rounds"]
    c = benchmark_current(benchmark, rounds + 50, campaignActive=True)
    assert c["progress"] == 1.0
    assert abs(c["federatedRecall"] - benchmark["roundTrajectory"][-1]) < 1e-9


def test_json_serializable(benchmark):
    b = dict(benchmark)
    b["current"] = benchmark_current(benchmark, 5, campaignActive=True)
    s = json.dumps(b)
    assert json.loads(s)["metrics"]["federatedRecall"] == b["metrics"]["federatedRecall"]
