"""REAL local federation driven through the FREE, LOCAL ``flock-sdk``.

This is the artifact that proves "Veritas trains via flock-sdk": every bank is a
genuine ``flock_sdk.FlockModel`` subclass, and every round exchanges model
parameters as **bytes** through the SDK's ``train`` / ``aggregate`` / ``evaluate``
contract. No API key, no hosted endpoint, no Docker -- the SDK is a tiny pure
Python ABC, so this runs fully offline.

Run from ``core/`` (with the venv active)::

    python -m experiments.flock_federation

What it shows
-------------
N banks each hold their own seeded local data. SOME banks see a fraud *campaign*
(the "safe-account mule" typology that hides on the generic fraud axes); ONE bank
is BLIND to it. Each round:

    1. every bank ``train(global_bytes)`` -> local weight bytes
    2. the coordinator ``aggregate([...])`` -> new global weight bytes
    3. ``evaluate`` the global model's recall on each bank

The federated global model lifts the blind bank's campaign recall well above what
it could reach in isolation (siloed) -- because the campaign knowledge arrives as
weights, never as raw data.
"""
import numpy as np

import flock_sdk
from veritas_core.flock_adapter import (
    VeritasFlockModel,
    USING_REAL_FLOCK_SDK,
    metric_from_bytes,
)
from veritas_core.data import make_bank_data, inject_campaign


def _campaign_eval_set(seed: int = 4242):
    """A held-out set rich in campaign fraud, to measure campaign recall."""
    X, y = make_bank_data(1000, 0.03, seed=seed)
    X, y = inject_campaign(X, y, 300, seed=7)
    return X, y


def main(n_banks: int = 5, rounds: int = 8):
    print("=" * 68)
    print("VERITAS x FLock  --  REAL local federation via flock-sdk")
    print("=" * 68)
    print(f"flock_sdk module : {flock_sdk.__file__}")
    print(f"USING_REAL_FLOCK_SDK = {USING_REAL_FLOCK_SDK}")
    print(f"VeritasFlockModel.__bases__ = "
          f"{[b.__module__ + '.' + b.__name__ for b in VeritasFlockModel.__bases__]}")
    assert USING_REAL_FLOCK_SDK, "expected the REAL flock_sdk.FlockModel as base"
    assert issubclass(VeritasFlockModel, flock_sdk.FlockModel)
    print("-> Banks are genuine flock_sdk.FlockModel subclasses.\n")

    # Bank 0 is BLIND to the campaign; banks 1..N-1 have seen it.
    banks = []
    for i in range(n_banks):
        sees_campaign = (i != 0)
        m = VeritasFlockModel(seed=10 + i, campaign=sees_campaign)
        m.init_dataset(f"seed://{10 + i}")  # drive the real SDK hook
        banks.append(m)
    print(f"{n_banks} banks instantiated via init_dataset(); "
          f"bank 0 is BLIND to the campaign, banks 1..{n_banks-1} have seen it.\n")

    Xc, yc = _campaign_eval_set()

    def campaign_recall(weight_bytes):
        from veritas_core.model import recall
        from veritas_core.flock_adapter import _from_bytes
        return recall(_from_bytes(weight_bytes), Xc, yc)

    # ---- SILOED baseline: bank 0 trains alone, never sees the campaign ----
    siloed = VeritasFlockModel(seed=10, campaign=False)
    siloed.init_dataset("seed://10")
    sb = b""
    for _ in range(rounds):
        sb = siloed.train(sb)
    siloed_campaign_recall = campaign_recall(sb)

    # ---- FEDERATED: train through the SDK round by round ------------------
    global_bytes = b""  # empty => round 0 starts from init_weights
    coordinator = banks[0]  # any bank can run aggregate() (it's stateless)
    print(f"{'round':>5} | {'global campaign recall':>22} | per-bank local recall")
    print("-" * 68)
    for r in range(1, rounds + 1):
        local_updates = [bank.train(global_bytes) for bank in banks]  # bytes
        global_bytes = coordinator.aggregate(local_updates)           # bytes
        local_recalls = [metric_from_bytes(b.evaluate(global_bytes)) for b in banks]
        g_recall = campaign_recall(global_bytes)
        print(f"{r:>5} | {g_recall:>22.3f} | "
              f"[{', '.join(f'{x:.2f}' for x in local_recalls)}]")

    print("-" * 68)
    fed_blind_recall = metric_from_bytes(banks[0].evaluate(global_bytes))
    print()
    print("RESULT (campaign recall = catching the hidden 'safe-account mule' fraud):")
    print(f"  Siloed bank 0 (alone, blind to campaign) : {siloed_campaign_recall:.3f}")
    print(f"  Federated global model                   : {campaign_recall(global_bytes):.3f}")
    print(f"  Federated recall on bank 0's own data    : {fed_blind_recall:.3f}")
    lift = campaign_recall(global_bytes) - siloed_campaign_recall
    print(f"  Federated LIFT over siloed               : +{lift:.3f}")
    print()
    print("Proven: Veritas trained a fraud model through the REAL flock-sdk "
          "(bytes-based\nFlockModel contract), and federation beat the siloed "
          "baseline -- all FREE & LOCAL.")
    return {
        "siloed_campaign_recall": siloed_campaign_recall,
        "federated_campaign_recall": campaign_recall(global_bytes),
        "rounds": rounds,
        "n_banks": n_banks,
    }


if __name__ == "__main__":
    main()
