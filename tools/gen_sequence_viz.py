"""OFFLINE generator for the Veritas SEQUENCE-model website visualization.

Runs the project's REAL pure-numpy GRU temporal-fraud model
(``veritas_core.sequence`` over ``veritas_core.sequence_data``) and exports a
small, deterministic JSON of example customer transaction TIMELINES with the
GRU's RUNNING fraud score after each step. The website animates these to tell
the temporal story: no single transaction is damning; the GRU reads the mule
*pattern* (a quiet dormant prefix then a high-value high-fan-out layering
burst) as it unfolds over time, and its score climbs.

How the "running score" is real
-------------------------------
The GRU reads out a fraud probability only from its FINAL hidden state. To get
a genuine per-step score we run the REAL model forward on each growing PREFIX of
the sequence — ``predict_proba(weights, X[:, :t+1, :])`` — so step t's score is
exactly what the trained GRU would say if the customer's timeline ended at
transaction t. For a mule the score climbs as the burst arrives; for a legit
account it stays flat/low.

Human-readable fields come straight from the REAL feature columns of the chosen
sequences (amount[0], velocity[6], fanout[7], recency[8]); we map those model
features to plain display units (currency amount, hour-of-day, fan-out count)
deterministically so the timeline is legible without inventing any signal.

Everything is deterministic (fixed seed) and re-runnable.

Run:  python tools/gen_sequence_viz.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Make veritas_core importable when run from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "core"))

from veritas_core import sequence  # noqa: E402
from veritas_core.sequence_data import make_sequence_data  # noqa: E402
from veritas_core.data import FEATURE_DIM  # noqa: E402

SEED = 20240620
OUT_DIR = os.path.join(REPO, "web", "public", "viz")

# Feature-column conventions (shared across veritas datasets).
AMOUNT, VELOCITY, FANOUT, RECENCY = 0, 6, 7, 8

HIDDEN = 16
T = 8


def _r(x, nd=4):
    return round(float(x), nd)


def _train_gru():
    """Train the REAL GRU on real sequence data; return weights + the dataset."""
    X, y = make_sequence_data(n=600, fraud_rate=0.18, T=T, seed=SEED)
    w = sequence.init_weights(feat=FEATURE_DIM, hidden=HIDDEN, seed=SEED)
    # BPTT-train the real model on the real sequences.
    w = sequence.train_local(w, X, y, epochs=120, lr=0.08, feat=FEATURE_DIM)
    return w, X, y


def _running_scores(w, seq):
    """REAL per-step GRU score: score the GRU on each growing prefix.

    `seq` is (T, feat). Returns a length-T array where element t is
    predict_proba on the prefix seq[:t+1] — i.e. the GRU's fraud probability if
    the customer's timeline ended at transaction t.
    """
    scores = np.empty(T)
    for t in range(T):
        prefix = seq[np.newaxis, : t + 1, :]            # (1, t+1, feat)
        scores[t] = float(sequence.predict_proba(w, prefix, feat=FEATURE_DIM)[0])
    return scores


def _human_fields(seq):
    """Map real feature columns to legible display units per timestep.

    Deterministic, no invented signal — purely a readable projection of the
    actual feature values the GRU consumes.
    """
    amt_z = seq[:, AMOUNT]
    vel_z = seq[:, VELOCITY]
    fan_z = seq[:, FANOUT]
    rec_z = seq[:, RECENCY]

    # Amount: base ~ £180, scaled up by the (real) amount feature. Quiet steps
    # land small; burst steps spike — exactly as the feature column does.
    amount = np.clip(180.0 + amt_z * 1400.0, 12.0, None)

    # Hour-of-day: dormant/quiet transactions read as daytime; the rapid burst
    # reads as the small hours (recency drops sharply in the burst). Derived
    # from the real recency column so it tracks the model's signal.
    hour = np.clip(np.round(13.0 - rec_z * 5.0), 0, 23).astype(int)

    # Fan-out: number of distinct counterparties this step touches.
    fanout = np.clip(np.round(1.0 + np.maximum(fan_z, 0.0) * 4.0), 1, None).astype(int)

    # Velocity flag for the note.
    velocity = vel_z
    return amount, hour, fanout, velocity


def _build_sequence(seq_id, label, w, seq):
    amount, hour, fanout, velocity = _human_fields(seq)
    scores = _running_scores(w, seq)

    # Identify the burst onset (first step where amount + fanout jump) purely for
    # human-readable notes; the model already encodes it.
    burst_step = None
    for t in range(1, T):
        if (amount[t] > amount[: max(1, t)].mean() * 2.2) and fanout[t] >= 3:
            burst_step = t
            break

    steps = []
    for t in range(T):
        note = None
        if label == 1:
            if burst_step is not None and t < burst_step:
                note = "Dormant — small, ordinary activity."
            elif burst_step is not None and t == burst_step:
                note = "Burst begins: high value, many new counterparties."
            elif burst_step is not None and t > burst_step:
                note = "Rapid layering continues — pattern now clear."
        else:
            note = None
        step = {
            "t": int(t),
            "amount": _r(amount[t], 2),
            "hour": int(hour[t]),
            "fanout": int(fanout[t]),
            "score": _r(scores[t], 4),
        }
        if note is not None:
            step["note"] = note
        steps.append(step)

    return {"id": seq_id, "label": int(label), "steps": steps}, float(scores[-1])


def _pick_examples(w, X, y):
    """Pick the clearest real fraud + legit sequences by final GRU score.

    Fraud: highest-scoring true mule (score climbs to the top).
    Legit: a true-legit sequence the model keeps low AND flat (so contrast is
    crisp). All picks are REAL rows from the trained dataset.
    """
    p = sequence.predict_proba(w, X, feat=FEATURE_DIM)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    # clearest mule = highest final score among true frauds
    best_fraud = fraud_idx[np.argmax(p[fraud_idx])]
    # crispest legit = lowest final score among true legits
    best_legit = legit_idx[np.argmin(p[legit_idx])]
    # a second legit with mild mid-sequence wobble that still stays under
    # threshold, to show "ordinary noise isn't fraud".
    legit_sorted = legit_idx[np.argsort(p[legit_idx])]
    mid_legit = legit_sorted[len(legit_sorted) // 3]
    return best_fraud, best_legit, mid_legit


def gen_sequence():
    w, X, y = _train_gru()
    best_fraud, best_legit, mid_legit = _pick_examples(w, X, y)

    seqs = []
    finals = {}

    s_fraud, f_fraud = _build_sequence("mule-layering", 1, w, X[best_fraud])
    seqs.append(s_fraud)
    finals["fraud"] = f_fraud

    s_legit, f_legit = _build_sequence("salary-and-bills", 0, w, X[best_legit])
    seqs.append(s_legit)
    finals["legit"] = f_legit

    s_legit2, f_legit2 = _build_sequence("busy-but-honest", 0, w, X[mid_legit])
    seqs.append(s_legit2)
    finals["legit2"] = f_legit2

    out = {
        "sequences": seqs,
        "meta": {
            "note": ("A from-scratch numpy GRU reads each customer's ordered "
                     "transaction timeline. Its fraud score is the real model's "
                     "probability on the growing prefix at each step: no single "
                     "transaction is damning, but the mule's quiet-then-burst "
                     "layering pattern makes the score climb past threshold, "
                     "while honest accounts stay flat and low."),
            "model": "GRU",
        },
    }
    return out, finals


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate(d):
    assert set(d) == {"sequences", "meta"}, set(d)
    assert d["meta"]["model"] == "GRU"
    assert isinstance(d["meta"]["note"], str) and d["meta"]["note"]
    assert len(d["sequences"]) >= 2
    labels = set()
    for s in d["sequences"]:
        assert set(s) >= {"id", "label", "steps"}
        assert s["label"] in (0, 1)
        labels.add(s["label"])
        assert isinstance(s["id"], str) and s["id"]
        prev_t = -1
        for st in s["steps"]:
            assert set(st) >= {"t", "amount", "hour", "score"}
            assert set(st) <= {"t", "amount", "hour", "fanout", "score", "note"}
            assert isinstance(st["t"], int) and st["t"] == prev_t + 1
            prev_t = st["t"]
            assert 0 <= st["hour"] <= 23
            assert 0.0 <= st["score"] <= 1.0
            assert isinstance(st["amount"], (int, float))
            if "fanout" in st:
                assert isinstance(st["fanout"], (int, float))
            if "note" in st:
                assert isinstance(st["note"], str)
    # must contain at least one fraud and one legit
    assert 0 in labels and 1 in labels, "need both a fraud and a legit sequence"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("[sequence] training REAL GRU on real transaction sequences ...")
    out, finals = gen_sequence()

    path = os.path.join(OUT_DIR, "sequence.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

    with open(path) as fh:
        _validate(json.load(fh))

    size_kb = os.path.getsize(path) / 1024.0

    print("\n==================== SUMMARY ====================")
    print(f"sequence.json    {size_kb:7.2f} KB  {path}")
    print(f"  sequences = {len(out['sequences'])}  (steps each = {T})")
    for s in out["sequences"]:
        kind = "FRAUD/mule" if s["label"] == 1 else "LEGIT"
        scores = [st["score"] for st in s["steps"]]
        print(f"  [{kind:10}] id={s['id']:18} "
              f"start={scores[0]:.3f}  final={scores[-1]:.3f}")
    print("  ---")
    print(f"  FRAUD final score  = {finals['fraud']:.3f}  (should be HIGH)")
    print(f"  LEGIT final score  = {finals['legit']:.3f}  (should be LOW)")
    print(f"  LEGIT-2 final score= {finals['legit2']:.3f}  (should be LOW)")
    assert size_kb < 80.0, f"sequence.json too large: {size_kb:.1f} KB"
    assert finals["fraud"] > finals["legit"], "fraud must outscore legit"
    print("=================================================")
    print("sequence.json written, schema-validated, < 80 KB.")


if __name__ == "__main__":
    main()
