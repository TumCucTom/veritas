import numpy as np

from veritas_core.embeddings import (
    init_weights, predict_proba, train_local, recall, weight_dim,
    make_categorical_data,
)
from veritas_core import mlp
from veritas_core.aggregation import fedavg, multi_krum


def test_train_improves_recall():
    X_cat, X_dense, y, card = make_categorical_data(4000, seed=1)
    w0 = init_weights(card, seed=0)
    r0 = recall(w0, X_cat, X_dense, y, cardinalities=card)
    w1 = train_local(w0, X_cat, X_dense, y, epochs=60, lr=0.2,
                     cardinalities=card)
    r1 = recall(w1, X_cat, X_dense, y, cardinalities=card)
    assert r1 > r0 + 0.2


def test_proba_range():
    X_cat, X_dense, y, card = make_categorical_data(300, seed=2)
    p = predict_proba(init_weights(card), X_cat, X_dense, card)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p.shape == (len(y),)


def test_embeddings_beat_dense_only_baseline():
    """On the categorical-interaction (mule-corridor) data, the embedding model
    must beat a dense-only MLP baseline that cannot see the categorical fields."""
    X_cat, X_dense, y, card = make_categorical_data(4000, seed=7)
    Xc_te, Xd_te, y_te, _ = make_categorical_data(2000, seed=8)

    # embedding model
    w = init_weights(card, seed=0)
    w = train_local(w, X_cat, X_dense, y, epochs=80, lr=0.2, cardinalities=card)
    emb_rec = recall(w, Xc_te, Xd_te, y_te, cardinalities=card)

    # dense-only baseline (same MLP family, no categorical inputs)
    wb = mlp.init_weights(in_dim=X_dense.shape[1], seed=0)
    wb = mlp.train_local(wb, X_dense, y, epochs=80, lr=0.2,
                         in_dim=X_dense.shape[1])
    base_rec = mlp.recall(wb, Xd_te, y_te, in_dim=X_dense.shape[1])

    assert emb_rec > base_rec + 0.1


def test_federation_dim_consistent_and_aggregators_accept():
    card = [12, 8, 4]
    # several banks, same schema -> identical flat dim
    updates = []
    for bank in range(4):
        Xc, Xd, y, c = make_categorical_data(1500, seed=10 + bank)
        w = init_weights(card, seed=bank)
        w = train_local(w, Xc, Xd, y, epochs=10, lr=0.2, cardinalities=card)
        updates.append(w)

    expected = weight_dim(card)
    assert all(u.shape == (expected,) for u in updates)

    avg = fedavg(updates)
    assert avg.shape == (expected,)

    krum, sel = multi_krum(updates, n_byzantine=1, m=2)
    assert krum.shape == (expected,)
    assert len(sel) == 2

    # aggregated weights are usable for inference (valid flat vector)
    Xc, Xd, y, _ = make_categorical_data(200, seed=99)
    p = predict_proba(avg, Xc, Xd, card)
    assert p.shape == (len(y),) and p.min() >= 0.0 and p.max() <= 1.0
