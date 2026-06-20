import numpy as np

from veritas_core.text_features import (
    extract_text_features, LocalHashingExtractor, TextFeatureExtractor,
    make_text_data, N_FLAGS,
)


def test_deterministic_and_shape():
    texts = ["move your money to a safe account now", "see you at lunch"]
    a = extract_text_features(texts, dim=32)
    b = extract_text_features(texts, dim=32)
    assert a.shape == (2, 32)
    assert np.array_equal(a, b)  # fully deterministic across calls


def test_values_in_sane_range():
    texts, _ = make_text_data(n_per_class=30, seed=0)
    F = extract_text_features(texts, dim=48)
    assert F.shape == (len(texts), 48)
    assert np.all(np.isfinite(F))
    # flags are binary {0,1}; hashed block is L2-normalised so |v|<=1
    assert np.all(np.abs(F) <= 1.0 + 1e-9)


def test_protocol_conformance():
    ext = LocalHashingExtractor(dim=32)
    assert isinstance(ext, TextFeatureExtractor)
    assert ext.dim == 32
    out = ext.extract(["urgent: verify your pin"])
    assert out.shape == (1, 32)


def test_scam_signal_flags_fire():
    scam = "URGENT: HMRC fraud team - move your money to this safe account now"
    benign = "thanks for dinner, see you next week"
    fs = extract_text_features([scam], dim=32)[0, :N_FLAGS]
    fb = extract_text_features([benign], dim=32)[0, :N_FLAGS]
    assert fs.sum() >= 3  # several scam flags fire
    assert fb.sum() == 0  # benign trips none


def _train_logreg(X, y, epochs=300, lr=0.5):
    """Tiny numpy logistic regression for the separability check."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xa.shape[1])
    n = len(y)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ w, -30, 30)))
        w -= lr * (Xa.T @ (p - y) / n)
    return w


def test_classifier_separates_scam_above_chance():
    texts_tr, y_tr = make_text_data(n_per_class=80, seed=1)
    texts_te, y_te = make_text_data(n_per_class=40, seed=2)
    Xtr = extract_text_features(texts_tr, dim=48)
    Xte = extract_text_features(texts_te, dim=48)

    w = _train_logreg(Xtr, y_tr.astype(np.float64))
    Xte_a = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    pred = (1.0 / (1.0 + np.exp(-np.clip(Xte_a @ w, -30, 30)))) > 0.5
    acc = float((pred == y_te).mean())
    assert acc > 0.8  # well above the 0.5 chance baseline
