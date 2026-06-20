"""Evasion-robust adversarial training for the Veritas FL core (numpy only).

THREAT MODEL
------------
Multi-Krum defends the *training* phase: it filters poisoned gradient
updates a malicious bank tries to inject. This module defends the
*inference* phase: a fraudster who already knows (or can probe) the
deployed model nudges the raw transaction features by a tiny amount so a
genuinely fraudulent transaction scores *below* the alert threshold and
slips through. That is an **evasion** attack, and it is orthogonal to
poisoning -- a model can be perfectly robust to poisoning and still be
trivially evadable.

The defence is **adversarial training**: at each local training round we
manufacture worst-case perturbed copies of the bank's data (FGSM / PGD)
and train on a mix of clean + adversarial examples. The model is forced
to keep classifying transactions as fraud even inside an epsilon-ball
around each point, which is exactly the budget the evader is assumed to
have.

GRADIENT STRUCTURE (logistic)
-----------------------------
The core logistic model in ``model.py`` is
``predict_proba(w, X) = sigmoid(X_aug @ w)`` with ``X_aug = [X | 1]``.
For binary cross-entropy ``L = -[y log p + (1-y) log(1-p)]`` the gradient
of the loss wrt a *single* input row x is

    dL/dx = (p - y) * w[:-1]

(the trailing weight ``w[-1]`` is the bias and multiplies the constant 1
column, so it does NOT feed back into the input gradient). FGSM steps the
input along ``sign(dL/dx)`` -- the direction that *increases* the loss,
i.e. pushes a fraud point (y=1, want p high) toward p low so it evades.

This module is written generically over a ``predict_proba(weights, X)``
callable plus a matching ``input_grad`` function. The logistic
input-gradient is provided as ``logistic_input_grad`` and wired up as the
default. To harden the MLP or GNN instead, pass an ``input_grad`` that
runs that model's existing backprop one step *past* the first layer and
returns ``dL/dX`` (shape == X.shape); everything else -- ``fgsm_perturb``,
``pgd_perturb``, ``adversarial_train_local``, ``evasion_success_rate`` --
works unchanged because they only touch the input gradient, never the
model internals.
"""
import numpy as np

from .model import predict_proba as _logistic_predict_proba


def logistic_input_grad(weights, X, y, predict_proba_fn=None):
    """dL/dX for binary cross-entropy of the logistic model.

    Returns an array shaped like ``X``. ``predict_proba_fn`` is accepted
    for signature symmetry with the generic perturb functions but the
    closed form only needs the weights, so it is ignored here.
    """
    w = np.asarray(weights, dtype=np.float64)
    p = _logistic_predict_proba(w, X)            # sigmoid(X_aug @ w)
    # (p - y) is the per-row loss derivative wrt the logit; w[:-1] maps
    # logit-space back to input/feature space (w[-1] is the bias).
    return np.outer(p - np.asarray(y, dtype=np.float64), w[:-1])


def fgsm_perturb(weights, X, y, predict_proba_fn=_logistic_predict_proba,
                 epsilon=0.3, input_grad=logistic_input_grad, clip=None):
    """Fast Gradient Sign Method: one step of size ``epsilon`` along the
    sign of the input loss-gradient.

        X_adv = X + epsilon * sign(dL/dX)

    Moving *up* the loss surface makes correctly-classified points harder
    to classify -- for a fraud row (y=1) this drives the fraud
    probability down, i.e. toward evasion. ``clip`` optionally bounds the
    resulting features to ``(lo, hi)``.
    """
    X = np.asarray(X, dtype=np.float64)
    g = input_grad(weights, X, y, predict_proba_fn)
    X_adv = X + epsilon * np.sign(g)
    if clip is not None:
        X_adv = np.clip(X_adv, clip[0], clip[1])
    return X_adv


def pgd_perturb(weights, X, y, predict_proba_fn=_logistic_predict_proba,
                epsilon=0.3, alpha=None, steps=10,
                input_grad=logistic_input_grad, clip=None, rng=None):
    """Projected Gradient Descent: iterated FGSM with projection back into
    the L-infinity ball of radius ``epsilon`` around the original X.

    Strictly stronger than (or equal to) a single FGSM step under the same
    epsilon budget, because it can follow the curvature of the loss
    surface in several small ``alpha`` steps instead of one big jump while
    never leaving the allowed perturbation box. An optional random start
    inside the ball (when ``rng`` is given) further strengthens it.
    """
    X = np.asarray(X, dtype=np.float64)
    if alpha is None:
        # a few small steps that comfortably reach the ball boundary
        alpha = 2.5 * epsilon / max(1, steps)
    X_adv = X.copy()
    if rng is not None:
        X_adv = X_adv + rng.uniform(-epsilon, epsilon, size=X.shape)
    for _ in range(steps):
        g = input_grad(weights, X_adv, y, predict_proba_fn)
        X_adv = X_adv + alpha * np.sign(g)
        # project back into the L-inf epsilon-ball around the original X
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        if clip is not None:
            X_adv = np.clip(X_adv, clip[0], clip[1])
    return X_adv


def robust_logistic_train(weights, X, y, epsilon=0.5, epochs=2500, lr=0.5,
                          l2=1e-4):
    """Closed-form adversarial training for the logistic model.

    For an L-infinity bounded evader the inner maximisation
    ``max_{||delta||_inf <= eps} BCE(sigmoid((X+delta)@w_feat + b), y)`` has
    an exact solution: the worst delta shifts the logit by exactly
    ``-eps * ||w_feat||_1`` for a positive (fraud) row and ``+eps *
    ||w_feat||_1`` for a negative row, i.e. always *against* the true
    class. So the robust training logit is

        z_adv = X @ w_feat + b  -  eps * ||w_feat||_1 * (2y - 1)

    and we minimise BCE on ``z_adv`` directly. This is Goodfellow & Shlens'
    result that FGSM adversarial training of a linear classifier is
    equivalent to an ``eps * ||w||_1`` penalty -- it provably shrinks the
    weight mass the evader gets to exploit (especially weight wasted on
    non-predictive features), which is the only thing a *linear* model can
    do to resist L-inf evasion. Returns weights in the same ``[w_feat | b]``
    layout as ``model.train_local`` so the result drops straight into
    FedAvg / Multi-Krum aggregation.

    For the MLP / GNN there is no closed form; use ``adversarial_train_local``
    with a PGD ``perturb_fn`` instead (documented there).
    """
    from .model import _aug, init_weights
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xa = _aug(X)
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.shape[0] != Xa.shape[1]:
        w = init_weights(X.shape[1])
    n = len(y)
    npos = int((y == 1).sum())
    pw = np.where(y == 1, (y == 0).sum() / max(1, npos), 1.0)  # class balance
    sgn = (2 * y - 1)                                          # +1 fraud, -1 legit
    for _ in range(epochs):
        wf = w[:-1]
        z = Xa @ w - epsilon * np.abs(wf).sum() * sgn          # worst-case logit
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) * pw
        # d z / d w_feat = X - eps * (2y-1) * sign(w_feat)
        dz_dwf = Xa[:, :-1] - epsilon * sgn[:, None] * np.sign(wf)[None, :]
        gf = (dz_dwf * err[:, None]).mean(0) + l2 * wf
        gb = err.mean() + l2 * w[-1]
        w = w - lr * np.concatenate([gf, [gb]])
    return w


def adversarial_train_local(weights, X, y, train_fn,
                            perturb_fn=fgsm_perturb, epsilon=0.3, mix=0.5,
                            rounds=3, predict_proba_fn=_logistic_predict_proba,
                            input_grad=logistic_input_grad, clip=None,
                            robust_logistic=False, robust_kwargs=None,
                            **train_kwargs):
    """Harden ``weights`` against evasion by training on clean + adversarial
    data.

    Two engines:

    * ``robust_logistic=True`` -- use the exact closed-form robust loss
      (``robust_logistic_train``). This is the recommended path for the
      logistic model: it provably minimises the *worst-case* L-inf loss and
      reliably lowers the evasion rate. ``robust_kwargs`` (epochs/lr/l2) is
      forwarded.

    * ``robust_logistic=False`` (default, generic) -- the model-agnostic
      PGD/FGSM-augmented engine. Each round:
        1. generate adversarial copies of (a ``mix`` fraction of) the data
           against the *current* weights -- a moving target;
        2. append them (with their TRUE labels -- the adversary changed the
           features, not the ground truth) to the clean set;
        3. take an optimisation pass with the model's own ``train_fn``.
      This is the path to use for the MLP / GNN: pass that model's
      ``train_fn``, ``predict_proba_fn`` and an ``input_grad`` that runs its
      backprop to ``dL/dX``; everything else is identical.

    Either way it is a drop-in replacement for the bank-local ``train_local``,
    so it composes directly with federated learning: each bank trains an
    evasion-robust *local* update and the server aggregates those updates
    (FedAvg / Multi-Krum) exactly as before. Robustness is baked into every
    contributed update, so the federated global model inherits it.
    """
    if robust_logistic:
        return robust_logistic_train(weights, X, y, epsilon=epsilon,
                                     **(robust_kwargs or {}))

    w = np.array(weights, dtype=np.float64).copy()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    k = int(round(mix * n))

    def _perturb(idx):
        return perturb_fn(w, X[idx], y[idx], predict_proba_fn=predict_proba_fn,
                          epsilon=epsilon, input_grad=input_grad, clip=clip)

    rng = np.random.default_rng(0)
    for _ in range(max(1, rounds)):
        if k <= 0:
            X_tr, y_tr = X, y
        else:
            idx = rng.permutation(n)[:k]
            X_adv = _perturb(idx)
            X_tr = np.vstack([X, X_adv])
            y_tr = np.concatenate([y, y[idx]])
        w = train_fn(w, X_tr, y_tr, **train_kwargs)
    return w


def evasion_success_rate(weights, X, y, predict_proba_fn=_logistic_predict_proba,
                         epsilon=0.3, perturb_fn=pgd_perturb,
                         input_grad=logistic_input_grad, thr=0.5, clip=None,
                         **perturb_kwargs):
    """Fraction of TRUE-FRAUD examples an adversary can flip to "legitimate"
    under an epsilon perturbation.

    Considers only fraud rows the model *already* catches on clean data
    (proba > thr); an evasion is a row whose perturbed proba drops to
    <= thr. This is the robustness metric: adversarial training should
    drive it down. Returns 0.0 when there are no caught-fraud rows to
    attack.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    fraud = y == 1
    if not fraud.any():
        return 0.0
    Xf, yf = X[fraud], y[fraud]
    caught = predict_proba_fn(weights, Xf) > thr
    if not caught.any():
        return 0.0
    Xc, yc = Xf[caught], yf[caught]
    X_adv = perturb_fn(weights, Xc, yc, predict_proba_fn=predict_proba_fn,
                       epsilon=epsilon, input_grad=input_grad, clip=clip,
                       **perturb_kwargs)
    evaded = predict_proba_fn(weights, X_adv) <= thr
    return float(evaded.mean())
