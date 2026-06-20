"""STACKED FEDERATED ENSEMBLE — the capstone of the Veritas model stack.

Why stack at all
----------------
Each Veritas base model is tuned to a DIFFERENT fraud sub-pattern:

    * ``model.py``   (logistic)   -> a linearly-separable slice;
    * ``mlp.py``     (MLP)        -> XOR / interaction slices;
    * ``fedgbdt.py`` (GBDT)       -> axis-aligned interaction / threshold slices;
    * ``sequence.py``(GRU)        -> temporal-burst (layering) slices;
    * ``embeddings.py``(emb-MLP)  -> categorical-corridor slices.

Real fraud is a MIX of these typologies, and no single model covers all of them:
the logistic model is blind to the temporal burst, the GRU is blind to the
categorical corridor, and so on. A STACKED ENSEMBLE lets a meta-learner LEARN,
per account, which base model's vote to trust — so the ensemble's recall exceeds
the best single base model because it unions their coverage.

Architecture
------------
::

    bank views ── base_1 (adapter) ──┐  p_1
               ── base_2 (adapter) ──┤  p_2
               ── ...               ──┼──► [p_1 .. p_N]  ──► META logistic ──► p
               ── base_N (adapter) ──┘  p_N

    * Each base learner is a thin ADAPTER object exposing
        - ``.fed_train(bank_views)``  : federated-train this base model across
          all banks (NN bases via fedavg/multi_krum on the flat weight vector;
          GBDT via its histogram protocol), storing the global model on self;
        - ``.predict(view)``          : per-account fraud probability in [0,1].
      Adapters declare which keys of a bank's "view" dict they consume, so the
      ensemble stays generic over ANY subset of base models.
    * The META-LEARNER is a small federated LOGISTIC REGRESSION (REUSING
      ``veritas_core.model``) whose input features are the base learners'
      predicted probabilities (one column per base model). It, too, federates
      via fedavg / multi_krum on its flat weight vector.

Federated stacking WITHOUT leakage (the train / meta split)
-----------------------------------------------------------
Stacking leaks if the meta-learner is trained on base predictions made on the
SAME rows the bases were fit on: the bases overfit those rows, so their training
predictions are unrealistically good and the meta-learner learns to trust an
optimism that won't hold at inference. We avoid this with a clean, per-bank
TRAIN / META split (documented choice; an out-of-fold variant is equivalent but
needs K× the base training):

    1. Each bank splits its LOCAL rows into a ``base`` part and a disjoint
       ``meta`` part (same split everywhere, by index).
    2. Every base model is federated-trained on the banks' ``base`` parts ONLY.
    3. Each bank produces base predictions on its OWN ``meta`` part — rows NO
       base model has ever seen. These predictions (N columns) are the
       meta-learner's features.
    4. The meta logistic regression is federated-trained on those held-out
       meta-features across banks (each bank trains locally on its meta rows,
       weights are fedavg/multi_krum-aggregated). Because the meta-features come
       from unseen rows, the meta-learner sees realistic base behaviour and does
       not learn to trust train-set optimism.

Raw rows never leave a bank: only base-model weight vectors / GBDT histograms and
the meta-learner's weight vector cross the boundary — exactly the messages the
existing FL primitives already exchange.
"""
from __future__ import annotations

import numpy as np

from . import model as logistic
from . import mlp
from . import sequence as gru
from . import embeddings as emb
from . import fedgbdt
from .aggregation import fedavg
from . import robust


# ===========================================================================
# View helpers
# ===========================================================================
# A "bank view" is a dict describing ONE bank's local data. Adapters read only
# the keys they need, so different banks/datasets can carry extra views freely.
#
# Conventional keys (the headline experiment populates these):
#   "y"        : (n,) int labels                              [required]
#   "tab"      : (n, d) dense tabular features                [logistic/mlp/gbdt]
#   "seq"      : (n, T, feat) per-account transaction sequence [gru]
#   "cat"      : (n, F) int categorical indices               [embeddings]
#   "dense"    : (n, dense_dim) dense features for emb head    [embeddings]
#   "cardinalities" : list[int] categorical schema            [embeddings]
# ===========================================================================
def _slice_view(view, idx):
    """Return a shallow copy of ``view`` with array-like fields sliced by idx.

    Scalars / schema entries (e.g. ``cardinalities``) pass through untouched.
    """
    out = {}
    n = len(view["y"])
    for k, v in view.items():
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def split_view(view, meta_frac=0.4, seed=0):
    """Split one bank's view into disjoint ``(base_view, meta_view)`` by index.

    The split is the anti-leakage device: bases train on ``base_view``, the
    meta-learner trains on base predictions over ``meta_view`` (unseen rows).
    """
    n = len(view["y"])
    if n <= 1:
        # Anti-leakage guard: a clean disjoint base/meta split needs at least
        # one row on EACH side. At n<=1 any split degenerates to full leakage
        # (base and meta would share the only row, or one side is empty), so we
        # refuse rather than silently train+evaluate on the same row.
        raise ValueError(
            f"split_view needs >=2 rows for a disjoint base/meta split, got n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # Keep at least one row on BOTH sides so neither bases nor the meta-learner
    # train on an empty set and the split stays genuinely disjoint.
    n_meta = min(n - 1, max(1, int(round(n * meta_frac))))
    meta_idx = np.sort(perm[:n_meta])
    base_idx = np.sort(perm[n_meta:])
    return _slice_view(view, base_idx), _slice_view(view, meta_idx)


# ===========================================================================
# Base-learner ADAPTERS
# ===========================================================================
# Each adapter wraps one Veritas base model behind a uniform interface:
#   .name              : str
#   .fed_train(views)  : federated-train across the list of bank views
#   .predict(view)     : (n,) fraud probability in [0,1]
# NN adapters expose .updates_from(views) producing per-bank flat weight vectors
# so the ensemble can route them through fedavg OR multi_krum centrally.
# ===========================================================================
class _NNAdapter:
    """Shared federation loop for the flat-weight-vector (NN) base models.

    Subclasses provide ``_init``, ``_train_local``, ``_predict`` and ``_xy``.
    Federation = repeated rounds of {local train, robust-aggregate weights}.
    """

    name = "nn"

    def __init__(self, rounds=8, local_epochs=12, lr=0.2, robust=False,
                 n_byzantine=0, seed=0):
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.robust = robust
        self.n_byzantine = n_byzantine
        self.seed = seed
        self.w = None  # global flat weight vector after fed_train

    # --- hooks subclasses implement -------------------------------------
    def _init(self, views):
        raise NotImplementedError

    def _train_local(self, w, view):
        raise NotImplementedError

    def _predict(self, w, view):
        raise NotImplementedError

    # --- federation driver ----------------------------------------------
    def _aggregate(self, updates, num_examples=None):
        if self.robust and len(updates) >= 3:
            # Krum SELECTS a Byzantine-free majority (delegates to the single
            # canonical implementation in robust.py, which clamps negative
            # squared distances) rather than averaging everyone after dropping
            # one outlier.
            agg, _, _ = robust.multi_krum_select(
                updates, n_byzantine=self.n_byzantine,
                m=max(1, len(updates) - self.n_byzantine))
            return agg
        # FedAvg weighted by each bank's sample count (numExamples): banks with
        # more local rows pull the global model proportionally harder, so an
        # imbalanced federation is averaged fairly rather than one-vote-per-bank.
        return fedavg(updates, weights=num_examples)

    def fed_train(self, views):
        w = self._init(views)
        num_examples = [len(v["y"]) for v in views]
        for _ in range(self.rounds):
            updates = [self._train_local(w, v) for v in views]
            w = self._aggregate(updates, num_examples=num_examples)
        self.w = w
        return self

    def predict(self, view):
        return np.clip(self._predict(self.w, view), 0.0, 1.0)


class LogisticAdapter(_NNAdapter):
    """Linear (logistic regression) base learner over the dense tabular view."""

    name = "logistic"

    def __init__(self, view_key="tab", **kw):
        super().__init__(**kw)
        self.view_key = view_key

    def _dim(self, views):
        return views[0][self.view_key].shape[1]

    def _init(self, views):
        return logistic.init_weights(self._dim(views))

    def _train_local(self, w, view):
        return logistic.train_local(w, view[self.view_key], view["y"],
                                    epochs=self.local_epochs, lr=self.lr)

    def _predict(self, w, view):
        return logistic.predict_proba(w, view[self.view_key])


class MLPAdapter(_NNAdapter):
    """Non-linear MLP base learner (XOR / interaction slices)."""

    name = "mlp"

    def __init__(self, view_key="tab", hidden=mlp.HIDDEN, **kw):
        super().__init__(**kw)
        self.view_key = view_key
        self.hidden = hidden

    def _dim(self, views):
        return views[0][self.view_key].shape[1]

    def _init(self, views):
        return mlp.init_weights(in_dim=self._dim(views), hidden=self.hidden,
                                seed=self.seed)

    def _train_local(self, w, view):
        return mlp.train_local(w, view[self.view_key], view["y"],
                               epochs=self.local_epochs, lr=self.lr,
                               in_dim=view[self.view_key].shape[1],
                               hidden=self.hidden)

    def _predict(self, w, view):
        return mlp.predict_proba(w, view[self.view_key],
                                 in_dim=view[self.view_key].shape[1],
                                 hidden=self.hidden)


class SequenceAdapter(_NNAdapter):
    """GRU base learner over per-account transaction sequences (temporal burst)."""

    name = "sequence"

    def __init__(self, view_key="seq", hidden=16, rounds=6, local_epochs=30,
                 lr=0.05, **kw):
        super().__init__(rounds=rounds, local_epochs=local_epochs, lr=lr, **kw)
        self.view_key = view_key
        self.hidden = hidden

    def _feat(self, views):
        return views[0][self.view_key].shape[2]

    def _init(self, views):
        return gru.init_weights(feat=self._feat(views), hidden=self.hidden,
                                seed=self.seed)

    def _train_local(self, w, view):
        return gru.train_local(w, view[self.view_key], view["y"],
                               epochs=self.local_epochs, lr=self.lr,
                               feat=view[self.view_key].shape[2])

    def _predict(self, w, view):
        return gru.predict_proba(w, view[self.view_key],
                                 feat=view[self.view_key].shape[2])


class EmbeddingAdapter(_NNAdapter):
    """Categorical-embedding base learner (mule-corridor interaction)."""

    name = "embeddings"

    def __init__(self, cat_key="cat", dense_key="dense", emb_dim=4, hidden=16,
                 rounds=8, local_epochs=20, lr=0.2, **kw):
        super().__init__(rounds=rounds, local_epochs=local_epochs, lr=lr, **kw)
        self.cat_key = cat_key
        self.dense_key = dense_key
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.card = None
        self.dense_dim = None

    def _init(self, views):
        self.card = list(views[0]["cardinalities"])
        self.dense_dim = views[0][self.dense_key].shape[1]
        return emb.init_weights(self.card, emb_dim=self.emb_dim,
                                dense_dim=self.dense_dim, hidden=self.hidden,
                                seed=self.seed)

    def _train_local(self, w, view):
        return emb.train_local(w, view[self.cat_key], view[self.dense_key],
                               view["y"], epochs=self.local_epochs, lr=self.lr,
                               cardinalities=self.card, emb_dim=self.emb_dim,
                               dense_dim=self.dense_dim, hidden=self.hidden)

    def _predict(self, w, view):
        return emb.predict_proba(w, view[self.cat_key], view[self.dense_key],
                                 self.card, emb_dim=self.emb_dim,
                                 dense_dim=self.dense_dim, hidden=self.hidden)


class GBDTAdapter:
    """Federated GBDT base learner (its own histogram protocol, not weight avg).

    Mirrors the NN-adapter interface (``fed_train`` / ``predict``) but federates
    via ``fedgbdt.fed_train_gbdt`` because trees have no averageable weights.
    """

    name = "gbdt"

    def __init__(self, view_key="tab", n_trees=20, max_depth=3, n_bins=16,
                 lr=0.3, secure=True):
        self.view_key = view_key
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lr = lr
        self.secure = secure
        self.model = None

    def fed_train(self, views):
        bank_data = [(v[self.view_key], v["y"]) for v in views]
        self.model = fedgbdt.fed_train_gbdt(
            bank_data, n_trees=self.n_trees, max_depth=self.max_depth,
            n_bins=self.n_bins, lr=self.lr, secure=self.secure)
        return self

    def predict(self, view):
        return np.clip(fedgbdt.predict_proba(self.model, view[self.view_key]),
                       0.0, 1.0)


# ===========================================================================
# The STACKED ENSEMBLE
# ===========================================================================
class StackedEnsemble:
    """N base learners + a federated logistic meta-learner over their probs."""

    def __init__(self, base_learners, meta_epochs=200, meta_lr=0.3,
                 meta_rounds=6, robust=False, n_byzantine=0):
        self.bases = list(base_learners)
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.meta_rounds = meta_rounds
        self.robust = robust
        self.n_byzantine = n_byzantine
        self.meta_w = None  # flat logistic weight vector over base-prob features

    # --- meta-feature construction --------------------------------------
    def _meta_features(self, views):
        """Stack base predictions into an (n, N) feature matrix per view list."""
        return [
            np.column_stack([b.predict(v) for b in self.bases]) for v in views
        ]

    def _aggregate_meta(self, updates, num_examples=None):
        if self.robust and len(updates) >= 3:
            agg, _, _ = robust.multi_krum_select(
                updates, n_byzantine=self.n_byzantine,
                m=max(1, len(updates) - self.n_byzantine))
            return agg
        # sample-count (numExamples) weighted FedAvg over the meta rows.
        return fedavg(updates, weights=num_examples)

    # --- training -------------------------------------------------------
    def fed_train(self, bank_views, meta_frac=0.4, seed=0):
        """Federated stacked training with a per-bank train/meta split.

        bank_views : list of per-bank view dicts (one per bank).
        Returns self. See module docstring for the leakage argument.
        """
        # 1. per-bank disjoint base / meta split (anti-leakage)
        splits = [split_view(v, meta_frac=meta_frac, seed=seed + i)
                  for i, v in enumerate(bank_views)]
        base_views = [s[0] for s in splits]
        meta_views = [s[1] for s in splits]

        # 2. federated-train every base model on the BASE rows only
        for b in self.bases:
            b.fed_train(base_views)

        # 3. meta-features = base preds on each bank's held-out META rows
        meta_X = self._meta_features(meta_views)
        meta_y = [v["y"] for v in meta_views]

        # 4. federated logistic meta-learner over base-prob features (REUSE model.py)
        n_feat = len(self.bases)
        w = logistic.init_weights(n_feat)
        meta_counts = [len(my) for my in meta_y]
        for _ in range(self.meta_rounds):
            updates = [
                logistic.train_local(w, mx, my, epochs=self.meta_epochs,
                                     lr=self.meta_lr)
                for mx, my in zip(meta_X, meta_y)
            ]
            w = self._aggregate_meta(updates, num_examples=meta_counts)
        self.meta_w = w
        return self

    # --- inference ------------------------------------------------------
    def predict_proba(self, views):
        """Final fraud probability per account, concatenated across views."""
        meta_X = self._meta_features(views)
        return np.concatenate(
            [logistic.predict_proba(self.meta_w, mx) for mx in meta_X]
        )

    def base_predict_proba(self, base_idx, views):
        """Convenience: a single base model's probabilities across views."""
        b = self.bases[base_idx]
        return np.concatenate([b.predict(v) for v in views])

    def labels(self, views):
        return np.concatenate([v["y"] for v in views])

    def recall(self, views, thr=0.5):
        y = self.labels(views)
        if y.sum() == 0:
            return 1.0
        pred = self.predict_proba(views) > thr
        return float((pred & (y == 1)).sum() / (y == 1).sum())


# ===========================================================================
# Module-level convenience API (mirrors the base models' free-function style)
# ===========================================================================
def fed_train_ensemble(bank_views, base_learners, meta_frac=0.4, seed=0,
                       meta_epochs=200, meta_lr=0.3, meta_rounds=6,
                       robust=False, n_byzantine=0):
    """Build and federated-train a :class:`StackedEnsemble`. Returns it."""
    ens = StackedEnsemble(base_learners, meta_epochs=meta_epochs,
                          meta_lr=meta_lr, meta_rounds=meta_rounds,
                          robust=robust, n_byzantine=n_byzantine)
    ens.fed_train(bank_views, meta_frac=meta_frac, seed=seed)
    return ens


def predict_proba(ensemble, views):
    return ensemble.predict_proba(views)


def recall(ensemble, views, thr=0.5):
    return ensemble.recall(views, thr=thr)


def auc(scores, y):
    """Mann-Whitney AUC (pure numpy). 0.5 if a class is absent."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# Default roster covering all five sub-patterns. Banks must carry the matching
# views (tab / seq / cat+dense+cardinalities). The experiment uses this.
def default_base_learners(seed=0):
    return [
        LogisticAdapter(seed=seed),                         # linear slice
        MLPAdapter(seed=seed),                              # XOR / interaction
        GBDTAdapter(),                                      # threshold interaction
        SequenceAdapter(seed=seed),                         # temporal burst
        EmbeddingAdapter(seed=seed),                        # categorical corridor
    ]
