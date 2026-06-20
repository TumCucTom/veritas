# Veritas — Model Architecture

**Status:** ensemble built, tested (`core/`, 122 tests), and proven by a federated experiment. The **live node↔plane loop still trains the logistic model**; wiring the ensemble in as the live model is the scoped next step (§4).

The fraud signal is heterogeneous — linear patterns, feature interactions, temporal layering, categorical mule corridors, and unstructured scam text. No single model catches all of it. Veritas uses a **stacked ensemble of base models, each federated**, combined by a federated meta-learner.

---

## 1. Base models

Each neural model is a **flat fixed-dimension weight vector**, so it federates through the existing substrate (FedAvg → Multi-Krum → DP → the weight-delta contract) **unchanged**. GBDT is the exception — trees have no averageable weights, so it uses its own federated protocol.

| Model | File | Catches | Federation | Measured |
|---|---|---|---|---|
| Logistic | `model.py` | linear-separable | weight-avg (FedAvg/Krum/DP) | baseline |
| **MLP** | `mlp.py` | non-linear / interactions | weight-avg | logistic 0.51 → 0.91 on XOR |
| **Sequence GRU** | `sequence.py` | temporal layering / velocity | weight-avg (BPTT weights) | ordered 1.00 vs shuffled 0.49 |
| **Federated GBDT** | `fedgbdt.py` | axis-aligned tree interactions | **histogram protocol** (below) | fed == centralised (0.944) |
| **Embeddings** | `embeddings.py` | categorical mule corridors | weight-avg (incl. embedding tables) | beats dense-only |
| **Text features** | `text_features.py` | unstructured scam signals | feature extractor (see §3) | separates scam/benign |

**Adversarial training** (`adversarial.py`) is a *training mode*, not a model: FGSM/PGD evasion + adversarially-robust local training (evasion 0.50 → 0.12, 4.2×). Each bank can train any base model robustly, then aggregate normally — it composes with `train_local`. This defends the *inference-time* adversary (feature-gaming), complementing Multi-Krum, which defends the *training-time* adversary (poisoning).

### Federated GBDT (SecureBoost-style) — the non-weight-averaging path
Trees can't be weight-averaged. Instead, each boosting round: every bank bins its features and computes **per-feature gradient/hessian histograms** on its local rows; the coordinator **securely sums** those histograms (Bonawitz pairwise masking — it sees only the global histogram, never a bank's), finds the globally-best split, and grows the tree. Only securely-summed histograms cross the boundary; raw rows never do. Federation here is mathematically **lossless** (fed recall == centralised).

---

## 2. The stacked ensemble (`ensemble.py`)

```
   bank-local views ──► [ logistic ][ MLP ][ GBDT ][ GRU ][ embeddings ]   (each federated)
                              │       │       │      │         │
                              └──────── base fraud probabilities ──────┐
                                                                       ▼
                                       federated logistic META-LEARNER (model.py, dim N+1)
                                                                       │
                                                                       ▼
                                                           final fraud probability
```

- Each base is an **adapter** (`.fed_train(bank_views)` / `.predict(view)`); NN bases share one federation loop, GBDT uses its histogram protocol.
- The **meta-learner** is `model.py`'s logistic over the N base probabilities (weight vector dim `N+1`, identical at every bank → federates via FedAvg/Multi-Krum).
- **Leakage control:** per-bank **train/meta split** — bases train on `base` rows; meta-features are base predictions on each bank's held-out `meta` rows; the meta-learner trains on those. Raw rows never leave a bank — only base weight vectors / GBDT histograms and the meta weight vector.

### Proven result (`experiments/federated_ensemble.py`)
Heterogeneous fraud, 4 typologies, banks blind to different ones:

| model | sub-pattern | recall | AUC |
|---|---|---|---|
| logistic | linear | 0.524 | 0.631 |
| mlp | XOR/interaction | 0.614 | 0.741 |
| gbdt | threshold interact. | 0.369 | 0.754 |
| sequence | temporal burst | 0.434 | 0.624 |
| embeddings | categorical corridor | 0.449 | 0.575 |
| **ENSEMBLE** | **all (stacked)** | **0.761** | **0.893** |

**+0.147 recall over the best single model, +0.237 over logistic.** Each single model covers ~one typology; the meta-learner unions their coverage.

---

## 3. LLM / unstructured signals (honest scope)

`text_features.py` ships a **real, deterministic local featuriser** (n-gram hashing + scam-signal flags) behind a `TextFeatureExtractor` protocol. **Production swaps in an LLM-embedding backend** (FLock sovereign inference / MiniMax) returning the same shape. The LLM's role is **feature extraction from unstructured signals** (scam message text, payment references, KYC notes) and explanation — **not** a tabular scorer. Federating a real LLM encoder = **federated LoRA** (freeze the base, train low-rank adapters locally, FedAvg/Multi-Krum the small adapter deltas through the same flat-weight primitives) — **documented, not built here** (no torch/GPU/LLM in this build).

---

## 4. Making the ensemble the LIVE federated model (next step, not yet wired)

Today the live node↔plane loop trains `model.py` (logistic). To make the ensemble live, in priority order (each is shippable on its own):

1. **Logistic → MLP (smallest real win).** Swap the live federated model to the MLP: change the plane's genesis model + `DIM` (11 → MLP weight_dim), the node engine's `train_local`/`predict_proba`, and the `/predict` path. Same tabular data view — a near drop-in. Re-tune the demo DP σ (more params) and re-green the e2e. Strictly better on non-linear patterns.
2. **Add tabular ensemble members (GBDT + embeddings).** Node trains them locally; plane aggregates GBDT via the histogram path and the meta-learner over the tabular bases. Needs the connector to surface categorical fields for embeddings.
3. **Add temporal + text views.** Node's connector must produce per-account **sequences** (for the GRU) and **text** (for the extractor / LLM). This is the main data-plane lift — the feature-map/connector grows from a flat row to multi-view per account.
4. **Adversarial-robust local training** as a node training option; **federated LoRA** for the LLM member as the research track.

**Why it's a deliberate step, not a quick edit:** the live model's weight-vector dimension ripples through the plane genesis model, `dpParams`, secure-agg/VSA dims, `/predict`, the demo σ tuning, and the e2e assertions. It also sharpens the **privacy/utility tradeoff** (more parameters → more DP noise) — which pairs naturally with moving off synthetic data onto a **real fraud dataset** (IEEE-CIS / PaySim). Do it as a scoped change with the e2e as the gate, not speculatively.
