"""LLM-style text-feature extractor for UNSTRUCTURED fraud signals, with a
deterministic LOCAL fallback and a clear production-swap interface.

What this is (and is not)
-------------------------
Real fraud carries unstructured text: scam-message bodies, payment references /
narratives, KYC free-text notes, support-chat transcripts. This module turns such
strings into a fixed-width numeric feature vector that sits alongside the tabular
/ embedding / sequence features elsewhere in Veritas.

`LocalHashingExtractor` is the REFERENCE / fallback featuriser: it is pure NUMPY
+ stdlib, fully deterministic, dependency-free and offline. It combines

    * the HASHING TRICK over char- and word-level n-grams (stable token hashing
      into a fixed number of buckets — no learned vocabulary, no fitting), with
    * a handful of engineered scam-signal flags (urgency phrasing, "safe
      account" social-engineering, sort-code / account-number patterns, payment
      pressure, impersonation of authority, links, etc.).

In PRODUCTION the same `TextFeatureExtractor` protocol is satisfied by an
LLM-embedding backend (e.g. FLock sovereign inference / MiniMax) that returns
embeddings of the SAME `[n, dim]` shape. Swapping backend changes only the
feature *quality*, not the downstream contract — every consumer keeps working.

Honest framing on federating an LLM
-----------------------------------
This module does NOT federate or train an LLM. A local hashing featuriser has no
learnable weights to share. Federating an actual LLM text encoder across banks is
a *federated-LoRA* problem: freeze the base model, train low-rank adapters
locally, and FedAvg / Multi-Krum the (small, fixed-dim) LoRA deltas — which slots
into the exact same flat-weight aggregation primitives used by `model.py` /
`mlp.py` / `embeddings.py`. That is DOCUMENTED here as the production path and is
deliberately NOT built in this reference module (no torch, no network, no LLM).
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Production-swap interface
# ---------------------------------------------------------------------------
@runtime_checkable
class TextFeatureExtractor(Protocol):
    """Contract every text featuriser must satisfy.

    A production LLM-embedding backend (FLock sovereign inference / MiniMax)
    implements this same method and returns the SAME `[n, dim]` float array, so
    downstream code is backend-agnostic. `dim` is fixed for a given instance.
    """

    dim: int

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float64 feature matrix for `texts`."""
        ...


# ---------------------------------------------------------------------------
# Engineered scam-signal lexicon / regexes
# ---------------------------------------------------------------------------
_URGENCY = re.compile(
    r"\b(urgent|immediately|right now|asap|act now|expire[sd]?|within \d+ "
    r"(?:min|minute|hour)s?|last chance|final (?:notice|warning))\b", re.I)
_SAFE_ACCOUNT = re.compile(
    r"\b(safe account|secure account|holding account|move your (?:money|funds)|"
    r"transfer (?:your )?(?:money|funds|balance))\b", re.I)
_AUTHORITY = re.compile(
    r"\b(hmrc|police|fraud team|bank security|government|nat(?:ional)? "
    r"insurance|investigation|warrant|arrest)\b", re.I)
_PRESSURE = re.compile(
    r"\b(do not tell|don'?t tell|keep this (?:secret|confidential)|"
    r"verify (?:your )?(?:identity|account|details)|confirm your (?:pin|"
    r"password|otp|code))\b", re.I)
_SORTCODE = re.compile(r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\b")        # 12-34-56
_ACCOUNTNO = re.compile(r"\b\d{8}\b")                            # 8-digit acct
_MONEYAMT = re.compile(r"[£$€]\s?\d[\d,]*(?:\.\d{2})?")
_LINK = re.compile(r"(https?://|www\.|\bbit\.ly\b|\.co\b|click (?:here|this))",
                   re.I)
_CRYPTO = re.compile(r"\b(bitcoin|btc|crypto|usdt|wallet|gift ?card)\b", re.I)

_FLAG_REGEXES = [
    _URGENCY, _SAFE_ACCOUNT, _AUTHORITY, _PRESSURE,
    _SORTCODE, _ACCOUNTNO, _MONEYAMT, _LINK, _CRYPTO,
]
N_FLAGS = len(_FLAG_REGEXES)

_WORD = re.compile(r"[a-z0-9']+")


def _stable_hash(token: str) -> int:
    """Deterministic, process-independent hash (md5 -> int).

    Python's builtin `hash` is salted per process, so we use md5 for stability
    across runs / machines — important for a reproducible featuriser."""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8],
                          "big")


def _hash_features(text: str, n_buckets: int) -> np.ndarray:
    """Hashing-trick vector over word unigrams+bigrams and char 3-grams.

    Signed hashing (a second hash bit picks +1/-1) reduces collision bias. L2
    normalised so vector length is independent of message length."""
    vec = np.zeros(n_buckets, dtype=np.float64)
    if n_buckets <= 0:
        return vec
    low = text.lower()
    words = _WORD.findall(low)

    tokens: List[str] = []
    tokens.extend(f"w:{w}" for w in words)                       # unigrams
    tokens.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:]))  # bigrams
    compact = low.replace(" ", "")
    tokens.extend(f"c:{compact[i:i + 3]}" for i in range(len(compact) - 2))

    for tok in tokens:
        h = _stable_hash(tok)
        bucket = h % n_buckets
        sign = 1.0 if (h >> 1) & 1 else -1.0
        vec[bucket] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _flag_features(text: str) -> np.ndarray:
    """Binary scam-signal flags (one per regex), order matches `_FLAG_REGEXES`."""
    return np.array(
        [1.0 if rgx.search(text) else 0.0 for rgx in _FLAG_REGEXES],
        dtype=np.float64)


class LocalHashingExtractor:
    """Deterministic, dependency-free reference text featuriser.

    Layout of each row (length == `dim`):
        [ N_FLAGS engineered scam flags | (dim - N_FLAGS) hashing-trick buckets ]

    `dim` must be > N_FLAGS so there is room for at least one hash bucket. This is
    the offline fallback; production swaps in an LLM-embedding backend implementing
    the same `TextFeatureExtractor` protocol and returning the same `[n, dim]`.
    """

    def __init__(self, dim: int = 32):
        if dim <= N_FLAGS:
            raise ValueError(f"dim must be > {N_FLAGS} (engineered flags)")
        self.dim = dim
        self._n_buckets = dim - N_FLAGS

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            t = "" if t is None else str(t)
            flags = _flag_features(t)
            hashed = _hash_features(t, self._n_buckets)
            rows.append(np.concatenate([flags, hashed]))
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack(rows).astype(np.float64)


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------
def extract_text_features(texts: Sequence[str], dim: int = 32) -> np.ndarray:
    """Deterministic local featurisation of unstructured strings -> (n, dim).

    Reference implementation (hashing trick + engineered scam flags). Stable
    across runs/machines, no deps, no network. Production swaps in an LLM backend
    via the `TextFeatureExtractor` protocol returning the same shape."""
    return LocalHashingExtractor(dim=dim).extract(texts)


# ---------------------------------------------------------------------------
# Demonstration helper: do the features carry scam signal?
# ---------------------------------------------------------------------------
def make_text_data(n_per_class: int = 60, seed: int = 0):
    """Synthetic scam vs benign messages (templated with randomised fills).

    Returns (texts, y) where y==1 is scam. Used by the test to show a simple
    classifier on the extracted features separates the two classes above chance.
    """
    rng = np.random.default_rng(seed)

    scam_templates = [
        "URGENT: your account is at risk, move your money to this safe account "
        "now. Sort code {sc} account {ac}. Do not tell anyone.",
        "HMRC fraud team: act immediately or a warrant will be issued. Confirm "
        "your PIN and transfer your funds to the secure account {ac}.",
        "Bank security alert - we detected fraud. Within 10 minutes verify your "
        "identity and move your balance to holding account {sc}.",
        "Final warning: click here {lnk} to secure your account or lose £{amt} "
        "right now. Keep this confidential.",
        "Police investigation in progress. Send £{amt} in bitcoin to wallet to "
        "clear your name immediately, last chance.",
    ]
    benign_templates = [
        "Hi, thanks for lunch yesterday - I'll send you my half later this week.",
        "Your monthly statement is ready to view in the app. No action needed.",
        "Reminder: book club meets Thursday at 7pm at the usual cafe.",
        "Payment received for invoice {ac}, thanks for your business.",
        "The plumber can come Tuesday morning, does that work for you?",
        "Happy birthday! Hope you have a lovely day with the family.",
    ]

    def fill(tpl: str) -> str:
        return tpl.format(
            sc=f"{rng.integers(10,99)}-{rng.integers(10,99)}-{rng.integers(10,99)}",
            ac=f"{rng.integers(10_000_000, 99_999_999)}",
            amt=f"{rng.integers(50, 5000)}",
            lnk="http://bit.ly/" + "".join(
                rng.choice(list("abcdef0123456789"), 6)),
        )

    texts: List[str] = []
    y: List[int] = []
    for _ in range(n_per_class):
        texts.append(fill(scam_templates[rng.integers(len(scam_templates))]))
        y.append(1)
        texts.append(fill(benign_templates[rng.integers(len(benign_templates))]))
        y.append(0)
    return texts, np.array(y, dtype=np.int64)
