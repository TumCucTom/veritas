"""Veritas as a genuine ``flock-sdk`` FlockModel.

The real ``flock_sdk.FlockModel`` (v0.0.3) is an ``abc.ABC`` whose
``train`` / ``evaluate`` / ``aggregate`` exchange model parameters as **bytes**,
plus an ``init_dataset(dataset_path)`` hook. flock-sdk is FREE and LOCAL: it is a
tiny pure-Python ABC -- no API key, no hosted endpoint, no Docker required to
subclass and drive it. This adapter implements those exact signatures so Veritas'
NumPy logistic fraud model can be dropped onto the FLock federated runtime and run
a real local federation.

Signatures matched exactly against the installed SDK::

    init_dataset(self, dataset_path: str) -> None
    train(self, parameters: bytes) -> bytes
    evaluate(self, parameters: bytes) -> bytes
    aggregate(self, parameters_list: list[bytes]) -> bytes

Bytes serialization format
--------------------------
Weights are a 1-D ``np.ndarray`` of shape ``(FEATURE_DIM + 1,)`` (last entry is
the bias). We serialise losslessly with ``numpy.save`` into an in-memory
``BytesIO`` buffer (the ``.npy`` format), which records dtype + shape, so
``_from_bytes(_to_bytes(w))`` round-trips bit-for-bit. ``evaluate`` returns the
metric as bytes via ``str(round(value, 6)).encode()`` (UTF-8), which decodes back
to a float with ``float(b.decode())``.

If ``flock-sdk`` is unavailable (an environment without the package), a minimal
fallback ABC keeps the module importable. When the SDK IS installed (it is here),
the real ``flock_sdk.FlockModel`` is the base class -- ``issubclass`` proves it.
"""
import io
import numpy as np

from .data import make_bank_data, inject_campaign, FEATURE_DIM
from .model import init_weights, train_local, recall, predict_proba
from .aggregation import fedavg, multi_krum

try:
    from flock_sdk import FlockModel
    USING_REAL_FLOCK_SDK = True
except Exception:  # pragma: no cover - exercised only when SDK is absent
    from abc import ABC, abstractmethod

    class FlockModel(ABC):  # minimal stand-in matching the real ABC surface
        @abstractmethod
        def init_dataset(self, dataset_path: str) -> None: ...
        @abstractmethod
        def train(self, parameters: bytes) -> bytes: ...
        @abstractmethod
        def evaluate(self, parameters: bytes) -> bytes: ...
        @abstractmethod
        def aggregate(self, parameters_list: list) -> bytes: ...

    USING_REAL_FLOCK_SDK = False

_DIM = FEATURE_DIM + 1  # weights include the bias term


# --------------------------------------------------------------------------- #
# Lossless bytes <-> ndarray serialization (the FLock parameter wire format)
# --------------------------------------------------------------------------- #
def _to_bytes(weights: np.ndarray) -> bytes:
    """Serialise a weight vector losslessly using the NumPy ``.npy`` format."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(weights, dtype=np.float64))
    return buf.getvalue()


def _from_bytes(parameters) -> np.ndarray:
    """Deserialise weights from FLock parameter bytes.

    Empty / ``None`` parameters (the first federation round, before any global
    model exists) yield a fresh ``init_weights`` vector.
    """
    if parameters is None:
        return init_weights(FEATURE_DIM)
    if isinstance(parameters, (bytes, bytearray, memoryview)):
        b = bytes(parameters)
        if len(b) == 0:
            return init_weights(FEATURE_DIM)
        return np.load(io.BytesIO(b), allow_pickle=False).astype(np.float64)
    # tolerate list/ndarray inputs too (handy for direct unit use)
    return np.asarray(parameters, dtype=np.float64)


def _metric_to_bytes(value: float) -> bytes:
    """Encode a scalar metric as bytes (decode with ``float(b.decode())``)."""
    return str(round(float(value), 6)).encode("utf-8")


def metric_from_bytes(b) -> float:
    """Decode a metric produced by ``evaluate`` back into a float."""
    if isinstance(b, (bytes, bytearray, memoryview)):
        return float(bytes(b).decode("utf-8"))
    return float(b)


class VeritasFlockModel(FlockModel):
    """Logistic fraud model packaged for the FLock SDK bytes contract.

    Each instance is one *bank*: it owns local, seeded synthetic data and never
    shares raw rows -- only model-weight bytes cross the FLock boundary.
    """

    def __init__(self, seed: int = 0, campaign: bool = False, n: int = 3000,
                 fraud_rate: float = 0.03, robust: bool = False):
        self.seed = seed
        self.campaign = campaign
        self.n = n
        self.fraud_rate = fraud_rate
        self.robust = robust  # use Multi-Krum instead of FedAvg in aggregate()
        self.X = None
        self.y = None
        self.w = init_weights(FEATURE_DIM)
        # Eagerly prepare data so the model is usable before FLock calls
        # init_dataset; init_dataset re-keys it from the path if given one.
        self.init_dataset(f"seed://{seed}")

    # -- FLock ABC: init_dataset ------------------------------------------- #
    def init_dataset(self, dataset_path: str) -> None:
        """Prepare this bank's LOCAL data.

        Veritas uses seeded synthetic bank data instead of a file on disk. The
        seed is parsed from ``dataset_path`` (e.g. ``"seed://7"`` or a path whose
        trailing token is an int); ``campaign`` is set on the instance. Stores
        ``self.X`` / ``self.y``. Returns ``None`` per the ABC contract.
        """
        seed = self.seed
        if dataset_path:
            tail = str(dataset_path).rstrip("/").split("/")[-1]
            digits = "".join(ch for ch in tail if (ch.isdigit() or ch == "-"))
            if digits.lstrip("-").isdigit():
                seed = int(digits)
        self.seed = seed
        X, y = make_bank_data(self.n, self.fraud_rate, seed=seed)
        if self.campaign:
            X, y = inject_campaign(X, y, 150, seed=1)
        self.X, self.y = X, y
        return None

    # -- FLock ABC: train -------------------------------------------------- #
    def train(self, parameters: bytes) -> bytes:
        """Load weights from bytes, train on local data, return new weights bytes.

        Empty / ``None`` ``parameters`` (round 0) start from ``init_weights``.
        """
        self.w = _from_bytes(parameters)
        self.w = train_local(self.w, self.X, self.y, epochs=8, lr=0.3)
        return _to_bytes(self.w)

    # -- FLock ABC: evaluate ----------------------------------------------- #
    def evaluate(self, parameters: bytes) -> bytes:
        """Load weights from bytes, return RECALL on local data as bytes.

        Recall is the operative fraud metric (catching the campaign). Decode with
        ``metric_from_bytes`` / ``float(b.decode())``.
        """
        w = _from_bytes(parameters)
        return _metric_to_bytes(recall(w, self.X, self.y))

    # -- FLock ABC: aggregate ---------------------------------------------- #
    def aggregate(self, parameters_list: list) -> bytes:
        """Aggregate a list of weight-bytes into one global weight-bytes.

        FedAvg by default (the SDK's documented behaviour); robust Multi-Krum
        when ``self.robust`` is set, to shrug off poisoned updates.
        """
        ups = [_from_bytes(p) for p in parameters_list]
        if self.robust and len(ups) >= 4:
            agg, _ = multi_krum(ups, n_byzantine=1, m=max(1, len(ups) - 2))
        else:
            agg = fedavg(ups)
        return _to_bytes(np.asarray(agg, dtype=np.float64))

    # -- convenience (not part of the ABC) --------------------------------- #
    def accuracy(self, parameters) -> float:
        w = _from_bytes(parameters)
        pred = predict_proba(w, self.X) > 0.5
        return float((pred == (self.y == 1)).mean())


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    m = VeritasFlockModel()
    print(f"USING_REAL_FLOCK_SDK={USING_REAL_FLOCK_SDK}")
    print(f"bases={[b.__name__ for b in VeritasFlockModel.__bases__]}")
    p = m.train(b"")
    print("recall_bytes=", m.evaluate(p), "->", metric_from_bytes(m.evaluate(p)))
