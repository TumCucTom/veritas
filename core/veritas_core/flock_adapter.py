"""Veritas as a genuine ``flock-sdk`` FlockModel.

The real ``flock_sdk.FlockModel`` (v0.0.3) is an ABC whose ``train`` /
``evaluate`` / ``aggregate`` exchange model parameters as **bytes**, plus an
``init_dataset(dataset_path)`` hook. This adapter implements those exact
signatures, serialising the NumPy logistic weights with ``ndarray.tobytes`` /
``np.frombuffer`` so Veritas can be dropped onto the FLock federated runtime.

If ``flock-sdk`` is unavailable (e.g. an environment without the package), a
minimal fallback base class keeps the module importable and the smoke test
runnable.
"""
import json
import numpy as np
from .data import make_bank_data, inject_campaign, FEATURE_DIM
from .model import init_weights, train_local, recall
from .aggregation import multi_krum

try:
    from flock_sdk import FlockModel
    _REAL_SDK = True
except Exception:  # pragma: no cover - exercised only when SDK is absent
    class FlockModel:  # minimal stand-in matching the real ABC surface
        def init_dataset(self, dataset_path): ...
        def train(self, parameters): ...
        def evaluate(self, parameters): ...
        def aggregate(self, parameters_list): ...
    _REAL_SDK = False

_DIM = FEATURE_DIM + 1  # weights include the bias term


def _to_bytes(w):
    return np.asarray(w, dtype=np.float64).tobytes()


def _from_bytes(parameters):
    if parameters is None:
        return init_weights(FEATURE_DIM)
    if isinstance(parameters, (bytes, bytearray, memoryview)):
        if len(parameters) == 0:
            return init_weights(FEATURE_DIM)
        return np.frombuffer(bytes(parameters), dtype=np.float64).copy()
    # tolerate list/ndarray inputs too (handy for the smoke test / unit use)
    return np.asarray(parameters, dtype=np.float64)


class VeritasFlockModel(FlockModel):
    """Logistic fraud model packaged for the FLock SDK contract."""

    def __init__(self, seed=0, campaign=False):
        self.w = init_weights(FEATURE_DIM)
        self.X, self.y = make_bank_data(3000, 0.03, seed=seed)
        if campaign:
            self.X, self.y = inject_campaign(self.X, self.y, 150, seed=1)

    def init_dataset(self, dataset_path):
        """FLock hands a dataset path; Veritas uses its seeded synthetic banks,
        so this is a no-op kept for ABC compliance."""
        return None

    def train(self, parameters):
        self.w = _from_bytes(parameters)
        self.w = train_local(self.w, self.X, self.y, epochs=8, lr=0.3)
        return _to_bytes(self.w)

    def evaluate(self, parameters):
        w = _from_bytes(parameters) if parameters is not None else self.w
        return {"recall": recall(w, self.X, self.y)}

    def aggregate(self, parameters_list):
        ups = [_from_bytes(p) for p in parameters_list]
        agg, _ = multi_krum(ups, n_byzantine=1, m=max(1, len(ups) - 2))
        return _to_bytes(agg)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    m = VeritasFlockModel()
    p = m.train(None)
    print(f"using_real_flock_sdk={_REAL_SDK}")
    print(m.evaluate(p))
