import numpy as np
from veritas_core.aggregation import fedavg, multi_krum
def test_fedavg_mean():
    assert np.allclose(fedavg([np.array([1.,1.]),np.array([3.,3.])]),[2.,2.])
def test_krum_rejects_outlier():
    honest=[np.array([1.,1.]),np.array([1.1,0.9]),np.array([0.9,1.1]),np.array([1.,1.]),np.array([1.05,0.95])]
    agg,sel=multi_krum(honest+[np.array([50.,-50.])],n_byzantine=1,m=3)
    assert 5 not in sel and np.linalg.norm(agg)<5.0
