import numpy as np
from veritas_core.dp import clip_update, add_noise
def test_clip(): assert np.linalg.norm(clip_update(np.array([3.,4.]),1.0))<=1.0+1e-9
def test_noise():
    n=add_noise(np.zeros(1000),0.1,1.0,np.random.default_rng(0))
    assert not np.allclose(n,0) and abs(n.mean())<0.05
