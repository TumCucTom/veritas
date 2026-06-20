import numpy as np
from veritas_core.attack import poisoned_update
def test_poison_large():
    h=np.array([1.,1.,1.]); assert np.linalg.norm(poisoned_update(h,10.0))>np.linalg.norm(h)*5
