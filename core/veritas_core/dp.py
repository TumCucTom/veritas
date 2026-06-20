import numpy as np
def clip_update(u,max_norm):
    nrm=float(np.linalg.norm(u)); return u if nrm<=max_norm else u*(max_norm/nrm)
def add_noise(u,sigma,max_norm,rng): return u+rng.normal(0,sigma*max_norm,size=u.shape)
def privatize(u,max_norm,sigma,rng): return add_noise(clip_update(u,max_norm),sigma,max_norm,rng)
