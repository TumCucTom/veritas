import numpy as np
def fedavg(updates,weights=None):
    U=np.stack(updates)
    if weights is None: return U.mean(axis=0)
    wv=np.array(weights,float); wv/=wv.sum(); return (U*wv[:,None]).sum(axis=0)
def multi_krum(updates,n_byzantine,m):
    U=np.stack(updates); n=len(U); k=max(1,n-n_byzantine-2)
    dist=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d=float(np.sum((U[i]-U[j])**2)); dist[i,j]=dist[j,i]=d
    scores=[np.sort(dist[i])[1:k+1].sum() for i in range(n)]
    sel=list(np.argsort(scores)[:m]); return U[sel].mean(axis=0), sel
