import numpy as np
def fedavg(updates,weights=None):
    U=np.stack(updates)
    if weights is None: return U.mean(axis=0)
    wv=np.array(weights,float); wv/=wv.sum(); return (U*wv[:,None]).sum(axis=0)
def multi_krum(updates,n_byzantine,m):
    # Thin compatibility shim over the single canonical Krum implementation in
    # robust.multi_krum_select (which clamps negative squared distances from
    # floating point). Returns (aggregate, selected_idx) for callers that only
    # need those two; use robust.multi_krum_select directly for the scores too.
    from .robust import multi_krum_select
    agg, sel, _ = multi_krum_select(updates, n_byzantine=n_byzantine, m=m)
    return agg, sel
