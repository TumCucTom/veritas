import numpy as np
FEATURE_DIM=10  # amount,oldOrig,newOrig,oldDest,newDest,accountAge,velocity,fanout,isTransfer,campaignSig
def make_bank_data(n,fraud_rate,seed):
    rng=np.random.default_rng(seed); X=rng.normal(0,1,(n,FEATURE_DIM)); X[:,-1]=0.0
    y=(rng.random(n)<fraud_rate).astype(np.int64)
    X[y==1,0]+=1.5; X[y==1,6]+=1.5; X[y==1,7]+=1.5
    return X.astype(np.float64), y
def inject_campaign(X,y,n_campaign,seed):
    # "Safe-account mule" typology: deliberately looks LEGITIMATE on the
    # generic fraud axes (amount[0], velocity[6], fanout[7]) so a detector
    # trained only on ordinary fraud misses it. The novel signal lives in the
    # campaign-signature feature [-1]; only a model that has SEEN the campaign
    # (directly, or via federated sharing) learns to flag it.
    rng=np.random.default_rng(seed+999); Xc=rng.normal(0,1,(n_campaign,X.shape[1]))
    Xc[:,0]=-0.5; Xc[:,6]=-0.5; Xc[:,7]=-0.5; Xc[:,5]=-2.0; Xc[:,-1]=1.5
    yc=np.ones(n_campaign,dtype=np.int64)
    return np.vstack([X,Xc]), np.concatenate([y,yc])
