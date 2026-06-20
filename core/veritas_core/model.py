import numpy as np
def init_weights(dim): return np.zeros(dim+1)
def _aug(X): return np.hstack([X,np.ones((X.shape[0],1))])
def predict_proba(w,X): return 1.0/(1.0+np.exp(-np.clip(_aug(X)@w,-30,30)))
def train_local(w,X,y,epochs=10,lr=0.3,l2=1e-4):
    Xa=_aug(X); w=w.copy(); n=len(y)
    npos=int((y==1).sum()); pw=np.where(y==1,(y==0).sum()/max(1,npos),1.0)
    for _ in range(epochs):
        g=Xa.T@((1.0/(1.0+np.exp(-np.clip(Xa@w,-30,30)))-y)*pw)/n + l2*w; w-=lr*g
    return w
def recall(w,X,y,thr=0.5):
    if y.sum()==0: return 1.0
    pred=predict_proba(w,X)>thr; return float((pred&(y==1)).sum()/y.sum())
