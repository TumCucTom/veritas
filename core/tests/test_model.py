from veritas_core.model import init_weights,predict_proba,train_local,recall
from veritas_core.data import make_bank_data,FEATURE_DIM
def test_train_improves_recall():
    X,y=make_bank_data(4000,0.05,seed=3); w0=init_weights(FEATURE_DIM)
    w1=train_local(w0,X,y,epochs=20,lr=0.3); assert recall(w1,X,y)>recall(w0,X,y)+0.2
def test_proba_range():
    X,_=make_bank_data(100,0.05,seed=4); p=predict_proba(init_weights(FEATURE_DIM),X)
    assert p.min()>=0 and p.max()<=1
