from veritas_core.data import make_bank_data, FEATURE_DIM, inject_campaign
def test_shapes_and_rate():
    X,y=make_bank_data(2000,0.02,seed=1)
    assert X.shape==(2000,FEATURE_DIM) and 0.005<y.mean()<0.05
def test_campaign_signature():
    X,y=make_bank_data(2000,0.0,seed=2); Xc,yc=inject_campaign(X,y,100,seed=2)
    assert yc.sum()==100 and Xc[yc==1,-1].mean()>0.8
