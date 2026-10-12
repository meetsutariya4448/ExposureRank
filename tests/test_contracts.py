import hashlib,itertools
import numpy as np
import pytest
import torch
from exposurerank.data import pair_split,valid,tie_keys,compact_map
from exposurerank.baselines import priors,baseline_score
from exposurerank.factorization import MatrixFactorization
from exposurerank.neural import NeuralInteraction
from exposurerank.ranking import Catalog,restrict_order
from exposurerank.metrics import metrics,linear_coefficients

def toy(y=(0,1,1,0),scores=(.9,.8,.7,.6)):
    f={'u':np.zeros(len(y),dtype=int),'i':np.arange(len(y)),'y':np.array(y,dtype=float),'r':np.array(y,dtype=float)}
    c=Catalog.build(f,tie_keys(f['i']),1);o=c.order(np.array(scores));return f,c,o

def test_pair_sha_independent_known_bytes():
    u=np.array([14,20,7162]);i=np.array([103,10595,106])
    expected=[]
    for a,b in zip(u,i):
        h=hashlib.sha256(('ExposureRank-split-v1|%d|%d'%(a,b)).encode()).hexdigest()[:16]
        v=int(h,16);expected.append(0 if v<.6*2**64 else (1 if v<.8*2**64 else 2))
    assert np.array_equal(pair_split(u,i),expected)

def test_sparse_maps_unknown_rejected():
    assert compact_map(np.array([100,20]),np.array([20,100])).tolist()==[1,0]
    with pytest.raises(ValueError):compact_map(np.array([50]),np.array([20,100]))

def test_heldout_label_permutation_cannot_change_training_features():
    u=np.arange(100);i=np.arange(100)[::-1];s=pair_split(u,i);y=np.ones(100);changed=y.copy();changed[s==2]=0
    a=priors(u[s==0],i[s==0],y[s==0],100,100);b=priors(u[s==0],i[s==0],changed[s==0],100,100)
    assert a['p0']==b['p0'] and np.array_equal(a['item'],b['item'])

@pytest.mark.parametrize('play,duration,ratio,expected',[(1,0,1,False),(-1,1,1,False),(1,1,float('nan'),False),(1,1,-1,False),(2,1,2,True)])
def test_invalid_feedback(play,duration,ratio,expected):
    assert bool(valid(np.array(play),np.array(duration),np.array(ratio)))==expected

def test_train_prior_by_hand():
    p=priors(np.array([0,0,1]),np.array([0,0,1]),np.array([1.,0,1]),2,2)
    assert p['p0']==3/5 and p['item'][0]==pytest.approx(13/22)

def test_missing_is_not_catalog_negative():
    f,c,o=toy();assert c.valid.sum()==4 and c.i.max()==3

def test_ndcg_recall_hand_calculation():
    f,c,o=toy();m=metrics(c.outcomes(f['y']),c.outcomes(f['r']),o,c.valid)
    expected=(1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3))
    assert m['ndcg10'][0]==pytest.approx(expected) and m['recall10'][0]==1

def test_ideal_and_zero_positives():
    f,c,o=toy((1,1,0,0));assert metrics(c.outcomes(f['y']),c.outcomes(f['r']),o,c.valid)['ndcg10'][0]==1
    f,c,o=toy((0,0,0,0));m=metrics(c.outcomes(f['y']),c.outcomes(f['r']),o,c.valid)
    assert m['ndcg10'][0]==m['recall10'][0]==0 and m['zero_positive'][0]==1

def test_ties_and_user_constant_preserve_rank():
    f,c,o=toy(scores=(1,1,1,1));assert np.array_equal(o,c.order(np.full(4,8.)))
    assert np.array_equal(o,np.argsort(tie_keys(np.arange(4)))[None,:])

def test_rank_restriction_direct_sort():
    f,c,o=toy();selected=np.array([[False,True,True,False]]);restricted=restrict_order(o,selected)
    assert restricted[0,:2].tolist()==[1,2]

@pytest.mark.parametrize('klass',[MatrixFactorization,NeuralInteraction])
def test_pytorch_update_reload_dropout(klass,tmp_path):
    torch.manual_seed(17);model=klass(3,4,16,.5);u=torch.tensor([0,1,2]);i=torch.tensor([1,2,3]);y=torch.tensor([1.,0,1])
    before=model.user.weight.detach().clone();opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(u,i),y)+.001/2*model.penalty();loss.backward();opt.step()
    assert not torch.equal(before,model.user.weight);model.eval();s=model(u,i).detach()
    assert torch.equal(s,model(u,i));cp=tmp_path/'m.pt';torch.save(model.state_dict(),cp)
    reload=klass(3,4,16,.5);reload.load_state_dict(torch.load(cp,weights_only=True));reload.eval();assert torch.equal(s,reload(u,i))

def test_penalty_excludes_global_intercept():
    model=MatrixFactorization(2,3,16,.5);p=model.penalty().item()
    with torch.no_grad(): model.mu+=100
    assert model.penalty().item()==p

def test_linear_dcg_matches_metric():
    f,c,o=toy();y=c.outcomes(f['y']);a=linear_coefficients(o,c.valid)
    assert (a*y).sum()==pytest.approx(metrics(y,y,o,c.valid)['dcg10'].mean())
