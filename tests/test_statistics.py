import itertools
import numpy as np
import pytest
from exposurerank.masks import observe,probability,offset,naive_zero
from exposurerank.weighting import ht,ObservedView
from exposurerank.uncertainty import paired_bootstrap,reversal,wilson
from exposurerank.candidates import sample_candidates
from exposurerank.ranking import restrict_order

def test_ht_exhaustive_independent_expectation_variance():
    y=np.array([1.,0,1,1]);pi=np.array([.2,.4,.6,.8]);a=np.array([.4,.3,-.2,.1]);true=np.dot(a,y);vv=np.sum((a*y)**2*(1-pi)/pi);mean=second=varmean=0
    for mask in itertools.product([False,True],repeat=4):
        mask=np.array(mask);prob=np.prod(np.where(mask,pi,1-pi));ix=np.flatnonzero(mask);result=ht(a,ObservedView(ix,y[ix],pi[ix],4))
        mean+=prob*result['estimate'];second+=prob*(result['estimate']-true)**2;varmean+=prob*result['variance']
    assert mean==pytest.approx(true) and second==pytest.approx(vv) and varmean==pytest.approx(vv)

def test_ht_clipping_is_biased_and_full_is_exact():
    a=np.array([1.]);view=ObservedView(np.array([0]),np.array([1.]),np.array([.01]),1)
    assert .01*ht(a,view,clip=10)['estimate']==.1
    full=ObservedView(np.array([0]),np.array([1.]),np.array([1.]),1);assert ht(a,full)['estimate']==1 and ht(a,full)['variance']==0

def test_hidden_lookup_denied_and_no_complete_pi():
    view=ObservedView(np.array([1]),np.array([1.]),np.array([.2]),3)
    with pytest.raises(KeyError):view.lookup(0)
    assert len(view.probabilities)==1 and not hasattr(view,'truth')

@pytest.mark.parametrize('pi',[0.,-1.,1.1])
def test_invalid_observed_positivity(pi):
    with pytest.raises(ValueError):ObservedView(np.array([0]),np.array([1.]),np.array([pi]),1)

def test_validation_offset_hits_density_and_full_endpoint():
    z=np.array([-1.,1.,1.,-1]);c=offset(z,.25);pi=probability('outcome',.25,z,(z+1)/2,c)
    assert pi.mean()==pytest.approx(.25) and pi.min()>.02 and pi.max()<.98
    assert np.all(probability('outcome',1,z,z,c)==1)
    with pytest.raises(ValueError):offset(z,.10,.10)

def test_naive_normalized_ratio_is_not_unbiased():
    # Two items, first positive, second negative: true nDCG1=1.
    # Observe positive only with probability .5 => masked-IDCG=0 half the time.
    assert .5*1+.5*0 != 1

@pytest.mark.parametrize('sampler',['uniform','popularity'])
def test_shared_candidates_seed_full_and_restriction(sampler):
    valid=np.ones((3,8),dtype=bool);w=np.arange(1,9)[None,:]*np.ones((3,1))
    one=sample_candidates(valid,w,np.random.default_rng(1),5,sampler);two=sample_candidates(valid,w,np.random.default_rng(1),5,sampler)
    assert np.array_equal(one,two) and np.all(one.sum(axis=1)==5)
    assert np.array_equal(sample_candidates(valid,w,np.random.default_rng(1),20,sampler),valid)
    orders=np.tile(np.arange(8)[::-1],(3,1));restriction=restrict_order(orders,one)
    for row in range(3): assert np.array_equal(restriction[row,:5],orders[row][one[row,orders[row]]])

def test_paired_bootstrap_identical_models_zero():
    result=paired_bootstrap(np.arange(7),np.arange(7),np.random.default_rng(1),draws=5000)
    assert result['lo']==result['gap']==result['hi']==0

def test_reversal_near_tie_and_wilson_endpoints():
    assert reversal([1,-1],.0001)['state']=='reference near tie'
    result=reversal([1,-1],.1);assert result['reversals']==1 and result['lo']<.5<result['hi']
    lo,hi=wilson(0,200);assert lo==0 and 0<hi<.02
