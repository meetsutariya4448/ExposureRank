"""Independent reference calculations and failure fixtures from blueprint contracts."""
import math,itertools
import numpy as np
import pytest
import duckdb
from exposurerank.io import ROOT
from exposurerank.metrics import metrics
from exposurerank.ranking import Catalog
from exposurerank.data import tie_keys
from exposurerank.masks import standardized_popularity,observe
from exposurerank.weighting import ObservedView,ht
from exposurerank.uncertainty import mc_summary,wilson

def test_random_catalog_metrics_against_scalar_reference():
    rng=np.random.default_rng(8675309)
    for n in (1,2,9,10,19,20,21,53):
        for _ in range(12):
            y=rng.integers(0,2,n).astype(float);r=rng.random(n);s=rng.integers(0,4,n).astype(float)
            f={'u':np.zeros(n,dtype=int),'i':np.arange(n)};ties=tie_keys(np.arange(n))
            c=Catalog.build(f,ties,1);order=c.order(s);actual=metrics(y[None,:],r[None,:],order,c.valid)
            scalar=sorted(range(n),key=lambda i:(-s[i],int(ties[i]),i))
            numerator=sum(y[scalar[j]]/math.log2(j+2) for j in range(min(10,n)))
            ideal=sum(1/math.log2(j+2) for j in range(min(10,int(y.sum()))))
            expected={'ndcg10':numerator/ideal if ideal else 0,'dcg10':numerator,'recall10':sum(y[i] for i in scalar[:10])/y.sum() if y.sum() else 0,'recall20':sum(y[i] for i in scalar[:20])/y.sum() if y.sum() else 0,'reward10':sum(r[i] for i in scalar[:10])/min(n,10)}
            for key,value in expected.items(): assert actual[key][0]==pytest.approx(value,abs=1e-12)

@pytest.mark.parametrize('pi',[float('nan'),float('inf')])
def test_nonfinite_probability_rejected(pi):
    with pytest.raises(ValueError):ObservedView(np.array([0]),np.array([1.]),np.array([pi]),1)
    with pytest.raises(ValueError):observe(np.ones((1,1)),np.array([[pi]]),np.ones((1,1),dtype=bool),np.random.default_rng(0))

@pytest.mark.parametrize('indices',[np.array([-1]),np.array([1]),np.array([.5])])
def test_observed_index_boundaries_rejected(indices):
    with pytest.raises(ValueError):ObservedView(indices,np.array([1.]),np.array([.5]),1)

def test_zero_popularity_variance_fallback():
    assert np.array_equal(standardized_popularity(np.ones(4),np.log(2),0),np.zeros(4))

def test_actual_naive_ratio_counterexample_with_metric_function():
    y=np.array([[1.,0.]]);valid=np.ones_like(y,dtype=bool);order=np.array([[0,1]])
    mean=0
    for first,second in itertools.product((0,1),repeat=2):
        masked=y*np.array([[first,second]])
        mean+=.25*metrics(masked,masked,order,valid)['ndcg10'][0]
    assert mean==.5 and metrics(y,y,order,valid)['ndcg10'][0]==1

def test_mc_interval_scope_and_width():
    result=mc_summary(np.array([1.,2.,3.]),2)
    assert result['mcse']==pytest.approx(1/math.sqrt(3))
    assert result['mc_mean_lo']<2<result['mc_mean_hi'] and result['repetitions']==3
    assert mc_summary(np.array([2.]),2)['mc_mean_lo']==2
    with pytest.raises(ValueError):wilson(3,2)

def test_sql_quality_rejects_null_observations():
    con=duckdb.connect()
    con.execute('CREATE TABLE small(user_id INTEGER,video_id INTEGER,watch_ratio DOUBLE,video_duration DOUBLE,play_duration DOUBLE)')
    con.execute('INSERT INTO small VALUES(1,1,NULL,10,10),(1,2,1,NULL,10),(1,3,1,10,NULL),(1,4,1,10,10)')
    row=con.execute((ROOT/'sql/02_duration_ratio.sql').read_text()).fetchdf().iloc[0]
    assert row['invalid']==3
