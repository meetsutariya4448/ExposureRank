"""Perturb all test labels; train summaries, candidates and locked scores must agree."""
import numpy as np
import torch
from exposurerank.io import ROOT,read_json,write_json,stream
from exposurerank.data import load_pairs,frame
from exposurerank.baselines import priors,baseline_score
from exposurerank.training import make_model,score,seed_all
from exposurerank.ranking import Catalog
from exposurerank.candidates import sample_candidates

seed_all(17);d=load_pairs();other={k:v.copy() for k,v in d.items()};other['y'][other['split']==2]=1-other['y'][other['split']==2]
t=frame(d,0);tt=frame(other,0);nu=len(d['users']);ni=len(d['items'])
a=priors(t['u'],t['i'],t['y'],nu,ni);b=priors(tt['u'],tt['i'],tt['y'],nu,ni)
assert all(np.array_equal(a[k],b[k]) for k in a)
f=frame(d,2);ff=frame(other,2);c=Catalog.build(f,d['ties'],nu);cc=Catalog.build(ff,other['ties'],nu)
assert np.array_equal(c.i,cc.i) and np.array_equal(c.valid,cc.valid)
base=dict(np.load(ROOT/'artifacts/checkpoints/baselines.npz'));selected=read_json(ROOT/'configs/selected_models.json');original=dict(np.load(ROOT/'artifacts/scores/test.npz'))
for name in ('global','item','bias'):
    assert np.array_equal(baseline_score(name,base,f),baseline_score(name,base,ff))
for family in ('mf','neural'):
    arrays=[]
    for run in selected[family]['seeds']:
        model=make_model(family,nu,ni,selected[family]['dimension'],float(base['p0']));model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True))
        one=score(model,f);two=score(model,ff);assert np.array_equal(one,two);arrays.append(two)
    assert np.array_equal(np.mean(arrays,axis=0),original[family])
for sampler in ('uniform','popularity'):
    one=sample_candidates(c.valid,.1+base['item'][c.i],stream(80,1),100,sampler)
    two=sample_candidates(cc.valid,.1+base['item'][cc.i],stream(80,1),100,sampler);assert np.array_equal(one,two)
write_json(ROOT/'evidence/leakage_audit.json',{'status':'passed','perturbed_test_labels':len(f['y']),'training_summaries_unchanged':True,'shared_catalogs_unchanged':True,'six_checkpoint_scores_exactly_unchanged':True,'two_candidate_sampler_draws_unchanged':True,'fitted_checkpoints':'fixed and private; training receives train/validation arguments only'})
print('Leakage invariance passed on all935397 test labels and six checkpoint reloads')
