"""Recompute headlines and gate accounting; refuse missing/wrong experimental cells."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from exposurerank.io import ROOT,read_json,write_json,sha
from exposurerank.evaluation import assert_lock

lock=assert_lock();checks=[]
for claim in read_json(ROOT/'evidence/claim_ledger.json'):
    df=pd.read_csv(ROOT/claim['table'])
    for k,v in claim['key'].items():df=df[df[k]==v]
    assert len(df)==1 and np.isclose(df.iloc[0][claim['field']],claim['value'],atol=1e-12)
    checks.append(claim['id'])
mask=pd.read_csv(ROOT/'reports/tables/mask_repetitions.csv');cand=pd.read_csv(ROOT/'reports/tables/candidate_repetitions.csv')
assert mask.condition.nunique()==13 and cand.condition.nunique()==6
assert (mask.groupby(['condition','model','metric']).size()==200).all()
assert (cand.groupby(['condition','model','metric']).size()==200).all()
full=pd.read_csv(ROOT/'reports/tables/full_reference.csv');assert len(full)==5 and (full.users==1411).all() and (full.pairs==935397).all()
assert full.ndcg10.between(0,1).all() and full.recall10.between(0,1).all()
score=read_json(ROOT/'evidence/score_manifest.json');assert score['sha256']==sha(ROOT/'artifacts/scores/test.npz')
assert read_json(ROOT/'evidence/mask_manifest.json')['scores_sha256']==score['sha256']
assert read_json(ROOT/'evidence/candidate_manifest.json')['scores_sha256']==score['sha256']
runs=read_json(ROOT/'evidence/runs.json');assert len(runs)==16
for family in ('mf','neural'):
    fam=[x for x in runs if x['family']==family];assert len(fam)==8 and set(x['seed'] for x in fam)=={17,29,43}
payload=read_json(ROOT/'public/aggregate_payload.json')
forbidden={'user_id','video_id','indices','embeddings','checkpoint_data','scores'}
def scan(obj):
    if isinstance(obj,dict):
        assert not forbidden.intersection(obj)
        for v in obj.values():scan(v)
    elif isinstance(obj,list):
        for v in obj:scan(v)
scan(payload)
order_rows=[]
for kind,data in [('masks',mask),('candidates',cand)]:
    ref=full.set_index('model').ndcg10
    reference_rank=ref.rank(method='min').reindex(['global','item','bias','mf','neural']).to_numpy()
    for (condition,rep),part in data[data.metric=='ndcg10'].groupby(['condition','rep']):
        scores=part.set_index('model').value.reindex(['global','item','bias','mf','neural'])
        # Complete order is descriptive; primary reversal near-tie threshold remains separate.
        tau=kendalltau(reference_rank,scores.rank(method='min').to_numpy()).statistic
        order_rows.append({'experiment':kind,'condition':condition,'rep':rep,'kendall_tau':tau})
pd.DataFrame(order_rows).groupby(['experiment','condition']).kendall_tau.agg(['mean','std','count']).reset_index().to_csv(ROOT/'reports/tables/order_stability.csv',index=False)
write_json(ROOT/'evidence/claim_audit.json',{'status':'passed','headline_claims_recomputed':len(checks),'claim_ids':checks,'mask_conditions':13,'candidate_conditions':6,'independent_nonfull_draws':3800,'learned_runs':16,'frozen_score_hashes_unchanged':True,'aggregate_payload_keys_audited':True})
print('Claim and accounting audit passed:',len(checks),'headlines,3800 nonfull draws,16 learned runs')
