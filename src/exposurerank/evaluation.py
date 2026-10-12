import numpy as np
import pandas as pd
import torch
from .io import ROOT,config,read_json,write_json,sha,revision,stream,status
from .data import load_pairs,frame
from .training import make_model,score,seed_all
from .baselines import baseline_score
from .ranking import Catalog
from .metrics import metrics,linear_coefficients
from .masks import offset
from .uncertainty import paired_bootstrap

MODELS=['global','item','bias','mf','neural']

def freeze():
    d=load_pairs();val=frame(d,1);params=dict(np.load(ROOT/'artifacts/checkpoints/baselines.npz'))
    z=np.log1p(params['pos']);sd=z.std();z=(z-z.mean())/sd if sd else z*0
    offsets={}
    for eps in (.02,.10):
        for mech in ('popularity','outcome'):
            for alpha in ([.10,.25,.50] if eps==.02 else [.25,.50]):
                zz=z[val['i']] if mech=='popularity' else 2*val['y']-1
                offsets[f'{mech}|{alpha}|{eps}']=offset(zz,alpha,eps)
    write_json(ROOT/'configs/mask_offsets.json',{'offsets':offsets,'pop_mean':float(np.log1p(params['pos']).mean()),'pop_sd':float(sd),'source':'validation only'})
    files=[*sorted((ROOT/'configs').glob('*.json')),ROOT/'uv.lock',ROOT/'manifests/data.json',ROOT/'manifests/split.json',ROOT/'manifests/catalogs.json',*sorted((ROOT/'artifacts/checkpoints').glob('*'))]
    selected=read_json(ROOT/'configs/selected_models.json')
    lock={'source_revision':revision(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'contrasts':[[selected['best_learned'],selected['best_simple']],['neural','mf']],'models':MODELS,'test_scoring_started':False,'interpretation':'fixed historical observed-cell panel; bootstrap user sensitivity, simulated designs'}
    write_json(ROOT/'manifests/evaluation_lock.json',lock)

def assert_lock():
    lock=read_json(ROOT/'manifests/evaluation_lock.json')
    for name,h in lock['files'].items():
        if sha(ROOT/name)!=h: raise RuntimeError('Frozen artifact changed: '+name)
    return lock

def evaluate():
    lock=assert_lock();seed_all(17);d=load_pairs();test=frame(d,2);nu=len(d['users']);ni=len(d['items']);cat=Catalog.build(test,d['ties'],nu)
    selected=read_json(ROOT/'configs/selected_models.json');params=dict(np.load(ROOT/'artifacts/checkpoints/baselines.npz'))
    scores={name:baseline_score(name,params,test) for name in MODELS[:3]};seed_rows=[]
    for family in ('mf','neural'):
        per_seed=[]
        for rec in selected[family]['seeds']:
            model=make_model(family,nu,ni,selected[family]['dimension'],float(params['p0']))
            model.load_state_dict(torch.load(ROOT/rec['checkpoint'],weights_only=True));s=score(model,test);per_seed.append(s)
            mm=metrics(cat.outcomes(test['y']),cat.outcomes(test['r']),cat.order(s),cat.valid)
            seed_rows.append({'family':family,'seed':rec['seed'],**{k:float(v.mean()) for k,v in mm.items()}})
        scores[family]=np.mean(per_seed,axis=0)
    np.savez(ROOT/'artifacts/scores/test.npz',**scores)
    y=cat.outcomes(test['y']);r=cat.outcomes(test['r']);orders={m:cat.order(s) for m,s in scores.items()};values={m:metrics(y,r,orders[m],cat.valid) for m in MODELS}
    rows=[];top_pop=np.argsort(params['pos'])[-int(np.ceil(.1*ni)):]
    for m in MODELS:
        top_items=np.take_along_axis(cat.i,orders[m][:,:10],axis=1).ravel();counts=np.bincount(top_items,minlength=ni);shares=counts/counts.sum()
        rows.append({'model':m,'users':nu,'pairs':len(test['y']),**{k:float(v.mean()) for k,v in values[m].items()},'catalog_coverage':len(np.unique(top_items))/len(np.unique(cat.i[cat.valid])),'share_hhi':float(np.sum(shares**2)),'top_training_pop_decile_share':float(np.isin(top_items,top_pop).mean())})
    pd.DataFrame(rows).to_csv(ROOT/'reports/tables/full_reference.csv',index=False);pd.DataFrame(seed_rows).to_csv(ROOT/'reports/tables/seed_sensitivity.csv',index=False)
    contrasts=[]
    for idx,(a,b) in enumerate(lock['contrasts']):
        contrasts.append({'a':a,'b':b,**paired_bootstrap(values[a]['ndcg10'],values[b]['ndcg10'],stream(9,idx))})
    pd.DataFrame(contrasts).to_csv(ROOT/'reports/tables/primary_contrasts.csv',index=False)
    np.savez(ROOT/'artifacts/scores/reference.npz',y=y,r=r,items=cat.i,valid=cat.valid,**{m+'_order':orders[m] for m in MODELS},**{m+'_'+k:v for m in MODELS for k,v in values[m].items()})
    write_json(ROOT/'evidence/score_manifest.json',{'sha256':sha(ROOT/'artifacts/scores/test.npz'),'reference_sha256':sha(ROOT/'artifacts/scores/reference.npz'),'lock_sha256':sha(ROOT/'manifests/evaluation_lock.json'),'revision':revision()})
    sensitivity=[]
    for threshold in (.5,1,2):
        yy=cat.outcomes((test['ratio']>=threshold).astype(float))
        for m in MODELS:
            sensitivity.append({'model':m,'target':'binary','parameter':threshold,**{k:float(v.mean()) for k,v in metrics(yy,r,orders[m],cat.valid).items()}})
    for cap in (1,2,5):
        rr=cat.outcomes(np.minimum(test['ratio'],cap)/cap)
        for m in MODELS:sensitivity.append({'model':m,'target':'bounded_reward','parameter':cap,'reward10':float(metrics(y,rr,orders[m],cat.valid)['reward10'].mean())})
    for source,rr in [('recomputed',cat.outcomes(np.minimum(test['recomputed'],2)/2)),('published',r)]:
        for m in MODELS:sensitivity.append({'model':m,'target':source,'parameter':2,'reward10':float(metrics(y,rr,orders[m],cat.valid)['reward10'].mean())})
    pd.DataFrame(sensitivity).to_csv(ROOT/'reports/tables/target_sensitivity.csv',index=False)
    status(['E6'],'Locked common test scores; full_reference.csv, primary_contrasts.csv and private score_manifest hashes')
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print(pd.DataFrame(contrasts).to_string(index=False),flush=True)
