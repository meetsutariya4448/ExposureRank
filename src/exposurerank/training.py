"""Training receives train+validation only. Test frame is not an argument."""
import time, resource, platform, copy
import numpy as np
import torch
import pandas as pd
from .factorization import MatrixFactorization
from .neural import NeuralInteraction
from .baselines import priors,fit_bias,baseline_score
from .ranking import Catalog
from .metrics import metrics
from .data import load_pairs,frame
from .io import ROOT,config,write_json,read_json,sha,revision,status

def seed_all(seed):
    torch.set_num_threads(config('models')['threads']);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def make_model(family,nu,ni,d,p0):
    if family not in ('mf','neural'): raise ValueError('Unregistered learned family')
    return (MatrixFactorization if family=='mf' else NeuralInteraction)(nu,ni,d,p0)

def score(model,f,batch=65536):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(f['u']),batch):
            u=torch.from_numpy(f['u'][start:start+batch].astype(np.int64));i=torch.from_numpy(f['i'][start:start+batch].astype(np.int64))
            out.append(model(u,i).sigmoid().numpy())
    return np.concatenate(out)

def fit(family,dimension,l2,seed,train,val,cat,nu,ni,p0,epochs=None,tag=None):
    cfg=config('models');seed_all(seed);start=time.perf_counter()
    model=make_model(family,nu,ni,dimension,p0);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=0)
    tensors=[torch.from_numpy(train[k].astype(np.int64 if k!='y' else np.float32)) for k in ('u','i','y')]
    generator=torch.Generator().manual_seed(seed);best=-np.inf;wait=0;history=[];best_state=None;best_epoch=0
    for epoch in range(1,(epochs or cfg['max_epochs'])+1):
        epoch_start=time.perf_counter();model.train();perm=torch.randperm(len(train['y']),generator=generator);total=0
        for start_idx in range(0,len(perm),cfg['batch_size']):
            ix=perm[start_idx:start_idx+cfg['batch_size']];u,i,y=[t[ix] for t in tensors]
            opt.zero_grad(set_to_none=True);logits=model(u,i)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+l2/2*model.penalty()
            loss.backward();opt.step();total+=loss.item()*len(ix)
        scores=score(model,val);order=cat.order(scores);values=metrics(cat.outcomes(val['y']),cat.outcomes(val['r']),order,cat.valid)
        ndcg=float(values['ndcg10'].mean());history.append({'epoch':epoch,'loss':total/len(perm),'ndcg10':ndcg,'seconds':time.perf_counter()-epoch_start})
        if ndcg>best+cfg['min_delta']:
            best=ndcg;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());wait=0
        else: wait+=1
        print(tag or family, 'epoch',epoch,'validation nDCG',round(ndcg,6),'seconds',round(history[-1]['seconds'],2),flush=True)
        if wait>=cfg['patience']: break
    model.load_state_dict(best_state);model.eval();scores=score(model,val)
    metadata={'family':family,'dimension':dimension,'l2':l2,'seed':seed,'best_epoch':best_epoch,'validation_ndcg10':best,'seconds':time.perf_counter()-start,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'parameters':sum(p.numel() for p in model.parameters()),'history':history,'revision':revision(),'train_pairs':len(train['y']),'validation_pairs':len(val['y'])}
    if metadata['peak_process_rss_gib']>8: raise MemoryError('Training process exceeded 8GiB RSS')
    return model,scores,metadata

def preflight(train,val,cat,nu,ni,p0):
    rows=np.flatnonzero(train['u']<100);vrows=np.flatnonzero(val['u']<100)
    tt={k:a[rows] for k,a in train.items()};vv={k:a[vrows] for k,a in val.items()}
    cc=Catalog.build(vv,load_pairs()['ties'],100);probes=[]
    for family in ('mf','neural'):
        _,_,record=fit(family,32,1e-5,17,tt,vv,cc,nu,ni,p0,epochs=2,tag='preflight '+family);probes.append(record)
    # Fifteen epochs, eight actual runs/family (six grid + two selected seeds),
    # scale measured full validation+training epoch by train-row ratio and 1.5 safety.
    projected=sum(max(h['seconds'] for h in p['history'])*len(train['y'])/len(tt['y'])*15*8*1.5 for p in probes)
    evidence={'probes':probes,'projected_core_training_hours':projected/3600,'safety_multiplier':1.5,'training_rows':len(tt['y']),'validation_rows':len(vv['y']),'passed':projected<=8*3600 and max(p['peak_process_rss_gib'] for p in probes)<8,'hardware':platform.platform(),'torch':torch.__version__}
    write_json(ROOT/'evidence/mf_preflight.json',evidence)
    if not evidence['passed']: raise RuntimeError('Mandatory resource preflight failed; requires a pretest scope decision')

def train_all():
    d=load_pairs();train=frame(d,0);val=frame(d,1);nu=len(d['users']);ni=len(d['items']);cat=Catalog.build(val,d['ties'],nu)
    base=priors(train['u'],train['i'],train['y'],nu,ni);bias=fit_bias(train['u'],train['i'],train['y'],nu,ni)
    np.savez(ROOT/'artifacts/checkpoints/baselines.npz',**base,**bias)
    baseline=[];val_scores={}
    for name in ('global','item','bias'):
        params=base if name!='bias' else bias;s=baseline_score(name,params,val);val_scores[name]=s
        value=metrics(cat.outcomes(val['y']),cat.outcomes(val['r']),cat.order(s),cat.valid)
        baseline.append({'model':name,**{k:float(v.mean()) for k,v in value.items()}})
    pd.DataFrame(baseline).to_csv(ROOT/'reports/tables/baseline_validation.csv',index=False)
    write_json(ROOT/'evidence/bias_convergence.json',{k:bias[k] for k in ('seconds','iterations','gradient_max','converged')})
    status(['E3'],'Three train-only baselines; baseline_validation.csv and bias_convergence.json')
    preflight(train,val,cat,nu,ni,base['p0'])
    cfg=config('models');records=[];selected={}
    for family in cfg['families']:
        candidates=[]
        for dim in cfg['dimensions']:
            for l2 in cfg['l2']:
                tag=f'{family}_d{dim}_l{l2}_s17';checkpoint=ROOT/'artifacts/checkpoints'/f'{tag}.pt';meta_path=ROOT/'evidence'/f'{tag}.json'
                if checkpoint.exists() and meta_path.exists():
                    rec=read_json(meta_path)
                else:
                    model,s,rec=fit(family,dim,l2,17,train,val,cat,nu,ni,base['p0'],tag=tag)
                    torch.save(model.state_dict(),checkpoint);write_json(meta_path,rec)
                    reload=make_model(family,nu,ni,dim,base['p0']);reload.load_state_dict(torch.load(checkpoint,weights_only=True));assert np.array_equal(s,score(reload,val))
                rec['checkpoint']=str(checkpoint.relative_to(ROOT));rec['checkpoint_sha256']=sha(checkpoint);records.append(rec);candidates.append(rec)
                write_json(ROOT/'evidence/runs.json',records)
        best=max(x['validation_ndcg10'] for x in candidates)
        chosen=sorted((x for x in candidates if x['validation_ndcg10']>=best-1e-4),key=lambda x:(x['dimension'],-x['l2'],x['parameters']))[0]
        seed_records=[chosen];seed_scores=[]
        for seed in cfg['selected_seeds']:
            tag=f'{family}_d{chosen["dimension"]}_l{chosen["l2"]}_s{seed}';cp=ROOT/'artifacts/checkpoints'/f'{tag}.pt';mp=ROOT/'evidence'/f'{tag}.json'
            if seed!=17 and not cp.exists():
                model,s,rec=fit(family,chosen['dimension'],chosen['l2'],seed,train,val,cat,nu,ni,base['p0'],tag=tag);torch.save(model.state_dict(),cp);write_json(mp,rec)
            rec=read_json(mp);rec['checkpoint']=str(cp.relative_to(ROOT));rec['checkpoint_sha256']=sha(cp)
            if seed!=17: records.append(rec);seed_records.append(rec)
            model=make_model(family,nu,ni,chosen['dimension'],base['p0']);model.load_state_dict(torch.load(cp,weights_only=True));seed_scores.append(score(model,val))
        ensemble=np.mean(seed_scores,axis=0);val_scores[family]=ensemble
        metric=float(metrics(cat.outcomes(val['y']),cat.outcomes(val['r']),cat.order(ensemble),cat.valid)['ndcg10'].mean())
        selected[family]={'dimension':chosen['dimension'],'l2':chosen['l2'],'seeds':seed_records,'validation_ensemble_ndcg10':metric}
        status(['E4' if family=='mf' else 'E5'],f'{family}: six configs and selected seeds17/29/43 actually trained; evidence/runs.json')
    best_simple=max(x['ndcg10'] for x in baseline)
    simple=next(x['model'] for x in baseline if x['ndcg10']>=best_simple-1e-4)
    learned='mf' if selected['mf']['validation_ensemble_ndcg10']>=selected['neural']['validation_ensemble_ndcg10']-1e-4 else 'neural'
    selected.update(best_simple=simple,best_learned=learned)
    write_json(ROOT/'configs/selected_models.json',selected);write_json(ROOT/'evidence/runs.json',records)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ('history',)} for x in records]).to_csv(ROOT/'reports/tables/training_runs.csv',index=False)
    np.savez(ROOT/'artifacts/scores/validation.npz',**val_scores)
    print('Validation-selected models:',simple,learned,flush=True)
