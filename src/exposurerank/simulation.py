"""Simulator/evaluator owns reference truth; HT receives observed-only views."""
import time,hashlib
import numpy as np
import pandas as pd
from .io import ROOT,config,read_json,write_json,sha,stream,status
from .evaluation import MODELS,assert_lock
from .metrics import metrics,linear_coefficients
from .ranking import restrict_order
from .masks import probability,observe,naive_zero
from .weighting import ObservedView,ht
from .candidates import sample_candidates
from .uncertainty import mc_summary,reversal,paired_bootstrap,wilson

def inputs():
    assert_lock();p=dict(np.load(ROOT/'artifacts/scores/reference.npz'));base=dict(np.load(ROOT/'artifacts/checkpoints/baselines.npz'))
    return p,base

def summaries(rows,kind,p):
    df=pd.DataFrame(rows);out=[]
    grouping=['condition','model','metric']
    for key,g in df.groupby(grouping,sort=False):
        ref=float(p[key[1]+'_'+key[2]].mean())
        out.append(dict(zip(grouping,key))|mc_summary(g.value.to_numpy(),ref)|{'reference':ref,'experiment':kind})
    return pd.DataFrame(out)

def gaps_summary(rows):
    df=pd.DataFrame(rows);out=[]
    for (condition,a,b),g in df.groupby(['condition','a','b'],sort=False):
        ref=float(g.reference.iloc[0]);v=g.gap.to_numpy()
        out.append({'condition':condition,'a':a,'b':b,'reference_gap':ref,**mc_summary(v,ref),**reversal(v,ref)})
    return pd.DataFrame(out)

def mask_study():
    start=time.perf_counter();p,base=inputs();cfg=config('experiments');R=cfg['repetitions'];valid=p['valid'];y=p['y'];reward=p['r'];orders={m:p[m+'_order'] for m in MODELS}
    dcfg=read_json(ROOT/'configs/mask_offsets.json');z=(np.log1p(base['pos'])-dcfg['pop_mean'])/dcfg['pop_sd'];zz=z[p['items']]
    coeff={m:linear_coefficients(orders[m],valid) for m in MODELS};reward_coeff={m:linear_coefficients(orders[m],valid,reward=True) for m in MODELS}
    conditions=[(mech,a,.02) for mech in ('uniform','popularity','outcome') for a in (.1,.25,.5)]
    conditions += [(mech,a,.1) for mech in ('popularity','outcome') for a in (.25,.5)]
    rows=[];linear=[];gaps=[];uncertainty=[];manifest=[];pairs=[(a,b) for ai,a in enumerate(MODELS) for b in MODELS[ai+1:]]
    primary=assert_lock()['contrasts']
    for ci,(mech,alpha,epsilon) in enumerate(conditions):
        cond=f'{mech}|{alpha}|floor{epsilon}';constant=dcfg['offsets'].get(f'{mech}|{alpha}|{epsilon}')
        pi=probability(mech,alpha,zz,y,constant,epsilon);mean_pi=float(pi[valid].mean());digest=hashlib.sha256();per_user={tuple(pair):np.zeros(len(y)) for pair in primary}
        for rep in range(R):
            observed=observe(y,pi,valid,stream(7,ci,rep));digest.update(observed.indices.tobytes())
            naive=naive_zero(observed,y.shape)
            rv=ObservedView(observed.indices,reward.ravel()[observed.indices].copy(),observed.probabilities,observed.size)
            # Both label/reward views contain observed cells only.
            masked_reward=naive_zero(rv,y.shape);values={}
            for m in MODELS:
                mm=metrics(naive,masked_reward,orders[m],valid);values[m]=mm['ndcg10']
                for metric in ('ndcg10','recall10','recall20','dcg10','reward10'):
                    rows.append({'condition':cond,'rep':rep,'model':m,'metric':metric,'value':float(mm[metric].mean()),'expected_density':mean_pi,'realized_density':len(observed.indices)/valid.sum()})
                for target,a,view,ref in [('dcg10',coeff[m],observed,float(p[m+'_dcg10'].mean())),('reward10',reward_coeff[m],rv,float(p[m+'_reward10'].mean()))]:
                    result=ht(a,view);half=1.959963984540054*np.sqrt(result['variance'])
                    # evaluator-only truth/design variance does not enter HT interface.
                    truthvar=float(np.sum((a*(y if target=='dcg10' else reward))**2*(1-pi)/pi))
                    linear.append({'condition':cond,'rep':rep,'model':m,'target':target,'method':'ht','reference':ref,'true_variance':truthvar,'covered':abs(result['estimate']-ref)<=half,'zero_variance':result['variance']==0,**result})
                    if target=='dcg10':
                        for clip in (10,50):
                            linear.append({'condition':cond,'rep':rep,'model':m,'target':target,'method':f'clip{clip}','reference':ref,'true_variance':float('nan'),'covered':False,'zero_variance':False,**ht(a,view,clip)})
            for a,b in pairs:
                ref=float((p[a+'_ndcg10']-p[b+'_ndcg10']).mean());gaps.append({'condition':cond,'rep':rep,'a':a,'b':b,'reference':ref,'gap':float((values[a]-values[b]).mean())})
            for a,b in primary:
                per_user[(a,b)]+=values[a]-values[b]
                ref=float((p[a+'_dcg10']-p[b+'_dcg10']).mean());result=ht(coeff[a]-coeff[b],observed);half=1.959963984540054*np.sqrt(result['variance'])
                linear.append({'condition':cond,'rep':rep,'model':a+'-'+b,'target':'dcg10','method':'paired_ht','reference':ref,'true_variance':float(np.sum(((coeff[a]-coeff[b])*y)**2*(1-pi)/pi)),'covered':abs(result['estimate']-ref)<=half,'zero_variance':result['variance']==0,**result})
        for idx,((a,b),v) in enumerate(per_user.items()):
            uncertainty.append({'condition':cond,'a':a,'b':b,'scope':'paired users on mean-over-masks differences',**paired_bootstrap(v/R,np.zeros(len(v)),stream(97,ci,idx),level=.95)})
        manifest.append({'condition':cond,'repetitions':R,'mask_sha256':digest.hexdigest(),'nominal_density':alpha,'expected_test_density':mean_pi,'epsilon':epsilon,'offset':constant})
        print('Masks complete:',cond,'expected density',round(mean_pi,4),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'reports/tables/mask_repetitions.csv',index=False);summaries(rows,'outcome masking',p).to_csv(ROOT/'reports/tables/mask_results.csv',index=False)
    pd.DataFrame(linear).to_csv(ROOT/'reports/tables/linear_repetitions.csv',index=False);gaps_summary(gaps).to_csv(ROOT/'reports/tables/mask_reversals.csv',index=False)
    out=[]
    for key,g in pd.DataFrame(linear).groupby(['condition','model','target','method'],sort=False):
        ref=float(g.reference.iloc[0]);stats=mc_summary(g.estimate.to_numpy(),ref);is_unclipped=key[3] in ('ht','paired_ht');lo,hi=wilson(int(g.covered.sum()),len(g))
        out.append(dict(zip(['condition','model','target','method'],key))|stats|{'reference':ref,'empirical_variance':float(g.estimate.var(ddof=1)),'mean_estimated_variance':float(g.variance.mean()),'true_variance':float(g.true_variance.mean()),'coverage':float(g.covered.mean()) if is_unclipped else float('nan'),'coverage_lo':lo if is_unclipped else float('nan'),'coverage_hi':hi if is_unclipped else float('nan'),'mean_ess':float(g.ess.mean()),'max_weight':float(g.max_weight.max()),'mean_clipped_fraction':float(g.clipped_fraction.mean()),'zero_variance_frequency':float(g.zero_variance.mean())})
    pd.DataFrame(out).to_csv(ROOT/'reports/tables/linear_correction.csv',index=False);pd.DataFrame(uncertainty).to_csv(ROOT/'reports/tables/mask_user_uncertainty.csv',index=False)
    # The single deterministic full-observation endpoint verifies all methods exactly.
    full=observe(y,np.ones_like(y),valid,stream(7,999));endpoint=[]
    for m in MODELS:
        result=ht(coeff[m],full);ref=float(p[m+'_dcg10'].mean());assert np.isclose(result['estimate'],ref,atol=1e-12) and result['variance']==0
        endpoint.append({'model':m,'dcg10':ref,'ht_dcg10':result['estimate'],'variance':0,'repetitions':1})
    pd.DataFrame(endpoint).to_csv(ROOT/'reports/tables/full_observation_endpoint.csv',index=False)
    write_json(ROOT/'evidence/mask_manifest.json',{'conditions':manifest,'seconds':time.perf_counter()-start,'scores_sha256':sha(ROOT/'artifacts/scores/test.npz'),'design':'independent Bernoulli cells; shared mask across five models; simulator truth never passed to HT'})
    status(['E7'],'13 conditions ×200 masks; observed-only linear HT, clipping/floor and coverage diagnostics; all scores fixed')

def candidate_study():
    start=time.perf_counter();p,base=inputs();valid=p['valid'];y=p['y'];reward=p['r'];orders={m:p[m+'_order'] for m in MODELS};weights=.1+base['item'][p['items']];R=config('experiments')['repetitions']
    rows=[];gaps=[];user_rows=[];manifest=[];pairs=[(a,b) for ai,a in enumerate(MODELS) for b in MODELS[ai+1:]];primary=assert_lock()['contrasts']
    for ci,(sampler,q) in enumerate((sampler,q) for sampler in ('uniform','popularity') for q in (50,100,500)):
        cond=f'{sampler}|q{q}';digest=hashlib.sha256();per_user={tuple(pair):np.zeros(len(y)) for pair in primary}
        for rep in range(R):
            selected=sample_candidates(valid,weights,stream(8,ci,rep),q,sampler);digest.update(np.packbits(selected).tobytes());values={}
            for m in MODELS:
                order=restrict_order(orders[m],selected);mm=metrics(y*selected,reward*selected,order,selected);values[m]=mm['ndcg10']
                for metric in ('ndcg10','recall10','recall20','dcg10','reward10'):
                    rows.append({'condition':cond,'rep':rep,'model':m,'metric':metric,'value':float(mm[metric].mean()),'zero_positive_users':int(mm['zero_positive'].sum()),'q':q,'users':len(y)})
            for a,b in pairs:gaps.append({'condition':cond,'rep':rep,'a':a,'b':b,'reference':float((p[a+'_ndcg10']-p[b+'_ndcg10']).mean()),'gap':float((values[a]-values[b]).mean())})
            for a,b in primary:per_user[(a,b)]+=values[a]-values[b]
        for idx,((a,b),v) in enumerate(per_user.items()):
            user_rows.append({'condition':cond,'a':a,'b':b,'scope':'paired users on mean-over-candidate-draw differences',**paired_bootstrap(v/R,np.zeros(len(v)),stream(98,ci,idx),level=.95)})
        manifest.append({'condition':cond,'repetitions':R,'candidate_sha256':digest.hexdigest(),'min_q_eff':min(q,int(valid.sum(axis=1).min())),'max_q_eff':min(q,int(valid.sum(axis=1).max()))})
        print('Candidates complete:',cond,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'reports/tables/candidate_repetitions.csv',index=False);summaries(rows,'candidate task change',p).to_csv(ROOT/'reports/tables/candidate_results.csv',index=False)
    gaps_summary(gaps).to_csv(ROOT/'reports/tables/candidate_reversals.csv',index=False);pd.DataFrame(user_rows).to_csv(ROOT/'reports/tables/candidate_user_uncertainty.csv',index=False)
    complete=sample_candidates(valid,weights,stream(8,999),10**6,'uniform');assert np.array_equal(complete,valid)
    write_json(ROOT/'evidence/candidate_manifest.json',{'conditions':manifest,'seconds':time.perf_counter()-start,'scores_sha256':sha(ROOT/'artifacts/scores/test.npz'),'numpy':np.__version__,'full_endpoint_repetitions':1,'full_endpoint_exact':True,'law':'Generator.choice without replacement; normalized weights are not marginal inclusion probabilities'})
    (ROOT/'docs/ope_skip.md').write_text('# Optional E10 decision\n\nSkipped. Mandatory paired ranking, missingness and candidate experiments already answer the registered questions. OPE would add a different artificial policy target rather than real historical propensities; preserve focus and resource scope.\n')
    status(['E8','E9'],'Six candidate conditions ×200 shared draws;5000-draw primary/bootstrap user sensitivity; MCSE/Wilson and seed tables')
    record=read_json(ROOT/'evidence/status.json');record['E10']={'status':'skipped (optional)','evidence':'docs/ope_skip.md'};write_json(ROOT/'evidence/status.json',record)
