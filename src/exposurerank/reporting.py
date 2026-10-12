"""Reports and standalone figures generated exclusively from executed tables."""
import json,platform,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import ROOT,read_json,write_json,sha,revision,status

TABLE=ROOT/'reports/tables'

def table(name):return pd.read_csv(TABLE/(name+'.csv'))
def emit(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'reports/figures'/f'{name}.png',dpi=160);fig.savefig(ROOT/'reports/figures'/f'{name}.svg');plt.close(fig)

def reports():
    (ROOT/'reports/figures').mkdir(parents=True,exist_ok=True)
    full=table('full_reference');contrasts=table('primary_contrasts');mask=table('mask_results');cand=table('candidate_results');linear=table('linear_correction');mr=table('mask_reversals');cr=table('candidate_reversals');runs=read_json(ROOT/'evidence/runs.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,2,figsize=(10,4));split=read_json(ROOT/'manifests/split.json');axes[0].bar(['Train','Validation','Test'],split['counts'],color=['#477b9b','#dc974b','#4a9678']);axes[0].set_ylabel('Observed pairs');axes[0].set_title('Static hash split — observed cells only')
    axes[1].bar(['Observed','Absent'],[4676570,17827],color=['#477b9b','#c65e53']);axes[1].set_yscale('log');axes[1].set_ylabel('User-video cells (log scale)');axes[1].set_title('Small matrix: 99.620% observed');emit(fig,'01_coverage_and_splits')
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(full.model,full.ndcg10,color=['#a4a4a4','#477b9b','#789db2','#4a9678','#dc974b']);axes[0].set_ylim(0,1);axes[0].set_ylabel('Macro nDCG@10');axes[0].set_title('1,411 users; 595–743 held-out candidates')
    gap=contrasts.gap.to_numpy();labels=contrasts.a+' − '+contrasts.b;axes[1].errorbar(gap,np.arange(len(gap)),xerr=np.stack([gap-contrasts.lo.to_numpy(),contrasts.hi.to_numpy()-gap]),fmt='o',color='#477b9b',capsize=4);axes[1].axvline(0,color='black',lw=.8);axes[1].set_yticks(np.arange(len(gap)),labels);axes[1].set_xlabel('Paired nDCG@10 difference');axes[1].set_title('97.5% empirical paired-user intervals');emit(fig,'02_model_comparison')
    fig,axes=plt.subplots(1,2,figsize=(11,4));m=mask[(mask.model=='mf')&(mask.metric=='ndcg10')&mask.condition.str.endswith('floor0.02')]
    for mech in ('uniform','popularity','outcome'):
        sub=m[m.condition.str.startswith(mech)].copy();sub['alpha']=sub.condition.str.split('|').str[1].astype(float);axes[0].plot(sub.alpha,sub.bias,'o',label=mech)
    axes[0].set_xlabel('Nominal observation probability');axes[0].set_ylabel('Naive nDCG bias versus fixed reference');axes[0].legend();axes[0].set_title('Hidden outcomes zeroed; ranks unchanged')
    sub=linear[(linear.model=='mf')&(linear.target=='dcg10')&(linear.method=='ht')&linear.condition.str.endswith('floor0.02')]
    axes[1].barh(sub.condition,sub.rmse,color='#477b9b');axes[1].set_xlabel('HT linear DCG RMSE (200 masks/cell)');axes[1].set_title('Matching linear target; no nDCG correction');emit(fig,'03_observation_distortion')
    fig,axes=plt.subplots(1,2,figsize=(12,5));m=mr[(mr.a=='item')&(mr.b=='mf')&mr.condition.str.endswith('floor0.02')];c=cr[(cr.a=='item')&(cr.b=='mf')]
    for ax,data,title in [(axes[0],m,'Outcome masks — candidates fixed'),(axes[1],c,'Sampled candidates — outcomes complete')]:
        good=data.dropna(subset=['rate']);v=good.rate.to_numpy();ax.errorbar(v,np.arange(len(good)),xerr=np.stack([v-good.lo.to_numpy(),good.hi.to_numpy()-v]),fmt='o',capsize=3);ax.set_yticks(np.arange(len(good)),good.condition);ax.set_xlim(-.02,1.02);ax.set_xlabel('MF/item reversal frequency; Wilson 95% MC interval');ax.set_title(title)
    emit(fig,'04_ranking_reversal_frequency')
    fig,axes=plt.subplots(1,2,figsize=(11,4));gap=cr[(cr.a=='item')&(cr.b=='mf')]
    for sampler in ('uniform','popularity'):
        sub=gap[gap.condition.str.startswith(sampler)].copy();sub['q']=sub.condition.str.split('q').str[-1].astype(int);axes[0].plot(sub.q,-sub['mean'],'o',label=sampler)
    mf_minus_item=float(contrasts.iloc[0].gap);axes[0].axhline(mf_minus_item,color='black',ls='--',label='Full reference');axes[0].set_xlabel('Candidate subset size');axes[0].set_ylabel('MF − item nDCG difference within each task');axes[0].legend();axes[0].set_title('200 common draws per sampler/size')
    c=cand[(cand.metric=='ndcg10')&(cand.model.isin(['item','mf','neural']))]
    for model in ('item','mf','neural'):
        ss=c[(c.model==model)&c.condition.str.startswith('uniform')].copy();ss['q']=ss.condition.str.split('q').str[-1].astype(int);axes[1].plot(ss.q,ss['mean'],'o',label=model)
    axes[1].set_ylabel('Sample-task nDCG@10');axes[1].set_xlabel('Uniform sampled candidates');axes[1].legend();axes[1].set_title('Different denominators; not full-catalog accuracy');emit(fig,'05_candidate_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(11,5));ss=linear[(linear.model=='mf')&(linear.target=='dcg10')&(linear.method=='ht')&linear.condition.str.endswith('floor0.02')];v=ss.coverage.to_numpy();axes[0].errorbar(v,np.arange(len(ss)),xerr=np.stack([v-ss.coverage_lo.to_numpy(),ss.coverage_hi.to_numpy()-v]),fmt='o',capsize=3);axes[0].set_yticks(np.arange(len(ss)),ss.condition);axes[0].axvline(.95,color='black',ls='--');axes[0].set_xlim(0,1.01);axes[0].set_xlabel('Empirical normal-interval coverage; Wilson MC bounds');axes[0].set_title('200 masks; nominal 95% is not guaranteed')
    axes[1].bar(full.model,full.catalog_coverage*100,color='#477b9b');axes[1].set_ylabel('Distinct top10 items / observed test item union (%)');axes[1].set_title('Recommendation catalog coverage ≠ CI coverage');emit(fig,'06_reliability_and_concentration')

    headers=['model','ndcg10','recall10','recall20','catalog_coverage'];comparison=full[headers].to_csv(index=False)
    ctext='\n'.join(f'- {x.a} minus {x.b}: {x.gap:.6f}, 97.5% interval [{x.lo:.6f}, {x.hi:.6f}].' for x in contrasts.itertuples())
    reversal_candidates=cr[(cr.state=='reference separated')&(cr.reversals>0)];reversal_masks=mr[(mr.state=='reference separated')&(mr.reversals>0)]
    primary_mask=mr[(mr.a=='item')&(mr.b=='mf')&mr.condition.str.endswith('floor0.02')];primary_cand=cr[(cr.a=='item')&(cr.b=='mf')]
    training_seconds=sum(x['seconds'] for x in runs);peak=max(x['peak_process_rss_gib'] for x in runs)
    methods='''The small matrix contains 4,676,570 unique observed pairs, 1,411 users and 3,327 videos; 17,827 absent cells are excluded. Published finite watch ratios are retained, including 228 minor duration-ratio discrepancies; none alter tested binary thresholds. 181,992 rows lack clocks. Training uses 2,806,319 pairs; validation934,854; test935,397. Test catalogs595–743/user are observed held-out pairs, not all global items. Primary labels are watch_ratio>=1, a completion/repeated-play proxy. Secondary reward min(ratio,2)/2 is bounded.

Training-only priors and additive logistic biases challenge actual CPU PyTorch MF and a small [p,q,p*q] neural interaction model. Twelve configurations use d16/32 and L2 1e-5/1e-4/1e-3, BCE logits, Adam .001, explicit embedding/bias L2, no optimizer decay, max15 epochs and validation early stopping. Selected families use seeds17/29/43, averaged probabilities. Configuration and family selection use only full validation nDCG. Models/samplers/contrasts were locked and committed before test.

All users contribute to macro metrics, including zero-positive users as zero (none in the full binary test). Ties use a label-independent item hash. Two prespecified paired-user comparisons use5000 resamples and97.5% percentile intervals; approximate family95% under empirical exchangeability with fixed items. Selected users/items are not representative platform probability samples. These bounds are sensitivity to user composition, not uncertainty in the known frozen-panel mean.

Outcome masks keep scores/candidates/ranks fixed: independent uniform, popularity and outcome-dependent probabilities at.10/.25/.50, epsilon.02, beta2, offsets fitted on validation. Four additional biased-mask floor.10 conditions at.25/.50 are secondary. All13 conditions run200 masks and share each mask across models. The HT interface accepts only observed values and their probabilities. Known probabilities are synthetic, not recovered historical exposure. HT corrects linear DCG/top10 bounded reward; masked nDCG/Recall are intentionally naive diagnostics, not unbiased corrections. Clipping10/50 changes the target expectation. Normal HT intervals are approximate, assessed by empirical coverage/Wilson MC bounds.

Candidate experiments independently draw50/100/500 observed held-out candidates per user under uniform or training-weighted NumPy choice without replacement,200 common draws per cell. Selected outcomes are complete. No positive is forced. Cached rank restriction is exact; denominators are rebuilt on the selected catalog. Absolute sampled nDCG is a different task. Reference pair gaps with magnitude<=.001 are near ties; otherwise sign changes define reversals. Repetitions are conditional simulation draws, not new user populations. MCSE and Wilson uncertainty are separate from paired-user/bootstrap and training-seed variation.'''
    result=f'''# ExposureRank technical report

## Decision and findings

There is no clear evidence of a learned-model advantage over the strongest simple baseline on this fixed test. The comparison intervals include zero. This supports retaining the item baseline as a serious comparator, rather than assuming extra parameters improve the decision. It does not establish equivalence or an online effect.

```csv
{comparison}```

{ctext}

MF/item ordering reversed in {int(primary_mask.reversals.sum())} of {len(primary_mask)*200} primary mechanism/density mask repetitions combined, and {int(primary_cand.reversals.sum())} of {len(primary_cand)*200} candidate repetitions combined. Conditions have different laws, so these totals are inventory summaries rather than pooled probability estimates; per-condition rates and Wilson bounds are in the reversal tables. The reference MF/item gap is only slightly above the prespecified .001 near-tie cutoff. Other near-tied model pairs remain explicitly classified.

## Dataset, methods and inference

{methods}

## Linear correction and failure diagnostics

See linear_correction.csv for matching-target bias/RMSE, empirical versus analytic/estimated variance, ESS/max weights, clipping fraction and empirical coverage. Do not compare HT DCG error to naive nDCG error as if they shared a target. Estimates may exceed the reference reward domain; they are not clipped. Low observed weights/support may create inaccurate normal intervals; measured coverage is reported without a blanket nominal guarantee. The deterministic full-observation endpoint has exactly zero mask variance.

## Resources and reproducibility

Sixteen actual learned-model runs (12 configurations plus four additional selected-seed runs) took {training_seconds:.2f} measured training seconds in total. Maximum recorded process RSS was {peak:.3f} GiB on {platform.platform()}, PyTorch CPU; MPS was not used. Setup, source audit, baseline optimization, simulations, report generation and clean reconstruction are separate timings. See evidence/mf_preflight.json, runs.json and stage manifests. Final clean reproduction status is in evidence/reproduction.json; it must pass before the project is declared complete.

## Limits and publication

Historical2020 static observed-cell benchmark; no future session, satisfaction, causal engagement, real historical propensities or production lift. Data selection/blocked missing cells and shared items limit generalization. Three seeds are a small fitting sensitivity, not a complete training uncertainty distribution. Macro top10 preferences concentrate on a small item set. Coverage of recommended items and statistical intervals are different quantities. Numerical watch-ratio discrepancies have unknown cause; recomputed/capped/threshold sensitivities are reported without test-driven target changes.

KuaiRec authors, https://kuairec.com/, CIKM2022, DOI10.5281/zenodo.18164998. Metadata CC BY versus embedded CC BY-SA conflict is unresolved. Conservatively attribute dataset-derived reports/tables/figures under CC BY-SA4.0, indicate transformations, and exclude raw logs, maps, per-user metrics, scores, embeddings and checkpoints. Original code MIT. This is local preparation; no public publication or rights clearance for model redistribution.
'''
    (ROOT/'reports/technical_report.md').write_text(result)
    (ROOT/'reports/decision_memo.md').write_text('# Decision memo\n\nKeep the smoothed item baseline as a serious default comparator: the registered MF advantage is small and inconclusive. Compare models on exactly the same user/candidate population and retain missing feedback as unobserved. Refuse a strong conclusion when superiority depends on the sampling denominator, outcome-observation mechanism, a near tie, or uncertain heavy-weight correction.\n\n'+ctext+'\n\nThe outcome/candidate experiments measure evaluation fragility while trained scores remain fixed. Report each condition with MC bounds; do not choose a spectacular sampled winner. Known-probability HT helps estimate matching linear rewards only under this explicit simulator design. It does not repair arbitrary normalized ranking metrics or identify a causal online improvement. An online test would require a new randomized-exposure design and real operational outcomes.\n\nAttribution/license boundaries: DATA_NOTICE.md.\n')
    (ROOT/'reports/evaluation_card.md').write_text('# Evaluation card\n\n'+methods+'\n\nPrimary result intervals:\n'+ctext+'\n\nNo test-time refitting/reselection. Optional OPE and forced-positive illustration are skipped; see docs/ope_skip.md. Public artifacts are aggregate-only with source attribution.\n')
    (ROOT/'reports/model_cards.md').write_text('# Executed model cards\n\nThree simple train-only models and sixteen learned runs. Twelve declared configurations; selected three seeds/family. Frozen selection in configs/selected_models.json, every epoch/config/run in evidence/runs.json and training_runs.csv. Neural interaction layers are a specified modest variant, not an exact published NCF implementation.\n\n'+full.to_csv(index=False)+'\n\n'+ctext+'\n\nActual objective, exclusions and validation-only selection: docs/analysis_plan.md and src/exposurerank/training.py.\n')
    (ROOT/'reports/limitations.md').write_text(result.split('## Limits and publication')[1])
    (ROOT/'docs/polish_decisions.md').write_text('# Optional scope decisions\n\nExecuted: selected three seeds, threshold/cap and recomputed reward diagnostics, four positivity-floor conditions, clipping10/50, separate paired-user simulator sensitivity, two notebook walkthroughs (execution evidence required). Skipped: OPE, forced-positive conditional candidates, two-level user/repetition bootstrap and MPS. These are optional and do not replace mandatory experiments. A shared Portfolio edit is not authorized; no existing project/page was changed. No standalone live recommendation app is needed.\n')
    ledger=[]
    for row in full.to_dict('records'):
        for field in ('ndcg10','recall10','recall20','catalog_coverage'):
            ledger.append({'id':row['model']+'_'+field,'table':'reports/tables/full_reference.csv','key':{'model':row['model']},'field':field,'value':row[field],'unit':'fraction','revision':revision(),'data_manifest':'manifests/data.json','evaluation_lock':'manifests/evaluation_lock.json'})
    for idx,row in contrasts.iterrows():
        for field in ('gap','lo','hi'):ledger.append({'id':f'contrast{idx}_{field}','table':'reports/tables/primary_contrasts.csv','key':{'a':row.a,'b':row.b},'field':field,'value':row[field],'unit':'nDCG difference','revision':revision()})
    write_json(ROOT/'evidence/claim_ledger.json',ledger)
    payload={'disclosure':'Historical static held-out observed feedback. Masks/candidates explicitly simulated. No live recommendation/online effect.','attribution':'KuaiRec/Gao et al.; DOI10.5281/zenodo.18164998; conservative CC BY-SA4.0 aggregate treatment.','full_reference':full.to_dict('records'),'primary_contrasts':contrasts.to_dict('records'),'mask_results':mask.to_dict('records'),'candidate_results':cand.to_dict('records')}
    # Strict JSON, no NaN sentinel in a browser payload.
    write_json(ROOT/'public/aggregate_payload.json',payload)
    write_json(ROOT/'public/asset_manifest.json',{'aggregate_payload_sha256':sha(ROOT/'public/aggregate_payload.json'),'code_revision':revision(),'rows_only':'model/condition aggregates; no individual IDs/maps/scores'})
    (ROOT/'docs/resume_candidates.md').write_text(f'''# Evidence-backed resume candidates

- Trained PyTorch matrix-factorization and neural recommendation models on 2.81M observed interactions, comparing three-seed ensembles with three baselines across 1,411 users; found the MF advantage over the strongest baseline inconclusive under paired 97.5% intervals.
- Measured offline recommendation sensitivity across 19 synthetic observation/candidate conditions with 200 shared repetitions each, separating feedback missingness from candidate-set changes while holding trained model scores fixed.
- Implemented observed-only Horvitz–Thompson estimation for linear ranking rewards, validated expectation/variance by exhaustive toy enumeration, and reported empirical interval coverage and weight-clipping bias on the benchmark.

Source: reports/tables/*.csv and evidence/claim_ledger.json. Do not claim engagement lift, unbiased normalized nDCG, production deployment or public URLs. Final test/reproduction counts belong here only after verified evidence exists. Master resume was not edited.
''')
    (ROOT/'docs/project_walkthrough.md').write_text('# One-minute explanation\n\nI trained small recommendation models on historical observed video feedback and then froze them. I asked whether changing which outcomes are visible or which candidates are evaluated can alter the apparent winner. The full test gave MF nDCG0.8675 versus item rates0.8665, with an inconclusive paired interval. I separately ran13 mask conditions and6 candidate tasks,200 draws each, and checked linear inverse-probability correction. The benchmark is static and selected; completion is not satisfaction and simulated probabilities are not real exposure.\n\n# Five-minute walkthrough\n\n1. Define source/license and the held-out observed catalog.\n2. Explain pair hash, train-only baseline rates, MF BCE update and neural comparison.\n3. Show full_reference.csv and the two paired intervals.\n4. Compare one mask condition with one independently sampled candidate task; explain HT linear variance and the denied hidden-outcome lookup.\n5. Reproduce a headline from its table, discuss concentration/near ties and the clean rebuild evidence.\n\n# Deep dive\n\nFollow data.py → training.py → committed evaluation_lock → evaluation.py → simulation.py. Derive sigmoid(z)-y for one MF update, compute toy nDCG by hand, and enumerate HT masks in tests/test_statistics.py. A known observation probability cancels only a linear numerator; unseen IDCG is not known. Show user bootstrap, three seeds and conditional simulator randomness as separate evidence layers.\n')
    status(['E11'],'Six measured SVG/PNG figures, technical report/memo/cards, aggregate payload and claim ledger generated')
    print('Reports and six figures generated from executed tables.',flush=True)
