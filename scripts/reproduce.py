"""Fresh clone + fresh locked environment + full train/simulation rebuild."""
import argparse,subprocess,time,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from exposurerank.io import ROOT,write_json,read_json,revision

p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args();dest=Path(args.destination)
if dest.exists():raise SystemExit('Reproduction destination must be new; never overwrite existing work')
start=time.perf_counter();source_revision=revision();subprocess.run(['git','clone','--local',str(ROOT),str(dest)],check=True)
with (ROOT/'evidence/reproduction.log').open('w') as log:
    def run(command):
        log.write('\nCOMMAND '+repr(command)+'\n');log.flush();subprocess.run(command,cwd=dest,stdout=log,stderr=subprocess.STDOUT,check=True)
    run(['uv','sync','--frozen','--python','3.12'])
    run(['uv','run','--frozen','python','scripts/run_all.py','--archive',str(ROOT/'data/raw/KuaiRec.zip')])
    run(['uv','run','--frozen','python','scripts/notebooks.py'])
    run(['uv','run','--frozen','python','scripts/audit.py'])
    run(['uv','run','--frozen','python','scripts/leakage_audit.py'])
    run(['uv','run','--frozen','python','scripts/verify_tests.py'])
comparisons=[]
for name in ('split.json','catalogs.json','data.json'):
    assert read_json(ROOT/'manifests'/name)==read_json(dest/'manifests'/name)
    comparisons.append({'path':'manifests/'+name,'result':'exact structural equality'})
first=read_json(ROOT/'configs/selected_models.json');second=read_json(dest/'configs/selected_models.json')
assert first['best_simple']==second['best_simple'] and first['best_learned']==second['best_learned']
for fam in ('mf','neural'):
    for field in ('dimension','l2','validation_ensemble_ndcg10'):assert first[fam][field]==second[fam][field]
    assert [x['best_epoch'] for x in first[fam]['seeds']]==[x['best_epoch'] for x in second[fam]['seeds']]
comparisons.append({'path':'configs/selected_models.json','result':'exact choices,epochs,validation metric; runtime/provenance intentionally differ'})
for rel in ('data/derived/pairs.npz','artifacts/scores/test.npz','artifacts/scores/reference.npz'):
    one=dict(np.load(ROOT/rel));two=dict(np.load(dest/rel));assert one.keys()==two.keys()
    assert all(np.array_equal(one[k],two[k]) for k in one)
    comparisons.append({'path':rel,'result':'all arrays byte-exact values; container timestamps ignored'})
for path in sorted((ROOT/'artifacts/simulations').glob('*_user_differences_*.npz')):
    other=dest/'artifacts/simulations'/path.name
    one=dict(np.load(path));two=dict(np.load(other))
    assert one.keys()==two.keys() and all(np.array_equal(one[k],two[k]) for k in one)
    comparisons.append({'path':'artifacts/simulations/'+path.name,'result':'exact private per-user/per-repetition differences and contrast identities'})
for path in sorted((ROOT/'reports/tables').glob('*.csv')):
    if path.name=='training_runs.csv':continue
    a=pd.read_csv(path);b=pd.read_csv(dest/'reports/tables'/path.name);assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist()
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]):assert np.allclose(a[col],b[col],atol=1e-7,rtol=1e-5,equal_nan=True)
        else:assert a[col].fillna('NA').equals(b[col].fillna('NA'))
    comparisons.append({'path':'reports/tables/'+path.name,'result':'exact' if path.read_bytes()==(dest/'reports/tables'/path.name).read_bytes() else 'numeric atol1e-7 rtol1e-5; identities exact'})
for name,key in [('mask_manifest.json','mask_sha256'),('candidate_manifest.json','candidate_sha256')]:
    a=read_json(ROOT/'evidence'/name)['conditions'];b=read_json(dest/'evidence'/name)['conditions'];assert [x[key] for x in a]==[x[key] for x in b]
    comparisons.append({'path':'evidence/'+name,'result':'every realized draw hash exact'})
for name in ('mask_reversals.csv','candidate_reversals.csv'):
    a=pd.read_csv(ROOT/'reports/tables'/name);b=pd.read_csv(dest/'reports/tables'/name)
    assert a[['state','reversals']].equals(b[['state','reversals']])
from exposurerank.io import sha
write_json(ROOT/'evidence/reproduction.json',{'status':'passed','source_revision':source_revision,'fresh_environment':True,'complete_training_and_simulation_rebuilt':True,'seconds':time.perf_counter()-start,'compared_outputs':comparisons,'selection_and_reversal_categories_exact':True,'all_private_score_arrays_exact':True,'statistical_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))},'public_figures':'source tables checked; PNG metadata bytes not required identical','log':'evidence/reproduction.log'})
print('Complete fresh-environment reconstruction passed:',len(comparisons),'outputs checked')
