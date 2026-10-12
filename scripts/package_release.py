"""Rights-aware, local-only release inventory; raw and learned assets excluded."""
import json,subprocess,zipfile,hashlib
from pathlib import Path
from exposurerank.io import ROOT,read_json,write_json,sha,revision,status

assert read_json(ROOT/'evidence/reproduction.json')['status']=='passed'
assert read_json(ROOT/'evidence/claim_audit.json')['status']=='passed'
assert read_json(ROOT/'evidence/leakage_audit.json')['status']=='passed'
assert len(read_json(ROOT/'evidence/notebooks.json'))==2
assert read_json(ROOT/'evidence/tests.json')['failures']==0
status(['E12','E13'],'Full fresh-environment training/simulation rebuild passed; tests, claims, two executed notebooks,40 interview questions and three honest bullet candidates')
(ROOT/'evidence/publication_review.md').write_text('# Publication review\n\nLocal preparation only. Archive inventory excludes raw/derived pairs, real maps, per-user outcomes/metrics, ranks, scores, embeddings/checkpoints, credentials and environments. Data-derived reports/tables/figures/notebook benchmark outputs carry conservative CC BY-SA4.0 attribution; independently authored code MIT. KuaiRec metadata/embedded license conflict remains unresolved. No raw/model rights cleared and no public GitHub or hosting published. Expanded release requires a separate decision. No first-four-project or master resume edits.\n')
allowed_roots={'src','scripts','tests','sql','configs','manifests','reports','docs','notebooks','public','evidence'}
top={'README.md','DELIVERABLES.md','IMPLEMENTATION_STATUS.md','DATA_NOTICE.md','LICENSE','CITATION.cff','pyproject.toml','uv.lock','.gitignore'}
files=[]
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if not p.is_file() or '__pycache__' in rel.parts:continue
    if p.name=='release_inventory.json':continue
    if len(rel.parts)==1 and p.name in top:files.append(p)
    elif rel.parts[0] in allowed_roots:
        if p.suffix in {'.pt','.npz','.npy','.parquet','.duckdb','.csv.gz'}:raise AssertionError('Private asset entered public subtree: '+str(rel))
        if 'per_user' in p.name or 'rebuild' in rel.parts or p.name=='reproduction.log':continue
        if p.suffix in {'.py','.sql','.json','.csv','.md','.txt','.toml','.lock','.ipynb','.png','.svg','.log'}:files.append(p)
files=sorted(files);inventory=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
write_json(ROOT/'evidence/release_inventory.json',{'source_revision':revision(),'files':inventory,'excluded':'all data/artifacts/environments, real user-level outputs, reproduction temp tree','publication':'local ZIP only'})
files.append(ROOT/'evidence/release_inventory.json')
archive=ROOT/'dist/ExposureRank_v1.0.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,'ExposureRank/'+str(p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as z:
    assert not any('/data/' in n or '/artifacts/' in n or n.endswith(('.pt','.npz','.parquet')) for n in z.namelist());assert z.testzip() is None
(ROOT/'dist/SHA256SUMS.txt').write_text(sha(archive)+'  ExposureRank_v1.0.zip\n')
print('Local release:',len(files),'files',archive.stat().st_size,'bytes; private assets excluded')
