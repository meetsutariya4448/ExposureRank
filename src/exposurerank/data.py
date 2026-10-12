"""Full source audit and compact typed observed pairs. No absent-cell imputation."""
from pathlib import Path
import hashlib, zipfile, shutil, time
import duckdb
import numpy as np
from .io import ROOT, sha, write_json, config, status

ARCHIVE_SHA='c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c'
SMALL_SHA='6b601cd38b2600d8734b4aede309ad73d1e201fdc4bd76d4bc7d2534793d7d15'

def pair_split(users,items):
    values=np.fromiter((int.from_bytes(hashlib.sha256(f'ExposureRank-split-v1|{int(u)}|{int(i)}'.encode()).digest()[:8],'big')/2**64 for u,i in zip(users,items)),dtype=np.float64,count=len(users))
    return np.where(values<.6,0,np.where(values<.8,1,2)).astype(np.int8)

def tie_keys(items):
    return np.array([int.from_bytes(hashlib.sha256(f'ExposureRank-ties-v1|{int(i)}'.encode()).digest()[:8],'big') for i in items],dtype=np.uint64)

def valid(play,duration,ratio):
    return np.isfinite(play)&(play>=0)&np.isfinite(duration)&(duration>0)&np.isfinite(ratio)&(ratio>=0)

def compact_map(raw,known):
    idx=np.searchsorted(known,raw)
    if np.any(idx>=len(known)) or not np.array_equal(known[idx],raw): raise ValueError('Unknown entity ID')
    return idx.astype(np.int32)

def acquire(source=None):
    dest=ROOT/'data/raw/KuaiRec.zip'
    if not dest.exists():
        if source: shutil.copyfile(source,dest)
        else:
            import subprocess
            subprocess.run(['curl','--fail','--location','--retry','3','https://zenodo.org/records/18164998/files/KuaiRec.zip','--output',str(dest)],check=True)
    if sha(dest)!=ARCHIVE_SHA: raise ValueError('Archive differs from frozen primary release')
    members=[]
    with zipfile.ZipFile(dest) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            h=hashlib.sha256()
            with z.open(info) as f:
                while block:=f.read(1<<20): h.update(block)
            members.append({'name':info.filename,'bytes':info.file_size,'sha256':h.hexdigest(),'crc_verified':True})
            if info.filename.endswith(('small_matrix.csv','big_matrix.csv')):
                name=info.filename.rsplit('/',1)[-1]
                target=ROOT/'data/raw'/name
                if not target.exists() or sha(target)!=h.hexdigest():
                    with z.open(info) as src,target.open('wb') as out: shutil.copyfileobj(src,out)
            if info.filename.endswith('/LICENSE'):
                (ROOT/'docs/KuaiRec_embedded_LICENSE.txt').write_bytes(z.read(info))
    write_json(ROOT/'manifests/archive.json',{'sha256':sha(dest),'bytes':dest.stat().st_size,'members':members})

def prepare(source=None):
    start=time.perf_counter(); acquire(source)
    small=ROOT/'data/raw/small_matrix.csv'; assert sha(small)==SMALL_SHA
    con=duckdb.connect(); con.execute("SET memory_limit='4GB'");con.execute('SET threads=4')
    con.execute('CREATE TABLE small AS SELECT * FROM read_csv_auto(?)',[str(small)])
    con.execute('CREATE TABLE big AS SELECT * FROM read_csv_auto(?)',[str(ROOT/'data/raw/big_matrix.csv')])
    audit={}
    for path in sorted((ROOT/'sql').glob('0[1-5]*.sql')):
        cur=con.execute(path.read_text());audit[path.stem]=cur.fetchdf().to_dict('records')
    write_json(ROOT/'manifests/data.json',audit)
    grain=audit['01_grain'][0]; quality=audit['02_duration_ratio'][0]
    assert grain['rows']==4676570 and grain['unique_pairs']==4676570
    assert quality['invalid']==0 and quality['discrepancies']==228
    assert quality['binary_changes']==0 and audit['04_timestamp'][0]['missing_timestamp']==181992
    assert audit['05_overlap'][0]['pair_overlap']==0
    df=con.execute('SELECT user_id,video_id,watch_ratio,play_duration/video_duration recomputed FROM small ORDER BY user_id,video_id').fetchdf()
    raw_u=df.user_id.to_numpy();raw_i=df.video_id.to_numpy();split=pair_split(raw_u,raw_i)
    users=np.unique(raw_u[split==0]);items=np.unique(raw_i[split==0])
    u=compact_map(raw_u,users);i=compact_map(raw_i,items)
    support_u=np.bincount(u[split==0],minlength=len(users));support_i=np.bincount(i[split==0],minlength=len(items))
    assert support_u.min()>=100 and support_i.min()>=20
    ratio=df.watch_ratio.to_numpy(dtype=np.float64);recomputed=df.recomputed.to_numpy(dtype=np.float64)
    y=(ratio>=1).astype(np.float32);r=(np.minimum(ratio,2)/2).astype(np.float32)
    np.savez(ROOT/'data/derived/pairs.npz',u=u,i=i,y=y,r=r,ratio=ratio,recomputed=recomputed,split=split,users=users,items=items,ties=tie_keys(items))
    con.register('pairs', {'user_id':raw_u,'video_id':raw_i,'split':split,'y':y})
    con.execute('COPY pairs TO ? (FORMAT PARQUET)',[str(ROOT/'data/derived/valid_pairs.parquet')])
    for path in sorted((ROOT/'sql').glob('0[6-9]*.sql')):
        audit[path.stem]=con.execute(path.read_text()).fetchdf().to_dict('records')
    write_json(ROOT/'manifests/data.json',audit)
    counts=np.bincount(split,minlength=3).tolist();assert counts==config('splits')['expected_counts']
    catalogs={}
    for name,s in [('validation',1),('test',2)]:
        n=np.bincount(u[split==s],minlength=len(users));assert n.min()>=20
        catalogs[name]={'users':len(users),'min_candidates':int(n.min()),'max_candidates':int(n.max()),'pairs':int(n.sum())}
    write_json(ROOT/'manifests/split.json',{'counts':counts,'users':len(users),'items':len(items),'min_train_user':int(support_u.min()),'min_train_item':int(support_i.min()),'excluded_pairs':0,'pair_file_sha256':sha(ROOT/'data/derived/valid_pairs.parquet')})
    write_json(ROOT/'manifests/catalogs.json',catalogs)
    write_json(ROOT/'evidence/data_runtime.json',{'seconds':time.perf_counter()-start})
    import pandas as pd
    pd.DataFrame({'split':['train','validation','test'],'pairs':counts}).to_csv(ROOT/'reports/tables/split_counts.csv',index=False)
    (ROOT/'reports/data_quality.md').write_text('# Executed complete input audit\n\n'+str(grain)+'\n\n'+str(quality)+'\n\nPublished ratios and untimed rows retained; all small/big overlap checks executed. See manifests/data.json and sql/*.sql.\n')
    status(['E0','E1','E2'],'Protocol b702ee0; complete archive/SQL/split audits; manifests/data.json, split.json, catalogs.json')
    print('Data gates passed:',counts,catalogs,flush=True)

def load_pairs(): return dict(np.load(ROOT/'data/derived/pairs.npz'))

def frame(d,s):
    ix=np.flatnonzero(d['split']==s)
    return {k:d[k][ix] for k in ('u','i','y','r','ratio','recomputed')}
