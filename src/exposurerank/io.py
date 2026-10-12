from pathlib import Path
import hashlib, json, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')

def read_json(path): return json.loads(Path(path).read_text())
def config(name): return read_json(ROOT / 'configs' / (name+'.json'))
def revision(): return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
def stream(*keys): return np.random.default_rng(np.random.SeedSequence([config('experiments')['master_seed'], *keys]))

def status(gates, note):
    path=ROOT/'evidence/status.json'
    record=read_json(path) if path.exists() else {}
    for gate in gates: record[gate]={'status':'passed','evidence':note,'revision':revision()}
    write_json(path,record)
    rows='\n'.join(f'| E{i} | {record.get("E"+str(i),{}).get("status","not started")} | {record.get("E"+str(i),{}).get("evidence","")} |' for i in range(14))
    (ROOT/'IMPLEMENTATION_STATUS.md').write_text('# ExposureRank implementation status\n\nLocal implementation only. No public publishing, paid cloud, other-project edits or master resume edits.\n\n| Gate | Status | Evidence |\n|---|---|---|\n'+rows+'\n\nSee evidence/status.json for actual source revisions and stage logs for failures/runtimes.\n')
