"""Dependency-ordered local workflow. Refuse accidental refitting after evaluation."""
import argparse,subprocess,sys,os
from pathlib import Path
from exposurerank.io import ROOT
os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--archive');a=p.parse_args()
if Path('artifacts/scores/test.npz').exists():
    raise SystemExit('Test already evaluated. Use a fresh clone for reproduction; do not refit this locked run.')
if not (ROOT/'.git').exists():
    subprocess.run(['git','init'],check=True)
    subprocess.run(['git','add','.'],check=True)
    subprocess.run(['git','commit','-m','Freeze supplied reproduction protocol before fitting'],check=True)
def run(*args):subprocess.run([sys.executable,*args],check=True)
run('scripts/prepare.py',*(['--archive',a.archive] if a.archive else []))
run('-m','pytest');run('scripts/train.py');run('scripts/freeze.py')
subprocess.run(['git','add','.'],check=True);subprocess.run(['git','commit','--allow-empty','-m','Lock models before test in this fresh run'],check=True)
run('scripts/evaluate.py');run('scripts/simulate.py');run('scripts/report.py')
