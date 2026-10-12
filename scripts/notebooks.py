"""Execute two safe notebooks in the locked project kernel; outputs aggregate/synthetic."""
import os
from pathlib import Path
import subprocess,sys
import nbformat as nbf
from nbclient import NotebookClient
from exposurerank.io import ROOT,write_json,sha

subprocess.run([sys.executable,'-m','ipykernel','install','--prefix',str(ROOT/'.venv'),'--name','exposurerank','--display-name','ExposureRank locked Python'],check=True,capture_output=True)
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')

cells1=[
 nbf.v4.new_markdown_cell('# ExposureRank: the evaluation-design decision\nHistorical2020 observed feedback; static held-out cells. Masks and candidate sampling are explicitly simulated. These are aggregate executed results; no live engagement or causal conclusion.\n\nKuaiRec/Gao et al., DOI10.5281/zenodo.18164998. Conservative CC BY-SA4.0 treatment for dataset-derived notebook outputs.'),
 nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = Path.cwd().parent\nfull = pd.read_csv(root / 'reports/tables/full_reference.csv')\nfull[['model','ndcg10','recall10','catalog_coverage']]"),
 nbf.v4.new_markdown_cell('The strongest simple comparator was chosen on validation. The registered MF advantage is small; paired intervals describe empirical user sensitivity with fixed items, rather than uncertainty in a known finite-panel mean.'),
 nbf.v4.new_code_cell("primary = pd.read_csv(root / 'reports/tables/primary_contrasts.csv')\nassert (primary['lo'] <= primary['gap']).all() and (primary['gap'] <= primary['hi']).all()\nprimary[['a','b','gap','lo','hi','level','draws']]"),
 nbf.v4.new_code_cell("masks = pd.read_csv(root / 'reports/tables/mask_reversals.csv')\ncandidates = pd.read_csv(root / 'reports/tables/candidate_reversals.csv')\nselection = masks[(masks.a == 'item') & (masks.b == 'mf')]\nselection[['condition','reference_gap','mean','rate','lo','hi']]"),
 nbf.v4.new_markdown_cell('Mask experiments hold ranks and candidates fixed. Candidate experiments keep selected outcomes complete and change the task denominator. Each condition uses200 shared independent draws; masks are not200 new user populations. Near ties are distinguished from separated reference comparisons.'),
 nbf.v4.new_code_cell("candidates[(candidates.a == 'item') & (candidates.b == 'mf')][['condition','mean','rate','lo','hi']]"),
 nbf.v4.new_code_cell("correction = pd.read_csv(root / 'reports/tables/linear_correction.csv')\ncorrection[(correction.model == 'mf') & (correction.target == 'dcg10') & (correction.method == 'ht')][['condition','bias','rmse','coverage','coverage_lo','coverage_hi','max_weight']]"),
 nbf.v4.new_markdown_cell('Linear HT estimates matching DCG/reward targets. It does not automatically correct unseen ideal denominators in nDCG or Recall. Weight clipping introduces bias. No trained assets or real-user outputs belong in this notebook.')
]
cells2=[
 nbf.v4.new_markdown_cell('# Ranking and weighting walkthrough — SYNTHETIC ONLY\nEvery interaction in this notebook is an invented tiny mathematical fixture, not a KuaiRec user. Functions are the actual project implementation.'),
 nbf.v4.new_code_cell("import numpy as np\nfrom exposurerank.data import tie_keys\nfrom exposurerank.ranking import Catalog, restrict_order\nfrom exposurerank.metrics import metrics, linear_coefficients\nfrom exposurerank.weighting import ObservedView, ht\nfrom exposurerank.masks import observe, naive_zero\nfrom exposurerank.uncertainty import paired_bootstrap\ny = np.array([1.,0.,1.,0.])\nf = {'u':np.zeros(4,dtype=int),'i':np.arange(4)}\ncatalog = Catalog.build(f,tie_keys(np.arange(4)),1)\norder = catalog.order(np.array([.9,.8,.7,.6]))\ntruth = metrics(y[None,:],y[None,:],order,catalog.valid)\n{k: float(v[0]) for k,v in truth.items()}"),
 nbf.v4.new_code_cell("observed = ObservedView(np.array([0,1]), np.array([1.,0.]), np.array([.5,.5]), 4)\ntry:\n    observed.lookup(2)\n    raise AssertionError('Hidden outcome access was allowed')\nexcept KeyError:\n    print('Hidden outcome lookup correctly denied')\ncoefficients = linear_coefficients(order,catalog.valid)\nht(coefficients,observed)"),
 nbf.v4.new_code_cell("import itertools\npi=np.array([.2,.4,.6,.8]);a=np.array([.4,.3,-.2,.1])\nexpected=var_observable=var_actual=0.\nreference=float(a@y)\nfor bits in itertools.product([False,True], repeat=4):\n    mask=np.array(bits);prob=np.prod(np.where(mask,pi,1-pi));ix=np.flatnonzero(mask)\n    estimate=ht(a,ObservedView(ix,y[ix],pi[ix],4))\n    expected += prob*estimate['estimate']\n    var_actual += prob*(estimate['estimate']-reference)**2\n    var_observable += prob*estimate['variance']\nassert np.isclose(expected,reference) and np.isclose(var_actual,var_observable)\n{'truth':reference,'expected_HT':expected,'true_variance':var_actual,'expected_observable_variance':var_observable}"),
 nbf.v4.new_markdown_cell('Unbiased linear numerator estimation does not imply an unbiased ratio. With a single relevant item, missing that item can make masked IDCG zero. Outcome-dependent full probability vectors can reveal hidden labels; only observed probabilities are passed to HT.'),
 nbf.v4.new_code_cell("# Synthetic candidate reversal: same two rankers, smaller universe.\ny=np.array([[1.,0.,0.,1.]])\na_order=np.array([[0,1,2,3]]);b_order=np.array([[1,3,0,2]])\nfull_a=metrics(y,y,a_order,np.ones_like(y,dtype=bool))['ndcg10'][0]\nfull_b=metrics(y,y,b_order,np.ones_like(y,dtype=bool))['ndcg10'][0]\nsubset=np.array([[False,True,True,True]])\nsub_a=metrics(y*subset,y*subset,restrict_order(a_order,subset),subset)['ndcg10'][0]\nsub_b=metrics(y*subset,y*subset,restrict_order(b_order,subset),subset)['ndcg10'][0]\nassert full_a>full_b and sub_a<sub_b\n{'full_A_minus_B':full_a-full_b,'sampled_A_minus_B':sub_a-sub_b}"),
 nbf.v4.new_code_cell("paired_bootstrap(np.array([.7,.5,.6]),np.array([.6,.55,.6]),np.random.default_rng(17),draws=5000)"),
 nbf.v4.new_markdown_cell('The invented reversal demonstrates a mechanism; the benchmark evidence is in notebook01 and executed tables. Paired user resampling, seed variability and simulator Monte Carlo uncertainty address different questions.')
]
evidence=[]
for name,cells in [('01_evaluation_design_decision',cells1),('02_ranking_and_weighting_walkthrough',cells2)]:
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'exposurerank','display_name':'ExposureRank locked Python','language':'python'}})
    NotebookClient(nb,timeout=180,kernel_name='exposurerank',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
    path=ROOT/'notebooks'/f'{name}.ipynb';nbf.write(nb,path)
    assert all(output.output_type!='error' for cell in nb.cells if cell.cell_type=='code' for output in cell.get('outputs',[]))
    evidence.append({'notebook':str(path.relative_to(ROOT)),'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'sha256':sha(path),'status':'executed without errors','privacy':'aggregate historical / synthetic only'})
write_json(ROOT/'evidence/notebooks.json',evidence)
print(evidence)
