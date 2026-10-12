import argparse
from exposurerank.simulation import mask_study,candidate_study
p=argparse.ArgumentParser();p.add_argument('--study',choices=['masks','candidates','all'],default='all');a=p.parse_args()
if a.study in ('masks','all'): mask_study()
if a.study in ('candidates','all'): candidate_study()
