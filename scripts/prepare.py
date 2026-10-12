import argparse
from exposurerank.data import prepare
p=argparse.ArgumentParser();p.add_argument('--archive');a=p.parse_args();prepare(a.archive)
