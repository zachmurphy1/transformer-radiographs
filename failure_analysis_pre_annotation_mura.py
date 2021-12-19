# Standard
import os, sys, shutil, json
import pandas as pd
import numpy as np
import time
import argparse
import itertools
import random
import math
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Stats
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--scratch-dir", default='/export/gaon1/data/zmurphy/transformer-cxr', type=str, help='')
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final/MURA100_final', type=str, help='')
parser.add_argument("--results-table", default='/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final/MURA100_final/results_table.csv', type=str, help='')
parser.add_argument("--to-analyze", default='/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final/to_analyze_MURA100.json', type=str, help='')
args = parser.parse_args()


# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
labels = cfg['labels_mura_standard']
args.scratch_dir = args.scratch_dir.replace('~',os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))
args.results_table = args.results_table.replace('~',os.path.expanduser('~'))
args.to_analyze = args.to_analyze.replace('~',os.path.expanduser('~'))


with open(args.to_analyze, 'r') as f:
  to_analyze = json.load(f)
  
# Read results table
rtable = pd.read_csv(args.results_table)

# Regions
regions = ['shoulder','humerus','elbow', 'forearm','wrist','hand','finger']

name_rename = {'DenseNet121': 'DN', 'DeiT': 'DeiT'}
lists = {}
for f in to_analyze:
    with open(f['file'], 'rb') as fi:
        dat = pickle.load(fi)
     
    # Get data, group by study
    dat = pd.DataFrame().from_dict({'y':[int(x[1]) for x in dat['mura_test']['y']],
                                   'yhat':[x[1] for x in dat['mura_test']['yhat']],
                                   'region':dat['mura_test']['region'],
                                   'study':dat['mura_test']['study']})
    
    # Group by study, take study yhat as mean of image yhats
    dat_grouped = dat.groupby('study').mean()
    dat_region = dat[['study','region']].groupby('study').first()
    dat_grouped['y'] = dat_grouped['y'].astype(int)
    dat_grouped['region'] = dat_region['region']
    
    fns = pd.DataFrame()
    for region_idx, region in enumerate(regions):
      thresh = rtable[(rtable['Model']==f['name'])&(rtable['Measure']=='thresh_{}'.format(region))].iloc[0]['Mean']
      df = dat_grouped[dat_grouped['region']==region].copy()
      df.loc[:,'yhat_pred'] = (df['yhat'] >= thresh).astype(int)
      fn = df[(df['y']==1) & (df['yhat_pred']==0)].reset_index()
      fn['study'] = fn['study'].apply(lambda x: x[x.rfind('/')+1:]).tolist()
      fns = pd.concat([fns,fn], ignore_index=True)
    lists[name_rename[f['name']]] = fns['study'].tolist()

# Combine lists
to_annotate = []
for l in lists.values():
  to_annotate.extend(l)
to_annotate = pd.DataFrame().from_dict({'study':to_annotate}).drop_duplicates()

for k, l in lists.items():
  to_annotate[k] = to_annotate['study'].isin(l)
to_annotate['label'] = np.NaN
to_annotate = to_annotate[['study','label','DeiT', 'DN']].drop_duplicates()
print(to_annotate)
to_annotate.to_csv('/cis/home/zmurphy/code/transformer-radiographs/image_lists/failure_pre_mura.csv', index=False)
