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
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final/CXR100_final', type=str, help='')
parser.add_argument("--results-table", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final/eval_results/results_table.csv', type=str, help='')
parser.add_argument("--to-analyze", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final/to_analyze_CXR100.json', type=str, help='')
args = parser.parse_args()


# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
labels = cfg['labels_chexnet_14_standard']
args.scratch_dir = args.scratch_dir.replace('~',os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))
args.results_table = args.results_table.replace('~',os.path.expanduser('~'))
args.to_analyze = args.to_analyze.replace('~',os.path.expanduser('~'))

with open(args.to_analyze, 'r') as f:
  to_analyze = json.load(f)
  
# Read results table
rtable = pd.read_csv(args.results_table)
  
# Set label
label = 'Pneumothorax'
label_idx = labels.index(label)

# Get FP and FN lists
name_rename = {'DenseNet121': 'DN', 'DeiT': 'DeiT'}
lists = {}
for f in to_analyze:
  with open(f['file'], 'rb') as fi:
    dat = pickle.load(fi)
  
  # Get confusion matrix
  thresh = rtable[(rtable['Model']==f['name'])&(rtable['Set']=='nihcxr14_test')&(rtable['Measure']=='thresh_{}'.format(label))].iloc[0]['Mean']
  df = pd.DataFrame().from_dict({'y':[x[label_idx] for x in dat['nihcxr14_test']['y']], 
                                   'yhat':[x[label_idx] for x in dat['nihcxr14_test']['yhat']],
                                   'file':dat['nihcxr14_test']['file']})
  df['yhat_pred'] = (df['yhat'] >= thresh).astype(int)
  
  # FP
  lists['FP {}'.format(name_rename[f['name']])] = df[(df['y']==0) & (df['yhat_pred']==1)]['file'].sample(n=700, random_state=1234).apply(lambda x: x[x.rfind('/')+1:]).tolist()
  
  # FN
  lists['FN {}'.format(name_rename[f['name']])] = df[(df['y']==1) & (df['yhat_pred']==0)]['file'].apply(lambda x: x[x.rfind('/')+1:]).tolist()
  

# Combine lists
to_annotate = []
for l in lists.values():
  to_annotate.extend(l)
to_annotate = pd.DataFrame().from_dict({'study':to_annotate}).drop_duplicates()

for k, l in lists.items():
  to_annotate[k] = to_annotate['study'].isin(l)
to_annotate['chest_tube'] = np.NaN
to_annotate = to_annotate[['study','chest_tube','FP DeiT', 'FP DN', 'FN DeiT', 'FN DN']]
to_annotate.to_csv('/cis/home/zmurphy/code/transformer-radiographs/image_lists/failure_pre_cxr.csv')