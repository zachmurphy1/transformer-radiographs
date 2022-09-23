"""Gets bootstrapped metrics for given test results

Input:
  to_analyze: JSON list of models to bundle
    Structure:
      [
        {
          file: path to .pkl file from test.py
          name: display name for model
        }
      ]
  
  test results: .pkl file for each entry in to_analyze, located in [results-dir]

Args:
  See parser below
  
Output:
  bootstrap_raw.pkl: pickle of dict saved to [dir-name]
    Structure:
    {
      [name from to_analyze]:{
        [dataset]: DataFrame with row for each resample, columns as below
      }
    }

"""

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
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final', type=str, help='')
parser.add_argument("--to-analyze", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final/to_analyze_CXR100-Extended.json', type=str, help='')
parser.add_argument("--dir-name", default='CXR100_Extended', type=str, help='')
parser.add_argument("--bootstrap-dir", default='bootstrap_raw.pkl', type=str, help='')
parser.add_argument("--plots", default='y', type=str, help='')
args = parser.parse_args()


# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
labels = cfg['labels_chexnet_14_standard']
args.scratch_dir = args.scratch_dir.replace('~',os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))
args.to_analyze = args.to_analyze.replace('~',os.path.expanduser('~'))
args.dir_name = os.path.join(args.results_dir, args.dir_name)
if os.path.exists(args.dir_name):
  shutil.rmtree(args.dir_name)
os.mkdir(args.dir_name)

with open(args.to_analyze, 'r') as f:
  to_analyze = json.load(f)

# Set n resamples
n = 1000

# Set random seeds
random.seed(42)
seeds = random.sample(range(1,100000000),n)

# Get bootstrapped metrics
results = {}
for t in to_analyze:
  print(t['file'])
  with open(t['file'], 'rb') as f:
    m = pickle.load(f)
    
  sets = {}
  for s in m.keys():
    y = np.array(m[s]['y'])
    yhat = np.array(m[s]['yhat'])
    name = '{}-{}'.format(t['name'], s)
    
    # Resample
    metrics = {
      'auc_weighted':[],
      'precision_micro':[],
      'recall_micro':[],
      'f1_micro':[],
      'accuracy_micro':[]
    }
    for l in labels:
      metrics['auc_{}'.format(l)] = []
      metrics['wt_{}'.format(l)] = []
      metrics['precision_{}'.format(l)] = []
      metrics['recall_{}'.format(l)] = []
      metrics['f1_{}'.format(l)] = []
      metrics['accuracy_{}'.format(l)] = []
      metrics['thresh_{}'.format(l)] = []
    for i in range(n):
      random.seed(seeds[i])
      # Resample
      resampled_idxs = random.choices(range(y.shape[0]), k=y.shape[0])
      y_rs = y[resampled_idxs,:]
      yhat_rs = yhat[resampled_idxs,:]
        
      # By label
      
      confusion_matrix_total = np.zeros((2,2))
      for li, l in enumerate(labels):
        ys_sub = np.array(y_rs)[:,li]
        yhats_sub = np.array(yhat_rs)[:,li]
        if (np.mean(ys_sub) > 0) and (np.mean(ys_sub) < 1):
          # AUC
          metrics['auc_{}'.format(l)].append(roc_auc_score(ys_sub, yhats_sub))
          metrics['wt_{}'.format(l)].append(np.sum(ys_sub)/np.sum(y_rs))
          
          # Confusion matrix metricx
          # Get optimal threshold
          fpr, tpr, thresholds = roc_curve(ys_sub, yhats_sub, pos_label=1)
          fnr = 1 - tpr
          op_idx = np.nanargmin(np.absolute(((tpr) - (1-fpr))))
          op_thresh = thresholds[op_idx]
          metrics['thresh_{}'.format(l)].append(op_thresh)
          # Confusion matrix
          confusion_matrix = np.zeros((2,2))
          for j in range(y_rs.shape[0]):
            pred = 0
            if yhat_rs[j,li] >= op_thresh:
              pred = 1
            confusion_matrix[pred, int(y_rs[j,li])] += 1

          # Calculate confusion matrix metrics
          metrics['precision_{}'.format(l)].append(confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0]))
          metrics['recall_{}'.format(l)].append(confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1]))
          metrics['f1_{}'.format(l)].append(2*metrics['precision_{}'.format(l)][-1]*metrics['recall_{}'.format(l)][-1]/(metrics['precision_{}'.format(l)][-1]+metrics['recall_{}'.format(l)][-1]))
          metrics['accuracy_{}'.format(l)].append((confusion_matrix[0,0] + confusion_matrix[1,1]) / confusion_matrix.sum())
          # Add to confusion matrix
          confusion_matrix_total = np.add(confusion_matrix_total, confusion_matrix)
      
        else:
          metrics['auc_{}'.format(l)].append(np.NaN)
          metrics['wt_{}'.format(l)].append(np.NaN)
          metrics['precision_{}'.format(l)].append(np.NaN)
          metrics['recall_{}'.format(l)].append(np.NaN)
          metrics['f1_{}'.format(l)].append(np.NaN)
          metrics['accuracy_{}'.format(l)].append(np.NaN)
          metrics['thresh_{}'.format(l)].append(np.NaN)

          
          
      # Aggregate over labels
      metrics['auc_weighted'].append(np.average([metrics['auc_{}'.format(x)][-1] for x in labels if not math.isnan(metrics['auc_{}'.format(x)][-1])], weights=[metrics['wt_{}'.format(x)][-1] for x in labels if not math.isnan(metrics['wt_{}'.format(x)][-1])]))
      
      metrics['precision_micro'].append(confusion_matrix_total[1,1] / (confusion_matrix_total[1,1] + confusion_matrix_total[1,0]))
      metrics['recall_micro'].append(confusion_matrix_total[1,1] / (confusion_matrix_total[1,1] + confusion_matrix_total[0,1]))
      metrics['f1_micro'].append(2*metrics['precision_micro'][-1]*metrics['recall_micro'][-1]/(metrics['precision_micro'][-1]+metrics['recall_micro'][-1]))
      metrics['accuracy_micro'].append((confusion_matrix_total[0,0] + confusion_matrix_total[1,1]) / confusion_matrix_total.sum())

    sets[s] = pd.DataFrame(metrics)     
  results[t['name']] = sets

# Save raw bootstrap
with open(os.path.join(args.dir_name, args.bootstrap_dir), 'wb') as f:
    pickle.dump(results, f)