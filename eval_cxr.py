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
parser.add_argument("--to-analyze", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final/to_analyze_CXR100.json', type=str, help='')
parser.add_argument("--dir-name", default='CXR100_final', type=str, help='')
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
# if os.path.exists(args.dir_name):
#   shutil.rmtree(args.dir_name)
# os.mkdir(args.dir_name)

with open(args.to_analyze, 'r') as f:
  to_analyze = json.load(f)
  
# Read bootstrap data
with open(os.path.join(args.dir_name, args.bootstrap_dir), 'rb') as f:
    results = pickle.load(f)
  

# Get results table
rtable = pd.DataFrame()
ci = 0.95
ci_lower = ((1.0-ci)/2.0) * 100
ci_upper = (ci+((1.0-ci)/2.0)) * 100

print(results.keys())

for m_, m in results.items():
  for s_, s in m.items():
    
    r = s
    
    means = r.apply(lambda x: np.nanmean(x), axis=0)
    lci = r.apply(lambda x: max(0.0, np.percentile(x, ci_lower)), axis=0)
    uci = r.apply(lambda x: min(1.0, np.percentile(x, ci_upper)), axis=0)
    
    r_ = pd.concat([means,lci,uci], axis=1).reset_index(drop=False)
    
    r_.columns = ['Measure','Mean','LCI','UCI']
    r_['Model'] = m_
    r_['Set'] = s_
    r_ = r_[['Model','Set','Measure','Mean','LCI','UCI']]
    r_['Display'] = r_.apply(lambda x: '{:.3f} [{:.3f}-{:.3f}]'.format(x['Mean'], x['LCI'], x['UCI']), axis=1)
    rtable = pd.concat([rtable,r_], ignore_index=True)

rtable.to_csv(os.path.join(args.dir_name, 'results_table.csv'),index=False)

# Pairwise comparisons
comps = []
for m_, m in results.items():
  for s_, s in m.items():
    comps.append((m_,s_))
    
pc = []
for var in ['auc_weighted',
            'precision_micro','recall_micro','f1_micro','accuracy_micro'] + ['{}_{}'.format(m,l) for m in ['auc','precision','recall','f1','accuracy'] for l in labels]:
  for pair in itertools.combinations(comps, 2):
    model1 = pair[0][0]
    set1 = pair[0][1]
    model2 = pair[1][0]
    set2 = pair[1][1]

    list1 = results[model1][set1][var]
    list2 = results[model2][set2][var]
    ttest = stats.ttest_ind(list1,list2, nan_policy='omit')
    pc.append({'Model 1':model1, 'Set 1':set1, 'Model 2':model2, 'Set 2':set2, 'Metric':var,
               't':ttest.statistic, 'p':ttest.pvalue})
pc = pd.DataFrame(pc)
pc.to_csv(os.path.join(args.dir_name, 'ttest_table.csv'),index=False)


# Plots by label
if args.plots == 'y':
  dataset_nicenames = {
      'nihcxr14_test': 'NIH CXR14',
      'chexpert_test': 'CheXpert',
      'mimic_test': 'MIMIC',
      'padchest_test': 'PadChest'
  }
  model = ['DeiT','DenseNet121']
  dataset = ['nihcxr14_test','padchest_test','chexpert_test','mimic_test']
  metric = 'auc'


  for d in dataset:
    fig, ax = plt.subplots()
    for i, m in enumerate(model):
      dat_plot = pd.DataFrame()
      for l in labels:
        row = rtable[(rtable['Model']==m)&(rtable['Set']==d)&(rtable['Measure']=='{}_{}'.format(metric,l))]
        row['Label'] = l.replace('_',' ')
        dat_plot = dat_plot.append(row, ignore_index=True)

      colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
      shift = np.arange(-((len(model)-1)/2),(len(model)-1)/2+1,1)
      width = 0.8/len(model)


      labels_sub = dat_plot[~dat_plot['Mean'].isna()]['Label'].drop_duplicates()


      dat_sub = dat_plot[~dat_plot['Mean'].isna()]
      plt.bar(x=[x+shift[i]*width for x in range(len(labels_sub))],
              height=dat_sub['Mean'],
                yerr=[dat_sub['Mean']-dat_sub['LCI'], dat_sub['UCI']-dat_sub['Mean']],
                width=width,
                zorder=5,
                color=colors[i],
                label=m
             )


      weighted_mean = rtable[(rtable['Model'] == m) & 
                                    (rtable['Set'] == d) &
                                    (rtable['Measure'] == 'auc_weighted')
                                  ]    
      plt.axhline(weighted_mean['Mean'].tolist()[0], zorder=4, color=colors[i],label=None, linewidth=1)

    ax.set_xticks(range(len(labels_sub)))
    ax.set_xticklabels([x.replace('_', ' ') for x in labels_sub], rotation=-60, ha='left')
    
    # Significance labels
    adj=0
    alpha = 0.05
    for i, l in enumerate([x.replace(' ', '_') for x in labels_sub]):
      # Get p value
      pc_ = pc[(pc['Set 1']==d)&(pc['Set 2']==d)]
      pc_ = pc_[pc_['Metric']=='{}_{}'.format(metric,l)]
      p = pc_['p'].tolist()[0]
      
      if p < alpha:
        y = rtable[(rtable['Set']==d)&(rtable['Measure']=='{}_{}'.format(metric,l))]['UCI'].max()
        ax.annotate('*', (i, y+adj), horizontalalignment='center', zorder=10)

    ax.set_ylabel('AUC')
    ax.set_xlabel('Finding')
    ax.set_ylim(0.5,1)
    ax.legend([m_ for m_ in model], loc='lower right')
    ax.set_title(dataset_nicenames[d])
    ax.grid(True, axis='y', zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(args.dir_name, d+'.png'))



