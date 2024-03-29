import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import argparse
import os, sys
import json

# Parse args
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final', type=str, help='')
parser.add_argument("--dir-name", default='MURAsmall_final', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))
args.dir_name = os.path.join(args.results_dir, args.dir_name)

# Declare sets
sets = [
    ('mura_test', 'MURA')
]

# Read and format results table
rtable = pd.read_csv(os.path.join(args.dir_name,'results_table.csv'))
rtable = rtable[rtable['Measure']=='auc_weighted']
rtable['Size'] = rtable['Model'].apply(lambda x: float(x[x.rfind(' '):]))
rtable['Model'] = rtable['Model'].apply(lambda x: x[:x.rfind(' ')])
rtable

# Read and format ttests table
ttests = pd.read_csv(os.path.join(args.dir_name,'ttest_table.csv'))
ttests = ttests[ttests['Metric']=='auc_weighted']
ttests['Size 1'] = ttests['Model 1'].apply(lambda x: float(x[x.rfind(' '):]))
ttests['Size 2'] = ttests['Model 2'].apply(lambda x: float(x[x.rfind(' '):]))
ttests['Model 1'] = ttests['Model 1'].apply(lambda x: x[:x.rfind(' ')])
ttests['Model 2'] = ttests['Model 2'].apply(lambda x: x[:x.rfind(' ')])
ttests['p'] = ttests['p'].astype(float)
ttests

# Replace for 1 and 10 sizes
from scipy.stats import wilcoxon
ci = 0.95
ci_lower = ((1.0-ci)/2.0) * 100
ci_upper = (ci+((1.0-ci)/2.0)) * 100
dat_m = pd.read_csv('/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final/MURA1-10/results_table.csv')
dat_m = dat_m[dat_m['Measure']=='auc_weighted']
dat_m ['Size'] = dat_m ['Model'].apply(lambda x: x[x.find(' ')+1:x.find('-')]).astype(float)
dat_m ['Model'] = dat_m ['Model'].apply(lambda x: x[:x.find(' ')])
for se in dat_m['Set'].drop_duplicates().tolist():
  for s in dat_m['Size'].drop_duplicates().tolist():
    means = {}
    for m in dat_m['Model'].drop_duplicates().tolist():
      dat_m_ = dat_m[(dat_m['Model']==m)&(dat_m['Size']==s)&(dat_m['Set']==se)]
      means[m] = dat_m_['Mean'].tolist()
      mean = np.nanmean(dat_m_['Mean'])
      lci = np.percentile(dat_m_['Mean'], ci_lower)
      uci = np.percentile(dat_m_['Mean'], ci_upper)
      
      rtable.loc[(rtable['Model']==m)&(rtable['Size']==s)&(rtable['Set']==se),'Mean'] = mean
      rtable.loc[(rtable['Model']==m)&(rtable['Size']==s)&(rtable['Set']==se),'LCI'] = lci
      rtable.loc[(rtable['Model']==m)&(rtable['Size']==s)&(rtable['Set']==se),'UCI'] = uci
      rtable.loc[(rtable['Model']==m)&(rtable['Size']==s)&(rtable['Set']==se),'Display'] = ''
    mannw = wilcoxon(means['DeiT'],means['DenseNet121'])
    ttests.loc[(ttests['Set 1']==se)&(ttests['Set 2']==se)&(ttests['Metric']=='auc_weighted')&(ttests['Size 1']==s)&(ttests['Size 2']==s)&(ttests['Model 1']!=ttests['Model 2']), 't'] = mannw.statistic
    ttests.loc[(ttests['Set 1']==se)&(ttests['Set 2']==se)&(ttests['Metric']=='auc_weighted')&(ttests['Size 1']==s)&(ttests['Size 2']==s)&(ttests['Model 1']!=ttests['Model 2']), 'p'] = mannw.pvalue
    
# Rename DeiT --> DeiT-B
rtable['Model'] = rtable['Model'].replace('DeiT','DeiT-B')
ttests['Model 1'] = ttests['Model 1'].replace('DeiT','DeiT-B')
ttests['Model 2'] = ttests['Model 2'].replace('DeiT','DeiT-B')


# Significance value
p = 0.05

# Y adjust for annoations
adj=0

for s in sets:
    rtable_ = rtable[rtable['Set']==s[0]]
    
    shifts = [-.25,.25]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    
    # Minmax by size container
    minmax_by_size = {}
    for i in rtable_['Size'].drop_duplicates():
        minmax_by_size[i] = {'min':1, 'max':0}
    
    # Plot each model
    for i, model in enumerate(['DeiT-B', 'DenseNet121']):
        dat_sub = rtable_[rtable_['Model']==model]
        
        plt.errorbar(dat_sub['Size']+shifts[i], dat_sub['Mean'], yerr=[dat_sub['Mean']-dat_sub['LCI'],dat_sub['UCI']-dat_sub['Mean']], label=model, zorder=5+i)
        plt.plot(zorder=5+i)
        
        # Update minmax by size
        for r_i, r in dat_sub.iterrows():
            minmax_by_size[r['Size']]['max'] = max(r['UCI'], minmax_by_size[r['Size']]['max'])
            minmax_by_size[r['Size']]['min'] = min(r['LCI'], minmax_by_size[r['Size']]['min'])
    
    # Significance labels
    ttests_ = ttests[(ttests['Set 1']==s[0])&(ttests['Set 2']==s[0])]
    ttests_ = ttests_[ttests_['Size 1']==ttests_['Size 2']]
    for i, r in ttests_.iterrows():
        if r['p'] <= p:
            ax.annotate('*', (r['Size 1'], minmax_by_size[r['Size 1']]['max']+adj), horizontalalignment='center')
    
    # 1 and 10 caveats
    hadj = 2
    ax.annotate('†', (1-hadj, minmax_by_size[1]['max']+adj), horizontalalignment='center')
    ax.annotate('†', (10-hadj, minmax_by_size[10]['max']+adj), horizontalalignment='center')
    
    plt.legend(loc='lower right')
    plt.title(s[1])
    plt.xlabel('Dataset size (%)')
    plt.ylabel('AUC')
    fig.savefig(os.path.join(args.dir_name,'performance_by_size_{}.png'.format(s[0])))
