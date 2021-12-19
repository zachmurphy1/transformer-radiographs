import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import argparse
import os
import json

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/results/final', type=str, help='')
parser.add_argument("--dir-name", default='CXRsmall_final', type=str, help='')
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
    ('nihcxr14_test', 'NIHCXR14'),
    ('chexpert_test', 'CheXpert'),
    ('padchest_test', 'PadChest'),
    ('mimic_test', 'MIMIC')
]

# Read and format results table
rtable = pd.read_csv(os.path.join(args.dir_name,'results_table.csv'))
rtable = rtable[rtable['Measure']=='auc_weighted']
rtable['Size'] = rtable['Model'].apply(lambda x: float(x[x.rfind(' '):]))
rtable['Model'] = rtable['Model'].apply(lambda x: x[:x.rfind(' ')])
    
# Read and format ttests table
ttests = pd.read_csv(os.path.join(args.dir_name,'ttest_table.csv'))
ttests = ttests[ttests['Metric']=='auc_weighted']
ttests['Size 1'] = ttests['Model 1'].apply(lambda x: float(x[x.rfind(' '):]))
ttests['Size 2'] = ttests['Model 2'].apply(lambda x: float(x[x.rfind(' '):]))
ttests['Model 1'] = ttests['Model 1'].apply(lambda x: x[:x.rfind(' ')])
ttests['Model 2'] = ttests['Model 2'].apply(lambda x: x[:x.rfind(' ')])
ttests['p'] = ttests['p'].astype(float)

# Significance value
p = 0.05

# Y adjust for annoations
adj=0

for s in sets:
    rtable_ = rtable[rtable['Set']==s[0]]
    
    shifts = [-.25,.25]
    fig, ax = plt.subplots()
    
    # Minmax by size container
    minmax_by_size = {}
    for i in rtable_['Size'].drop_duplicates():
        minmax_by_size[i] = {'min':1, 'max':0}
    
    # Plot each model
    for i, model in enumerate(['DeiT', 'DenseNet121']):
        dat_sub = rtable_[rtable_['Model']==model]
        
        plt.errorbar(dat_sub['Size']+shifts[i], dat_sub['Mean'], yerr=[dat_sub['Mean']-dat_sub['LCI'],dat_sub['UCI']-dat_sub['Mean']], label=model)
        plt.plot()
        
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
    
    plt.legend(loc='lower right')
    plt.title(s[1])
    plt.xlabel('Dataset size (%)')
    plt.ylabel('AUC')
    fig.savefig(os.path.join(args.dir_name,'performance_by_size_{}.png'.format(s[0])))


