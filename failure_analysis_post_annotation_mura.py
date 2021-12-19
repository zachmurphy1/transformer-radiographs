import numpy as np
import pandas as pd
import argparse
import os
import json
from scipy.stats import chi2_contingency as chi2

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--image-list-dir", default='/cis/home/zmurphy/code/transformer-radiographs/image_lists/', type=str, help='')
parser.add_argument("--annotations", default='failure_post_mura.csv', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
args.image_list_dir = args.image_list_dir.replace('~',os.path.expanduser('~'))
args.annotations = args.annotations.replace('~',os.path.expanduser('~'))

# Load annotations
ann = pd.read_csv(os.path.join(args.image_list_dir,args.annotations))

# Condense labels
ann['label'] = ann['label'].apply(lambda x: 'Other' if 'Other (' in x else x)
ann['label'] = ann['label'].replace('Fracture', 'Fracture or Amputation')

# Get lists
deit_fn = ann[ann['DeiT']]['study'].tolist()
dn_fn = ann[ann['DN']]['study'].tolist()

# Get by region
ann['region'] = ann['study'].apply(lambda x: x[x.find('_')+1:])
ann['region'] = ann['region'].apply(lambda x: x[:x.find('_')].capitalize())

# Chi square overall annd by region
print('Overall')
fn = pd.DataFrame({'DeiT':ann[ann['DeiT']]['label'].value_counts(),
                   'DN':ann[ann['DN']]['label'].value_counts()}).fillna(0)
print(fn.sort_index())
print('X = {:.3f}, p = {:.3f}'.format(chi2(fn)[0],chi2(fn)[1]))

for r in ann['region'].drop_duplicates().tolist():
    print('')
    print(r)
    ann_ = ann[ann['region']==r]
    fn = pd.DataFrame({'DeiT':ann_[ann_['DeiT']]['label'].value_counts(),
                       'DN':ann_[ann_['DN']]['label'].value_counts()}).fillna(0)
    print(fn.sort_index())
    print('X = {:.3f}, p = {:.3f}'.format(chi2(fn)[0],chi2(fn)[1]))