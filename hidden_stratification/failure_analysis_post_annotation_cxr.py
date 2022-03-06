hu import numpy as np
import pandas as pd
import argparse
import os
import json
from scipy.stats import chi2_contingency as chi2

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--image-list-dir", default='/cis/home/zmurphy/code/transformer-radiographs/image_lists/', type=str, help='')
parser.add_argument("--annotations", default='failure_post_cxr.csv', type=str, help='')
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

# Get lists
deit_fp = ann[ann['FP DeiT']]['study'].tolist()
deit_fn = ann[ann['FN DeiT']]['study'].tolist()
dn_fp = ann[ann['FP DN']]['study'].tolist()
dn_fn = ann[ann['FN DN']]['study'].tolist()

# FP
fp = pd.DataFrame({'FP DeiT':ann[ann['FP DeiT']]['chest_tube'].value_counts(),
                   'FP DN':ann[ann['FP DN']]['chest_tube'].value_counts()})
print(fp.sort_index())
print('X={}, p={}'.format(chi2(fp)[0],chi2(fp)[1]))

# FN
fn = pd.DataFrame({'FN DeiT':ann[ann['FN DeiT']]['chest_tube'].value_counts(),
                   'FN DN':ann[ann['FN DN']]['chest_tube'].value_counts()})
print(fn.sort_index())
print('X={}, p={}'.format(chi2(fn)[0],chi2(fn)[1]))