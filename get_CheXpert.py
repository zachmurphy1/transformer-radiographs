"""
This script downloads data for the CheXpert dataset.

Arguments:
- cfg-dir: full path to cfg.json containing urls to data
- tmp-dir: scratch directory for processing data
- target-dir: subdirectory to save data under data_dir from cfg.json
"""

# Imports
import pandas as pd
import os, sys
import urllib.request, shutil, os, zipfile, json

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--tmp-dir", default='/export/gaon1/data/zmurphy', type=str, help='')
parser.add_argument("--target-dir", default='chexpert', type=str, help='')
args = parser.parse_args()

# Fill in home dirs
args.cfg_dir = args.cfg_dir.replace('~',os.path.expanduser('~'))
args.tmp_dir = args.tmp_dir.replace('~',os.path.expanduser('~'))

# Get cfg
with open(args.cfg_dir,'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~',os.path.expanduser('~'))

# Set dirs
target_dir = os.path.join(cfg['data_dir'],args.target_dir)
os.mkdir(target_dir)
image_dir = os.path.join(target_dir, 'images')
os.mkdir(image_dir)


# Get download links
link = cfg['chexpert_link']

# For each download zip
dir_ct = -1
ct = 0


# Set dirs
tmp_file = os.path.join(args.tmp_dir,link[link.rfind('/')+1:])
tmp_image_dir = os.path.join(args.tmp_dir,'tmp')

# Download to tmp_image_dir and extract
urllib.request.urlretrieve(link, tmp_file)
os.mkdir(tmp_image_dir)
with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
   zip_ref.extractall(tmp_image_dir)
os.remove(tmp_file)

# Get sample
labels = cfg['labels_chexnet_14_standard']

dat = pd.read_csv(os.path.join(tmp_image_dir,'CheXpert-v1.0-small/train.csv'))
dat = dat[dat['Frontal/Lateral'] == 'Frontal']
dat.rename(columns={'Pleural Effusion': 'Effusion'}, inplace=True)
for l in labels:
  if not l in dat.columns:
    dat[l] = 0
  dat[l] = dat[l].apply(lambda x: 1 if x==1 else 0)

dat['Image'] = dat['Path'].apply(lambda x: x.replace('CheXpert-v1.0-small/train/','').replace('/',''))

# Get sample
dat_sample = dat.sample(n=25000, random_state=42)

# Copy sample
n = 1000
file_num = -1
for i,r in dat_sample[['Path','Image']].reset_index().iterrows():
  if i % n == 0:
    file_num += 1
    os.mkdir(os.path.join(image_dir,str(file_num)))
  shutil.copy(os.path.join(tmp_image_dir,r['Path']), os.path.join(image_dir, str(file_num), r['Image']))
  print('\rCopied {}/{}'.format(i+1,dat_sample.shape[0]), end='')
    
dat_sample['Image'].to_csv(os.path.join(target_dir,'test.txt'), header=None, index=False)

shutil.rmtree(tmp_image_dir)
