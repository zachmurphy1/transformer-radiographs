"""
This script downloads data for the PadChest dataset.

Arguments:
- cfg-dir: full path to cfg.json containing urls to data
- tmp-dir: scratch directory for processing data
- target-dir: subdirectory to save data under data_dir from cfg.json
"""
# Imports
import pandas as pd
import os, sys
import urllib.request, shutil, os, tarfile, json
import getpass
from pathlib import Path
from PIL import Image

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--tmp-dir", default='/export/gaon1/data/zmurphy', type=str, help='')
parser.add_argument("--target-dir", default='nihcxr14', type=str, help='')
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
#os.mkdir(target_dir)
image_dir = os.path.join(target_dir, 'images')
#os.mkdir(image_dir)

# Process metadata
dat = pd.read_csv(os.path.join(target_dir,'periph/metadata.csv'))
dat = dat[dat['ViewPosition'].isin(['AP','PA'])]

dat_labels = pd.read_csv(os.path.join(target_dir,'periph/chexpert_labels.csv'))
dat = pd.merge(dat,dat_labels, how='inner', on=['subject_id','study_id'])

dat.rename(columns={'Pleural Effusion': 'Effusion'}, inplace=True)
for l in cfg['labels_chexnet_14_standard']:
  if not l in dat.columns:
    dat[l] = 0
  dat[l] = dat[l].apply(lambda x: 1 if x==1 else 0)

dat['Image'] = dat['dicom_id'].apply(lambda x: '{}.jpg'.format(x))
dat['Path'] = dat[['subject_id', 'study_id', 'dicom_id']].apply(lambda x: 'p{}/p{}/s{}/{}.jpg'.format(str(x[0])[:2], x[0], x[1], x[2]),axis=1)
dat = dat[['Path', 'Image']+cfg['labels_chexnet_14_standard']]
dat['Image'] = dat['Image'].apply(lambda x: x.replace('.jpg', '.png'))
dat[['Image']+cfg['labels_chexnet_14_standard']].to_csv(os.path.join(target_dir, 'labels.csv'), index=False)

# Get sample
dat_sample = dat.sample(n=25000, random_state=42)
dat_sample['Image'].to_csv(os.path.join(target_dir, 'test.csv'), header=None, index=False)

# Download images
un = input('PhysioNet User:')
pw = getpass.getpass('PhysioNet Password:')

for i, f in enumerate(dat_sample['Path'].tolist()):
  origPath = cfg['mimic_server'] + f
  destPath = os.path.join(image_dir,f)
  path = Path(destPath[:destPath.rfind('/')])
  path.mkdir(parents=True, exist_ok=True)
  os.system('wget -r -N -c -np --user {} --password {} {} -O {}'.format(un,pw,origPath, destPath))

  png = Image.open(destPath)
  png.thumbnail((1024,1024), resample=Image.LANCZOS)
  #png = Image.fromarray(np.uint8(np.array(png) / 256)).convert(mode='L')
  
  png.save(destPath)
    
  print('\r{}'.format(i), end='')