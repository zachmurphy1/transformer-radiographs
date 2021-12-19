"""
This script downloads data for the NIH CXR14 dataset.

Arguments:
- cfg-dir: full path to cfg.json containing urls to data
- tmp-dir: scratch directory for processing data
- target-dir: subdirectory to save data under data_dir from cfg.json
"""


# Imports
import pandas as pd
import os, sys
import urllib.request, shutil, os, tarfile, json

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
os.mkdir(target_dir)
image_dir = os.path.join(target_dir, 'images')
os.mkdir(image_dir)

# Get download links
links = cfg['nihcxr14_links']

# For each download zip
dir_ct = -1
ct = 0
for link in links:
    # Set dirs
    tmp_file = os.path.join(args.tmp_dir,link[link.rfind('/')+1:link.rfind('.')]+'.tar.gz')
    tmp_image_dir = os.path.join(args.tmp_dir,'images')

    # Download to scratch and extract
    urllib.request.urlretrieve(link, tmp_file)
    os.mkdir(tmp_image_dir)
    tar = tarfile.open(tmp_file, "r:gz")
    tar.extractall(path=args.tmp_dir)
    tar.close()
    os.remove(tmp_file)

    # Copy images to target dir, save 1000 in each directory
    _, _, files = next(os.walk(tmp_image_dir))
    for f in files:
        if ('.png' in f) | ('.jpg' in f):
          if ct % 1000 == 0:
            dir_ct += 1
            os.mkdir(os.path.join(image_dir,str(dir_ct)))

          shutil.copy(os.path.join(tmp_image_dir,f), os.path.join(image_dir, str(dir_ct), f[f.rfind('/')+1:]))
          ct += 1
          print('\rImages: {} Dir: {}'.format(ct, dir_ct), end='')
    shutil.rmtree(tmp_image_dir)
    print(link)

# Get image paths
all_files=[]
count = 0
for r, d, f in os.walk(os.path.join(cfg['data_dir'],args.target_dir)):
  for filed in f:
    if ('.png' in filed) | ('.jpg' in filed):
      all_files.append(os.path.join(r, filed))
with open(os.path.join(target_dir,'image_paths.txt'), 'w') as f:
  for i in all_files:
    f.write(i.replace(image_dir+'/','')+'\n')
    
# Get labels
metadata = pd.read_csv(os.path.join(target_dir,'periph/metadata.csv'))

labels = cfg['labels_chexnet_14_standard']

# Get label columns
for l in labels:
  metadata[l] = metadata['Finding Labels'].apply(lambda x: 1 if l in x else 0)

# Save metadata
metadata.rename(columns={'Image Index':'Image'}, inplace=True)
metadata[['Image']+labels].to_csv(os.path.join(target_dir,'labels.csv'), index=False)

# Split train into train/val at patient level
dat = metadata
train = pd.read_csv(os.path.join(target_dir,'periph/train_val_list.txt'),header=None)
dat = dat[dat['Image'].isin(train[0])]
patients = dat['Patient ID'].unique()

from sklearn.model_selection import train_test_split
# 100%
train, val = train_test_split(patients, test_size=0.1)
train100 = dat[dat['Patient ID'].isin(train)]
val100 = dat[dat['Patient ID'].isin(val)]
train100['Image'].to_csv(os.path.join(target_dir,'train_100.txt'), index=False, header=False)
val100['Image'].to_csv(os.path.join(target_dir,'val_100.txt'), index=False, header=False)

# Final training set
pd.concat([train100,val100])['Image'].to_csv(os.path.join(target_dir,'train_all.txt'), index=False, header=False)

# TODO: Smaller datasets