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
import urllib.request, shutil, os, zipfile, json

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--tmp-dir", default='/export/gaon1/data/zmurphy', type=str, help='')
parser.add_argument("--target-dir", default='padchest', type=str, help='')
args = parser.parse_args()

# Fill in home dirs
args.cfg_dir = args.cfg_dir.replace('~',os.path.expanduser('~'))
args.tmp_dir = args.tmp_dir.replace('~',os.path.expanduser('~'))

# Get cfg
with open(args.cfg_dir,'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~',os.path.expanduser('~'))

# Get dirs
target_dir = os.path.join(cfg['data_dir'],args.target_dir)
image_dir = os.path.join(target_dir, 'images')
os.mkdir(image_dir)

# Process raw metadata
labels = {
  'Atelectasis': ["'atelectasis'", "'total atelectasis'", "'lobal atelectasis'", "'segmental atelectasis'", "'laminar atelectasis'", "'round atelectasis'", "'atelectasis basal'"],
  'Cardiomegaly': ["'cardiomegaly'"],
  'Effusion': ["'pleural effusion'", "'loculated pleural effusion'", "'loculated fissural effusion'", "'hydropneumothorax'", "'empyema'", "'hemothorax'"],
  'Infiltration': ["'infiltrates","'interstitial pattern'", "'ground glass pattern'", "'reticular interstitial pattern'", "'reticulonodular interstitial pattern'", "'miliary opacities'", "'alveolar pattern'", "'air bronchogram'"],
  'Mass': ["'mass'", "'mediastinal mass'", "'breast mass'", "'pleural mass'", "'pulmonary mass'", "'soft tissue mass'"],
  'Nodule': ["'nodule'", "'multiple nodules'"],
  'Pneumonia': ["'pneumonia'", "'atypical pneumonia'"],
  'Pneumothorax': ["'pneumothorax'", "'hydropneumothorax'"],
  'Consolidation': ["'consolidation'", "'air bronchogram'"],
  'Edema': ["'pulmonary edema'"],
  'Emphysema': ["'emphysema'"],
  'Fibrosis': ["'pulmonary fibrosis'", "'post radiotherapy changes'", "'asbestosis signs'"],
  'Pleural_Thickening': ["'pleural thickening'", "'apical pleural thickening'", "'calcified pleural thickening'"],
  'Hernia': ["'hiatal hernia'"]
}

cols = ['ImageID', 'PatientID', 'ImageDir', 'Projection', 'Pediatric', 'Labels', 'labelCUIS']
dat = pd.read_csv(os.path.join(target_dir, 'periph/metadata.csv'), usecols=cols)

dat = dat[(dat.Projection == 'PA') | (dat.Projection == 'AP')]
dat = dat[dat.Pediatric == 'No']

mask = dat['Labels'].apply(lambda x: not ('exclude' in str(x)) | ('suboptimal' in str(x)))
dat = dat[mask]

for k, v in labels.items():
  dat[k] = dat['Labels'].apply(lambda x: 1 if any(l in str(x) for l in v) else 0)

dat.drop(columns=['Projection', 'Pediatric', 'Labels', 'labelCUIS'], inplace=True)

# Get random sample of 20,000 abnormal
abnormal = dat[(dat['ImageDir'] < 40) & (dat[labels.keys()].sum(axis=1)!=0)].sample(n=20000, random_state=42)
normal = dat[(dat['ImageDir'] < 15) & (dat[labels.keys()].sum(axis=1)==0)].sample(n=5000, random_state=21)

sample = pd.concat([abnormal,normal])

print(sample[labels.keys()].sum())
print((sample[labels.keys()].sum(axis=1)==0).value_counts())
print(sample['ImageDir'].value_counts().sort_index())

sample.drop(columns=['ImageDir', 'PatientID']).to_csv(os.path.join(target_dir, 'labels.csv'), index=False)
sample['ImageID'].to_csv(os.path.join(target_dir, 'test.txt'), index=False, header=False)

# Get images
from webdav3 import client
import zipfile
import os, shutil
from PIL import Image
from io import BytesIO

save_dir = os.path.join(target_dir, 'images')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

zips_to_get = sample['ImageDir'].unique().tolist()
for i, z in enumerate(zips_to_get):
  print(i,z)

def getImages(zip_to_get):
  tmp_zip_dir = os.path.join(args.tmp_dir, '{}.zip'.format(zip_to_get))
  c = client.Client(cfg['padchest_credentials'])
    
  c.download_sync(remote_path='{}.zip'.format(zip_to_get),local_path=tmp_zip_dir)
  
  files_to_get = sample[sample['ImageDir'] == zip_to_get]['ImageID'].tolist()
  
  os.mkdir(os.path.join(save_dir, str(zip_to_get)))
  with zipfile.ZipFile(tmp_zip_dir) as z:
    for img_i, img in enumerate(files_to_get):
        png = Image.open(BytesIO(z.read(img)))
        png.thumbnail((1024,1024), resample=Image.LANCZOS)
        png = Image.fromarray(np.uint8(np.array(png) / 256)).convert(mode='L')
        
        p = os.path.join(save_dir, str(zip_to_get), img)
        png.save(p)
        print('\r{} {}/{} saved'.format(zip_to_get, img_i, len(files_to_get)), end='')


  os.remove('{}.zip'.format(zip_to_get))
  print('{} done'.format(zip_to_get))

for z in zips_to_get:
  getImages(z)










