"""
This script downloads data for the MURA dataset.

Arguments:
- cfg-dir: full path to cfg.json containing the link to the data
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
parser.add_argument("--target-dir", default='mura', type=str, help='')
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
os.mkdir(os.path.join(target_dir, 'periph'))
os.mkdir(os.path.join(target_dir, 'images'))

tmp_file = os.path.join(args.tmp_dir,'mura.zip')
tmp_image_dir = args.tmp_dir

# Download to scratch and extract
urllib.request.urlretrieve(cfg['mura_link'], tmp_file)
with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
    zip_ref.extractall(tmp_image_dir)
os.remove(tmp_file)

files=[]
for r, d, f in os.walk(tmp_image_dir):
  for filed in f:
    if '.png' in filed:
      files.append(os.path.join(r, filed))
print(len(files))

head_path = '/content/MURA-v1.1/images'

dat = pd.DataFrame.from_dict({'OldPath': files})
dat['Path'] = dat['OldPath'].apply(lambda x: x.replace(os.path.join(tmp_image_dir,'MURA-v1.1')+'/', ''))
dat = dat[['/._' not in x for x in dat['Path']]]

print(dat)

Dataset = []
Region = []
Patient = []
Study = []
Image = []
for p in dat['Path']:
  row = p.split('/')
  Dataset.append(row[0])
  Region.append(row[1])
  Patient.append(row[2])
  Study.append(row[3])
  Image.append(row[4])
dat['Dataset'] = Dataset
dat['Region'] = Region
dat['Patient'] = Patient
dat['Study'] = Study
dat['Image'] = Image
dat['Positive'] = dat['Study'].apply(lambda x: 1 if 'positive' in x else 0)

dat['Region'] = dat['Region'].apply(lambda x: x.replace('XR_',''))
dat['Patient'] = dat['Patient'].apply(lambda x: x[len('patient'):])
dat['Study'] = dat['Study'].apply(lambda x: x[:x.rfind('_')].replace('study',''))

dat['NewPath'] = dat.apply(lambda x: '{}_{}_p{}_s{}_{}'.format(x['Dataset'], x['Region'], x['Patient'], x['Study'], x['Image']), axis=1)
print(dat[dat['NewPath'] == 'train_WRIST_p07840_s2_image2.png']['OldPath'])
dat = dat.apply(lambda x: x.replace('_._','_'))
dat.to_csv(os.path.join(target_dir,'periph/mura_metadata.csv'), index=False)

# Copy images to target
dir_ct = -1
for i, r in dat.iterrows():
  if i % 1000 == 0:
    dir_ct += 1
    os.mkdir(os.path.join(target_dir, 'images', str(dir_ct)))
  shutil.copyfile(r['OldPath'], os.path.join(target_dir, 'images', str(dir_ct), r['NewPath']))
  print('\r{}'.format(i), end='')

# Official test set
dat[dat['Dataset']=='valid']['NewPath'].to_csv(os.path.join(target_dir,'test.txt'), header=None, index=None)

# Train/val split at patient level
from sklearn.model_selection import train_test_split

train_patients, val_patients = train_test_split(dat[dat['Dataset']=='train']['Patient'].drop_duplicates(), test_size=0.1, random_state=42)
dat_train = dat[dat['Patient'].isin(train_patients.tolist())]
dat_val = dat[dat['Patient'].isin(val_patients.tolist())]

dat_train['NewPath'].to_csv(os.path.join(target_dir, 'train_100.txt'), header=None, index=None)
dat_val['NewPath'].to_csv(os.path.join(target_dir, 'val_100.txt'), header=None, index=None)

labels = dat[['NewPath', 'Positive']]
labels.columns = ['Image', 'Positive']
labels.to_csv(os.path.join(target_dir,'labels.csv'), index=False)

shutil.rmtree(os.path.join(args.tmp_dir,'MURA-v1.1'))

pd.concat([dat_train,dat_val])['NewPath'].to_csv(os.path.join(target_dir,'train_all.txt'), index=False, header=False)

# Subset sizes (by % studies)
subset_ptcs = [90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
# Subset sizes (by # studies), hand-crafted numbers (ensuring x mod 16 != 1 to avoid data loading errors)
subset_n_studies = [12111, 10766, 9420, 8074, 6729, 5383, 4037, 2691, 1346, 135]
assert len(subset_ptcs) == len(subset_n_studies)

# Load train_all.txt
dat = pd.read_table(os.path.join(target_dir,'train_all.txt'), header=None)
dat.columns = ['file']
dat['study'] = dat['file'].apply(lambda x: x[:x.find('_image')])
studies = dat['study'].drop_duplicates()
print(dat)
print(len(dat), len(studies))
for i in range(len(subset_ptcs)):
    print(subset_n_studies[i]/len(studies), end=' ')
print('\n')

# Loop through subset sizes
for i in range(len(subset_ptcs)):
    studies_subsample = studies.sample(n=subset_n_studies[i], random_state=42)
    dat_sub = dat[dat['study'].isin(studies_subsample)]
    dat_sub['file'].to_csv(os.path.join(target_dir,'train_all_{}.txt').format(subset_ptcs[i]),
                           header=None, index=False)
    # Update for next subsample
    dat = dat_sub
    studies = dat['study'].drop_duplicates()