"""Gets yhat for model and dataset

Input:
  model state: assumes model state from training is at path [model-state]

Args:
  See parser below
  
Output:
  Appends to dict for dataset saved in pickle file at [results-dir]/[model-state].pkl
    Structure:
      {
        [dataset]_[test-file]:{
          y: list of ys
          yhat: list of yhats
          file: list of file names
        }
      }

"""

# Standard
import os, sys, shutil, json
import pandas as pd
import numpy as np
import time
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='/cis/home/zmurphy/code/transformer-radiographs/cfg.json', type=str, help='')
parser.add_argument("--model-state", default='', type=str, help='')
parser.add_argument("--labels-set", default='mura', type=str, help='')
parser.add_argument("--batch-size", default=16, type=int, help='')
parser.add_argument("--dataset", default='mura', type=str, help='')
parser.add_argument("--test-file", default='test.txt', type=str, help='')
parser.add_argument("--use-parallel", default='y', type=str, help='y | n')
parser.add_argument("--num-workers", default=12, type=int, help='')
parser.add_argument("--image-size", default=224, type=int, help='')
parser.add_argument("--print-batches", default='n', type=str, help='y | n')
parser.add_argument("--scratch-dir", default='/export/gaon1/data/zmurphy/transformer-cxr', type=str, help='')
parser.add_argument("--results-dir", default='/export/gaon1/data/zmurphy/transformer-cxr/mura_results/final', type=str, help='')
parser.add_argument("--copy-to-local", default='n', type=str, help='y | n')
parser.add_argument("--use-gpus", type=str, help='')
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.use_gpus

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import roc_auc_score

# Custom
import custom_modules as MURA


# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
labels = []
if args.labels_set == 'chexnet-14-standard':
    labels = cfg['labels_chexnet_14_standard']
elif args.labels_set == 'mura':
    labels = cfg['labels_mura_standard']
args.use_parallel = args.use_parallel == 'y'
args.print_batches = args.print_batches == 'y'
args.copy_to_local = args.copy_to_local == 'y'
args.scratch_dir = args.scratch_dir.replace('~',os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))

# Model params
model_args = {
    'model_state': args.model_state,
    'labels_set': args.labels_set,
    'labels': labels,
    'n_labels': len(labels),
    'batch_size': args.batch_size,
    'data_dir': cfg['data_dir'],
    'dataset': args.dataset,
    'test_file': args.test_file,
    'use_parallel': args.use_parallel,
    'num_workers': args.num_workers,
    'img_size': args.image_size,
    'print_batches': args.print_batches,
    'scratch_dir':args.scratch_dir,
    'results_dir':args.results_dir
}

# Setup
model = None
if 'DeiT' in model_args['model_state']:
    # Load DeiT-B
    torch.hub.set_dir('.'+model_args['results_dir'].replace('.','_'))
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
    model.head = nn.Sequential(nn.Linear(in_features=768, out_features=model_args['n_labels']), nn.Sigmoid())

elif 'DenseNet121' in model_args['model_state']:
  # DenseNet-121
  model = torchvision.models.densenet121(pretrained=False)
  model.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=model_args['n_labels']), nn.Sigmoid())



# Datasets
dataset_root = os.path.join(model_args['data_dir'], model_args['dataset'])
test_data = MURA.MURADataset(images_file=os.path.join(dataset_root, model_args['test_file']),
                            images_dir=os.path.join(dataset_root, 'images'),
                            image_paths_file=os.path.join(dataset_root, 'image_paths.txt'),
                            labels_file=os.path.join(dataset_root, 'labels.csv'),
                            labels=model_args['labels'],
                            transform='none',
                            op='test',
                            img_size=model_args['img_size'],
                            return_index=True)

# Get device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available() and args.use_gpus!='none':
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('Device: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))
model.load_state_dict(torch.load(model_args['model_state'], map_location=torch.device(device)))

# Data loaders
testLoader = DataLoader(test_data, batch_size=model_args['batch_size'],
                         pin_memory=True, shuffle=True,
                         num_workers=model_args['num_workers'])

# Loss functions
loss_fxn = nn.BCELoss()

if model_args['use_parallel']:
  model = nn.DataParallel(model)

# Model to device
model = model.to(device)

# Test
model.eval()

test_loss = 0
batch_counter = 0
test_ys = []
test_yhats = []
test_regions = []
test_studies = []
test_files = []

# For each batch
for x, y, study, region, file in testLoader:
    if model_args['print_batches']:
      print('Batch {}/{}'.format(batch_counter, len(testLoader)))
    batch_counter += 1
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        loss = loss_fxn(yhat, y)

        test_loss += loss.item() / len(testLoader)
        test_ys.extend(y.to('cpu').numpy().tolist())
        test_yhats.extend(yhat.to('cpu').numpy().tolist())
        test_regions.extend(region)
        test_studies.extend(study)
        test_files.extend(file)

# Save
results_path = os.path.join(model_args['results_dir'], 
model_args['model_state'][model_args['model_state'].rfind('/')+1:model_args['model_state'].rfind('_model.pt')]+'.pkl')
if os.path.exists(results_path):
  with open(results_path,'rb') as f:
    results = pickle.load(f)
else:
  results = {}

results['{}_{}'.format(model_args['dataset'],model_args['test_file'].replace('.txt',''))] = {'y':test_ys, 'yhat':test_yhats, 'region':test_regions, 'study':test_studies, 'file':test_files}
print(test_loss)
print(results.keys())
with open(results_path,'wb') as f:
    pickle.dump(results, f)
