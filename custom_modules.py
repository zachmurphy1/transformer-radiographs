# Standard
import os, sys, shutil
import pandas as pd
import numpy as np
import random

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

# Image handling
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Special
from timm.data import create_transform

"""Custom transforms"""
def mixup(x, y, mixup_settings, epoch, batch_num, device):
  if np.random.uniform(0, 1) <= mixup_settings['mc_prob']:
    if mixup_settings['cutmix_thresh'][epoch][batch_num] <= mixup_settings['cutmix_prob']:
      # Cutmix pathway
      lam, x1, x2, y1, y2 = mixup_settings['cutmix_points'][epoch][batch_num]
      x[:, :, x1:x2, y1:y2] = x.flip(0)[:, :, x1:x2, y1:y2]
      y = y * lam + y.flip(0) * (1 - lam)

      return x, y

    else:
      # Mixup pathway
      lam = np.random.beta(mixup_settings['mixup_alpha'], mixup_settings['mixup_alpha'])
      x = x * lam + x.flip(0) * (1 - lam)
      y = y * lam + y.flip(0) * (1 - lam)

      return x, y
  else:
    return x, y


def getCutmixPoints(mixup_settings):
  lam = np.clip(np.random.beta(mixup_settings['cutmix_alpha'], mixup_settings['cutmix_alpha']), 0.3, 0.4)

  W = 224
  H = 224
  cut_ratio = np.sqrt(1 - lam)
  cut_w = np.int(W * cut_ratio)
  cut_h = np.int(H * cut_ratio)

  cx = np.random.randint(W)
  cy = np.random.randint(H)

  x1 = np.clip(cx - cut_w // 2, 0, W)
  x2 = np.clip(cx + cut_w // 2, 0, W)

  y1 = np.clip(cy - cut_h // 2, 0, H)
  y2 = np.clip(cy + cut_h // 2, 0, H)

  return (lam, x1, x2, y1, y2)

"""Models"""
def freeze_model(model):
  for p in model.parameters():
      p.requires_grad = False
  for p in model.head.parameters():
    p.requires_grad = True
    
  return model

def unfreeze_model(model):
  for p in model.parameters():
      p.requires_grad = True
      
  return model

def DeiT_B(model_args):
  # Load DeiT-B
  torch.hub.set_dir(model_args['results_dir'].replace('.','_'))
  model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=model_args['pretrained'])
  model.head = nn.Sequential(nn.Linear(in_features=768, out_features=model_args['n_labels']), nn.Sigmoid())

  # # Freeze all but head
  # if model_args['frozen']:
  #   model = freeze_model(model)
  # else:
  #   model = unfreeze_model(model)
  
  return model

def DeiT_Ti(model_args):
  # Load DeiT-Ti
  torch.hub.set_dir(model_args['results_dir'].replace('.','_'))
  model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=model_args['pretrained'])
  model.head = nn.Sequential(nn.Linear(in_features=192, out_features=model_args['n_labels']), nn.Sigmoid())

  # # Freeze all but head
  # if model_args['frozen']:
  #   model = freeze_model(model)
  # else:
  #   model = unfreeze_model(model)
  
  return model

def DenseNet121(model_args):
  # DenseNet-121
  model = torchvision.models.densenet121(pretrained=model_args['pretrained'])
  model.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=model_args['n_labels']), nn.Sigmoid())

  # # Freeze all but head
  # if model_args['frozen']:
  #   model = freeze_model(model)
  # else:
  #   model = unfreeze_model(model)
    
  return model

def ResNet152(model_args):
  # ResNet
  model = torchvision.models.resnet152(pretrained=model_args['pretrained'])
  model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=model_args['n_labels']), nn.Sigmoid())

  # # Freeze all but head
  # if model_args['frozen']:
  #   model = freeze_model(model)
  # else:
  #   model = unfreeze_model(model)
    
  return model

def EfficientNet_B7(model_args):
  # EfficientNet
  model = torchvision.models.efficientnet_b7(pretrained=model_args['pretrained'])
  model.classifier = nn.Sequential(nn.Linear(in_features=2560, out_features=model_args['n_labels']), nn.Sigmoid())

  # # Freeze all but head
  # if model_args['frozen']:
  #   model = freeze_model(model)
  # else:
  #   model = unfreeze_model(model)
    
  return model

def get_model(model_args):
  if model_args['model_type'] == 'DeiT-B':
    return DeiT_B(model_args)
  elif model_args['model_type'] == 'DeiT-Ti':
    return DeiT_Ti(model_args)
  elif model_args['model_type'] == 'DenseNet121':
    return DenseNet121(model_args)
  elif model_args['model_type'] == 'ResNet152':
    return ResNet152(model_args)
  elif model_args['model_type'] == 'EfficientNet_B7':
    return EfficientNet_B7(model_args)



"""CXR"""
class CXRDataset(Dataset):
  def __init__(self,
               images_list=None, # txt file with each image name on a separate line
               dataset=None, # dataset name, eg nihcxr14
               images_dir=None, # path to dir containing images
               image_paths=None, # txt file containing image paths from images_dir
               local_img_dir='images',
               labels_file=None, # csv file containing binary labels for each image
               labels=[], # list of labels
               transform=None, # transforms to apply
               op='train', # train/val/test
               img_size=224, # image size
              ):
    # Set attributes
    self.images_list = images_list
    self.images_dir = images_dir
    self.labels_file = labels_file
    self.labels = labels
    self.op = op
    self.img_size = img_size

    print('\n{} set: starting load'.format(self.op.capitalize()))

    # Get image paths
    all_files=[]
    count = 0
    if image_paths == None: # If no paths file, get image paths in images_dir
      for r, d, f in os.walk(images_dir):
        for filed in f:
          if ('.png' in filed) | ('.jpg' in filed):
            all_files.append(os.path.join(r, filed))
    else: # If paths file, use it
      print('Using image path file')
      with open(os.path.join(dataset, image_paths), 'r') as f:
        for i in f.readlines():
          all_files.append(i.strip('\n'))

    # Get list of images to include
    self.image_data = pd.read_table(images_list, header=None)
    self.image_data.columns = ['file']

    # Link full path for each image
    source_path = pd.DataFrame.from_dict({'path': all_files})
    source_path['file'] = source_path['path'].apply(lambda x: x[x.rfind('/')+1:])
    self.image_data = pd.merge(self.image_data, source_path, how='left', on='file')
    self.image_data['path'] = self.image_data['path'].apply(lambda x: os.path.join(images_dir,x))

    # Get labels and merge into image data df
    label_data = pd.read_csv(self.labels_file)
    self.image_data = pd.merge(self.image_data, label_data, left_on='file', right_on='Image').drop(columns=['Image'])
    self.image_data.set_index('file', inplace=True)

    # Set transforms
    if transform == 'hflip': # hflips only
      print('Using hflip transform')
      transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    elif transform == 'deit': # DeiT auto-augment
      print('Using deit transforms')
      transform = create_transform(
        input_size=self.img_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
      )
    else: # No transforms
      print('Using no transforms')
      transform = transforms.Compose([transforms.ToTensor()])

    # Combine transforms with standard transforms
    self.tfms = transforms.Compose([transform,
                    transforms.Resize(self.img_size, transforms.functional.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.img_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    print('Loaded {} images'.format(self.image_data.shape[0]))

  def __len__(self):
    return self.image_data.shape[0]

  # Gets tensor vector of labels for given image file name
  def getLabel(self, image_file):
    label = torch.zeros(len(self.labels))
    for i, l in enumerate(self.labels):
      if l in self.image_data.columns[1:]:
        label[i] = self.image_data.loc[image_file,l]
    return label

  def __getitem__(self, idx):

    # Get image
    img = Image.open(self.image_data.iloc[idx]['path']).convert('RGB')

    # Image transforms
    img = self.tfms(img)

    # Get label
    label = self.getLabel(self.image_data.index[idx])

    # Return image and label
    if self.op == 'test':
      return img, label, self.image_data.iloc[idx]['path']
    return img, label
  
"""MURA"""
class MURADataset(Dataset):
  def __init__(self,
         images_file=None,  # txt file with each image name on a separate line
         images_dir=None,  # path to dir containing images
         image_paths_file=None,  # txt file containing image paths from images_dir
         labels_file=None,  # csv file containing binary labels for each image
         labels=[],  # list of labels
         transform=None,  # transforms to apply
         op='train',  # train/val/test
         img_size=224,  # image size
         return_index=False):

    # Set attributes
    self.images_file = images_file
    self.images_dir = images_dir
    self.labels_file = labels_file
    self.labels = labels
    self.op = op
    self.img_size = img_size
    self.dataset = 'mura'
    self.return_index = return_index

    print('\n{} set: starting load'.format(self.op.capitalize()))

    # Get image paths
    all_files = []
    if image_paths_file is None:  # If no paths file, get image paths in images_dir
      for r, d, f in os.walk(images_dir):
        for filed in f:
          if ('.png' in filed) | ('.jpg' in filed):
            all_files.append(os.path.join(r, filed))
    else:  # If paths file, use it
      print('Using image path file')
      with open(os.path.join(self.dataset, image_paths_file), 'r') as f:
        for i in f.readlines():
          all_files.append(i.strip('\n'))
      # manually fix path of image_paths.txt by removing the beginning directories
      for i in range(len(all_files)):
        all_files[i] = all_files[i]

    # Get list of images to include
    self.image_data = pd.read_table(images_file, header=None)
    self.image_data.columns = ['file']

    # Link full path for each image
    source_path = pd.DataFrame.from_dict({'path': all_files})
    source_path['file'] = source_path['path'].apply(lambda x: x[x.rfind('/') + 1:])
    self.image_data = pd.merge(self.image_data, source_path, how='left', on='file')
    self.image_data['path'] = self.image_data['path'].apply(lambda x: os.path.join(images_dir, x))

    # Get labels and merge into image data df
    label_data = pd.read_csv(self.labels_file)
    self.image_data = pd.merge(self.image_data, label_data, left_on='file', right_on='Image').drop(
      columns=['Image'])
    self.image_data.set_index('file', inplace=True)

    # Set transforms
    if transform == 'hflip':  # hflips only
      print('Using hflip transform')
      transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    elif transform == 'deit':  # DeiT auto-augment
      print('Using deit transforms')
      transform = create_transform(
        input_size=self.img_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
      )
    else:  # No transforms
      print('Using no transforms')
      transform = transforms.Compose([transforms.ToTensor()])

    # Combine transforms with standard transforms
    self.tfms = transforms.Compose([transform,
                    transforms.Resize(self.img_size,
                              transforms.functional.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.img_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    print('Loaded {} images'.format(self.image_data.shape[0]))

  def __len__(self):
    return self.image_data.shape[0]

  # Gets tensor vector of labels for given image file name
  def getLabel(self, image_file):
    label = torch.zeros(len(self.labels))
    if self.image_data.loc[image_file, 'Positive'] == 1:
      label[1] = 1
    else:
      label[0] = 1
    return label
  
  # Get study id for given image index
  def getStudyName(self, idx):
    fileName = self.image_data.index[idx]
    return fileName[fileName.find("_")+1:fileName.find("_image")]

  def getRegion(self,x):
    if 'SHOULDER' in x:
      return 'shoulder'
    elif 'HUMERUS' in x:
      return 'humerus'
    elif 'ELBOW' in x:
      return 'elbow'
    elif 'FOREARM' in x:
      return 'forearm'
    elif 'WRIST' in x:
      return 'wrist'
    elif 'HAND' in x:
      return 'hand'
    elif 'FINGER' in x:
      return 'finger'

  def __getitem__(self, idx):

    # Get image
    img = Image.open(self.image_data.iloc[idx]['path']).convert('RGB')

    # Image transforms
    img = self.tfms(img)

    # Get label
    label = self.getLabel(self.image_data.index[idx])

    if self.op == 'test':
      region = self.getRegion(self.image_data.iloc[idx]['path'])
      f = self.image_data.iloc[idx]['path']
      study = f[:f.find('_image')]
      return img, label, study, region, f
    if self.return_index:
      # Return image, label, and index
      return img, label, torch.tensor([idx])
    else:
      # Return image and label
      return img, label