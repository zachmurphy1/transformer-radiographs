# Get data
Data should be stored within a single directory set by the data_dir key in cfg.json.

The dataset class will expect each dataset to be contained within its own directory inside data_dir. This directory should have a subdirectory named images that contains the actual images, with any level of subdirectory structure. The dataset directory (next to images) should also contain .txt's with lists of images belonging to distinct sets, i.e. train, val, test. As well, there should be a csv with the image labels indexed by image name.

The .txt's with set lists used in our study are included under image_lists.

## NIH CXR14
Citation:
Data is available at https://nihcc.app.box.com/v/ChestXray-NIHCC. Create directory nihcxr14 with subdirectory periph. From site, download Data_Entry_2017_v2020.csv (as metadata.csv), train_val_list.txt, test_list.txt into nihcxr14/periph directory. Set nihcxr14_links key in cfg.json to list of data urls from site.
Run Get_NIHCXR14.py.

## CheXnet
Citation:
Register for data access at https://stanfordmlgroup.github.io/competitions/chexpert/ (scroll to bottom). Set chexpert_link key in cfg.json to CheXpert-v1.0-small.zip url containing data. Run Get_CheXpart.py

## MIMIC
Citation: 
Sign up and get credentialled for PhysioNet at https://physionet.org/register/. Sign agreements for MIMIC-CXR-JPG at https://physionet.org/content/mimic-cxr-jpg/2.0.0/. Per site instructions, put data server as value of mimic_server in cfg.json. Create directory mimic/periph. Download mimic-cxr-2.0.0-metadata.csv.gz and mimic-cxr-2.0.0-chexpert.csv.gz, extract to mimic/periph/metadata.csv and /periph/chexpert_labels.csv respectively. Run Get_MIMIC.py.

## PadChest
Citation:
Sign up at https://bimcv.cipf.es/bimcv-projects/padchest/ by clicking "Download complete dataset." Set padchest_credentials as dict with keys webdav_hostname and webdav_login per website. Create directory padchest with subdirectory periph. Download PADCHEST_chest_x_ray_images_labels_160k_01.02.19.csv.gz into padchest/periph as metadata.csv. Run Get_PadChest.py.

## MURA
Citation:
Register for data access at https://stanfordmlgroup.github.io/competitions/mura/ (scroll to bottom). Set mura_link key in cfg.json to download target. Run Get_MURA.py.

# Train
Run train_cxr.py or train_mura.py. Model states are saved to argument results-dir, and results summary is appended to results.csv within results-dir. We found it helpful to have one results-dir for hyperparameter testing then another for the final runs. Each run is uniquely labelled by timestamp.

# Test
Run test_cxr.py or test_mura.py. These take in a model state path and get predictions on a test set, which are saved to a .pkl file containing a dict of ys, yhats, and image identifiers. 

# Bootstrap
Run bootstrap_cxr.py or bootstrap_mura.py. These take in a .pkl from the test script and bootstrap performance metrics.

# Eval
Run eval_cxr.py or eval_mura.py. The to-analyze argument takes a json file containing a list of dicts with keys "file" (path to .pkl from test script) and "name" (display name for model). This creates tables and figures of the bootstrapped results.

# Post-eval
All depend on output of eval script. 

## Perf by size
Creates figure for performance by varying training size.

## Get fractions
Outputs raw fractions for performance metrics on entire test set.

## Attention maps
Gets attention and GradCam maps for given images.

## Hidden stratification
Gets false positives and negatives for manual annnotation. Takes annotations and generates contingency tables.

## Hyperparam analysis
(Does not depend on eval script.) Generates visualizations of hyperparameter space.
