#!/bin/bash

source py3/bin/activate

lr=(0.01 0.05 0.005)
wd=(0 0.00001 0.0001 0.001)
mu=(y n)
dr=(0 0.5)
gpu=1
nvidia-smi
for l in "${lr[@]}"
do
  for w in "${wd[@]}"
  do
    for m in "${mu[@]}"
    do
      for d in "${dr[@]}"
      do
        python train_mura.py --architecture DeiT-Ti --initial-lr $l --optimizer-family SGD --weight-decay $w --norm-layer layer --dropout $d --use-mixup $m --results-dir /export/gaon1/data/zmurphy/transformer-cxr/results/revision/mura --use-gpus $gpu
      done
    done
  done
done