#!/bin/bash

source py3/bin/activate

lr=(0.05)
wd=(0 0.00001 0.0001 0.001)
mu=(y n)

gpu=0

for l in "${lr[@]}"
do
  for w in "${wd[@]}"
  do
    for m in "${mu[@]}"
    do
      python train_cxr.py --architecture EfficientNet_B7 --initial-lr $l --optimizer-family SGD --weight-decay $w --norm-layer batch --dropout 0.5 --use-mixup $m --results-dir /export/gaon1/data/zmurphy/transformer-cxr/results/revision/cxr --use-gpus $gpu
    done
  done
done