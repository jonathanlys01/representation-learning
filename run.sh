#!/bin/bash

# train

python3 train.py --model ae --n_epochs 500 --name ae --dataset cifar
python3 train.py --model ae --n_epochs 500 --name ae_noisy --noise --dataset cifar

python3 train.py --model conv --n_epochs 500 --name conv --dataset cifar
python3 train.py --model conv --n_epochs 500 --name conv_noisy --noise --dataset cifar

echo "Done"
echo $(date)