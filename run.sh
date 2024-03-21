#!/bin/bash

# train

#python3 train.py --model ae --n_epochs 500 --name ae --dataset cifar
#python3 train.py --model ae --n_epochs 500 --name ae_noisy --noise --dataset cifar
#python3 train.py --model conv --n_epochs 500 --name conv --dataset cifar
#python3 train.py --model conv --n_epochs 500 --name conv_noisy --noise --dataset cifar

#python3 train.py --model conv --n_epochs 500 --name conv --dataset cifar
#python3 test.py --model conv --name conv --dataset cifar --num_samples 10

python3 train.py --model vae --n_epochs 750 --name vae_long  --dataset cifar
python3 train.py --model vae --n_epochs 250 --name vae_short --dataset cifar

git add .
git commit -m "results $(date)"
git push

echo "Done"
