#!/bin/bash

# train

python3 train.py --model ae --n_epochs 500 --name ae --dataset cifar --batch_size 512
python3 train.py --model conv --n_epochs 500 --name conv --dataset cifar --batch_size 512

#python3 train.py --model conv --n_epochs 500 --name conv --dataset cifar
#python3 test.py --model conv --name conv --dataset cifar --num_samples 10

#python3 train.py --model vae --n_epochs 1000 --name vae_long  --dataset cifar --batch_size 512
#python3 train.py --model vae --n_epochs 100 --name vae_short --dataset cifar --batch_size 256

#python3 test.py --model vae --name vae_long --dataset cifar --num_samples 10
#python3 test.py --model vae --name vae_short --dataset cifar --num_samples 10

git add .
git commit -m "results $(date)"
git push

echo "Done"
