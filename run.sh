#!/bin/bash

# train

python3 train.py --model ae --n_epochs 100 --name ae
python3 train.py --model ae --n_epochs 100 --name ae_noisy --noise

python3 train.py --model conv --n_epochs 100 --name conv
python3 train.py --model conv --n_epochs 100 --name conv_noisy --noise

python3 train.py --model vae --n_epochs 100 --name vae

python3 train.py --model vae --n_epochs 1000 --name vae_long

# test

python3 test.py --model ae --name ae
python3 test.py --model ae --name ae_noisy

python3 test.py --model conv --name conv
python3 test.py --model conv --name conv_noisy

python3 test.py --model vae --name vae
python3 test.py --model vae --name vae_long

echo "Done"
echo $(date)