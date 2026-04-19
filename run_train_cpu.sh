#!/bin/bash

set -e

source venv/bin/activate

python train.py \
    --timesteps 100000 \
    --n-envs 4 \
    --lr 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ent-coef 0.01 \
    --log-dir ./logs \
    --model-dir ./models \
    --save-freq 10000 \
    "$@"
