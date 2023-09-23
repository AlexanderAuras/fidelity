#!/bin/bash

set -e

srun --ntasks=1\
     --cpus-per-task=8\
     --gpus-per-task=1\
     --gres=gpu:1\
     --mem=32G\
     --partition=gpu\
     --time=24:00:00\
     --nice=0\
     --job-name=debug\
     --pty\
     /bin/bash
