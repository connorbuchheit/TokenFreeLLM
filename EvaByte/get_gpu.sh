#!/bin/bash
# get_gpu.sh - Request an interactive GPU session on the gpu_test partition

salloc -p gpu_test --gres=gpu:nvidia_a100_3g.20gb:1 --cpus-per-gpu=4 --mem=24G --account=protopapas_lab -t 0-04:00