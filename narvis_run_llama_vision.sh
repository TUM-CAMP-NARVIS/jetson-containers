#!/bin/bash

export CUDA_VERSION=11.4
export PTYHON_VERSION=3.10

./jetson-containers run -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN $(autotag llama-vision) python3 /opt/llama_vision.py --model "meta-llama/Llama-3.2-11B-Vision" --image "/data/images/hoover.jpg" --prompt "I'm out in the" --max-new-tokens 32 --interactive
