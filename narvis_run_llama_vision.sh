#!/bin/bash

export CUDA_VERSION=11.8
export PTYHON_VERSION=3.12
export PIP_INDEX_URL=http://dev-pi.artekmed.narvis.lan
export PIP_UPLOAD_REPO=http://dev-pi.artekmed.narvis.lan/jp5/cu118
export PIP_UPLOAD_USER=jp5

./jetson-containers run -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN $(autotag llama-vision) python3 /opt/llama_vision.py --model "meta-llama/Llama-3.2-11B-Vision" --image "/data/images/hoover.jpg" --prompt "I'm out in the" --max-new-tokens 32 --interactive
