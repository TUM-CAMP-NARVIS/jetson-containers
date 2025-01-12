#!/bin/bash

export CUDA_VERSION=12.2
export PYTHON_VERSION=3.11
export PIP_INDEX_URL=http://dev-pi.artekmed.narvis.lan
export PIP_UPLOAD_REPO=http://dev-pi.artekmed.narvis.lan/jp5/cu122
export PIP_UPLOAD_USER=jp5

./jetson-containers run --name ollama ollama:r35.6.0-cu114

