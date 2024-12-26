#!/bin/bash

export CUDA_VERSION=11.4
export PTYHON_VERSION=3.10

./jetson-containers run --name ollama ollama:r35.6.0-cu114

