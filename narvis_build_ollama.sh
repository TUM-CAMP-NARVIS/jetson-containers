#!/bin/bash

export CUDA_VERSION=11.4
export PTYHON_VERSION=3.10

jetson-containers build ollama --build-args "FORCE_BUILD:on"
