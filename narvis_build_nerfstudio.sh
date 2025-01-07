#!/bin/bash

export CUDA_VERSION=11.4
export PTYHON_VERSION=3.10

jetson-containers build nerfstudio --build-args "FORCE_BUILD:on" --skip-tests bitsandbytes