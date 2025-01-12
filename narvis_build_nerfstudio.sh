#!/bin/bash

export CUDA_VERSION=11.8
export PTYHON_VERSION=3.12
export PIP_INDEX_URL=http://dev-pi.artekmed.narvis.lan
export PIP_UPLOAD_REPO=http://dev-pi.artekmed.narvis.lan/jp5/cu118
export PIP_UPLOAD_USER=jp5

jetson-containers build nerfstudio --build-args "FORCE_BUILD:on" --skip-tests bitsandbytes