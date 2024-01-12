#! /bin/bash

./build/main preprocess  \
 boreas \
 /data/datasets/dataset_boreas \
 --mode "voxel" \
 --ground -5.0 \
 --sphere 180.0 \
 --interval 10.0 \
 --seq boreas-2020-11-26-13-58 \
 --seq boreas-2021-01-26-10-59
