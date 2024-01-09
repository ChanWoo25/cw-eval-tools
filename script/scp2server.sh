#! /bin/bash

###################
# Template (file) #
###################
export LOC_SRC_PATH="~"
export REM_DST_PATH="~"
scp -P 48184 ${LOC_SRC_PATH} chanwoo@166.104.112.56:${REM_DST_PATH}

########################
# Template (directory) #
########################
export LOC_SRC_DIR="~"
export REM_DST_DIR="~"
scp -P 48184 -r ${LOC_SRC_DIR} chanwoo@166.104.112.56:${REM_DST_DIR}

######################
# 2023-12-18 / 15:20 #
######################
# export SCP_SRC_DIR="/data2/chanwoo/datasets/dataset_cs_campus/benchmark_datasets"
# export SCP_DST_DIR="/data/datasets"
# scp -P 48184 -r chanwoo@166.104.112.56:${SCP_SRC_DIR} ${SCP_DST_DIR}

