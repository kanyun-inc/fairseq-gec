#!/usr/bin/bash

device=0
if [ $# -ge 1 ]; then
    device=$1
fi

exp=''
if [ $# -ge 2 ]; then
    exp=$2
fi

DATA='datasets/preprocessed/conll05' # input dir
OUT='srl' # output dir

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models$exp
RESULT=$OUT/result$exp
mkdir -p $MODELS
mkdir -p $RESULT
