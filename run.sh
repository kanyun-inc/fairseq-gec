#!/usr/bin/bash -eu

USAGE="Usage: bash run.sh -g [GPU_ID] -m [MODE]"

while getopts g:m: OPT
do
  case ${OPT} in
    "g" ) FLG_G="TRUE"; GPU_ID=${OPTARG};;
    "m" ) FLG_M="TRUE"; MODE=${OPTARG};;
    * ) echo ${USAGE} 1>&2
        exit 1 ;;
  esac
done

if test "${GPU_ID}" != "TRUE"; then
  GPU_ID=0
elif test "${MODE}" != "TRUE"; then
  echo "SELECT MODE ... train/test/both"
  echo ${USAGE} 1>&2
  exit 1
fi


# run ==============================================

bash config.sh

cat <<EOS > /tmp/work.txt

GPU_ID=${GPU_ID}

ARCH=transformer
SEED=0
EPOCH=20
BATCH=16
M_TOKENS=3000
LOG_INTVL=1000
PRETRAIN=false

LR=0.001
LR_SCHEDULER=triangular
M_LR=0.004
LR_UPDATE=73328
LR_SHRINK=0.95
CLIP=2

DOUT=0.2
RELU_DOUT=0.2
ATT_DOUT=0.2
CP_ATT_DOUT=0.2

ENC_EDIM=512
M_SRC_POS=1024
ENC_FFN_EDIM=4096
ENC_ATT_HEADS=8

DEC_EDIM=512
M_TGT_POS=1024
DEC_FFN_EDIM=4096
DEC_ATT_HEADS=8

CP_ATT_HEADS=1
POS_LBL_W=1.2

EOS


if [ ${MODE} = 'train' ] || [ ${MODE} = 'both' ] ; then 
  bash train/srl_train.sh
  cp /tmp/work.txt ${OUT}/experiment.config
  echo 'FIN TRAIN'

else
  echo 'invalid MODE'

fi

