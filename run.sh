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

if [ ${MODE} = 'train' ] ; then
  echo -en "use pretrain --- [y?] "
  read PRETRAIN
  if [ ${PRETRAIN} = 'y' ] ; then
    PRETRAIN=true
  else
    PRETRAIN=false
  fi
fi

# run ==============================================

bash config.sh

cat <<EOS > /tmp/work.txt

GPU_ID=${GPU_ID}

ARCH=transformer
SEED=0
EPOCH=9
BATCH=8
M_TOKENS=3000
LOG_INTVL=1000
PRETRAIN=${PRETRAIN}
PRE_MODEL=srl/models_pretrain/checkpoint_last.pt

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


### train ###
if [ ${MODE} = 'train' ] || [ ${MODE} = 'both' ] ; then 
  bash sh/srl_train.sh
  cp /tmp/work.txt ${OUT}/experiment.config
  echo 'FIN TRAIN'

### evaluate ###
elif [ ${MODE} = 'test' ] || [ ${MODE} = 'both' ] ; then 

  cat <<EOS >> /tmp/work.txt
BEAM=5
NBEST=5
EPOCHS=_last
M_TOKENS_=6000
M_LEN_A=0
EOS

  bash sh/srl_eval.sh
  echo 'FIN TEST'

elif [ ! ${MODE} = 'train' ] ; then
  echo 'invalid MODE'

fi
