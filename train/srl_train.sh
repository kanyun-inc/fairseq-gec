#!/usr/bin/bash

source config.sh
. /tmp/work.txt


if "${PRETRAIN}" ; then
  echo "TRAIN w/ pretrained model"
  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python train.py ${DATA_BIN} \
  --source-lang src --target-lang tgt \
  --save-dir ${MODELS} \
  --seed ${SEED} \
  --max-epoch ${EPOCH} \
  --batch-size ${BATCH} \
  --max-tokens ${M_TOKENS} \
  --train-subset train \
  --valid-subset valid \
  --arch ${ARCH} \
  --lr-scheduler ${LR_SCHEDULER} --max-lr ${M_LR} --lr-period-updates ${LR_UPDATE} \
  --clip-norm ${CLIP} --lr ${LR} --lr-shrink ${LR_SHRINK} --shrink-min \
  --dropout ${DOUT} --relu-dropout ${RELU_DOUT} \
  --attention-dropout ${ATT_DOUT} --copy-attention-dropout ${CP_ATT_DOUT} \
  --encoder-embed-dim ${ENC_EDIM} --decoder-embed-dim ${DEC_EDIM} \
  --max-target-positions ${M_TGT_POS} --max-source-positions ${M_SRC_POS} \
  --encoder-ffn-embed-dim ${ENC_FFN_EDIM} --decoder-ffn-embed-dim ${DEC_FFN_EDIM} \
  --encoder-attention-heads ${ENC_ATT_HEADS} --decoder-attention-heads ${DEC_ATT_HEADS} \
  --copy-attention-heads ${CP_ATT_HEADS} \
  --share-all-embeddings \
  --no-progress-bar \
  --log-interval ${LOG_INTVL} \
  --positive-label-weight ${POS_LBL_W} \
  --pretrained-model ${PRE_MODEL} \
  --copy-attention --copy-attention-heads ${CP_ATT_HEADS} > ${OUT}/log${exp}.out 2>&1 &
else
  echo "TRAIN w/o pretrained model"
  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python train.py ${DATA_BIN} \
  --source-lang src --target-lang tgt \
  --save-dir ${MODELS} \
  --seed ${SEED} \
  --max-epoch ${EPOCH} \
  --batch-size ${BATCH} \
  --max-tokens ${M_TOKENS} \
  --train-subset train \
  --valid-subset valid \
  --arch ${ARCH} \
  --lr-scheduler ${LR_SCHEDULER} --max-lr ${M_LR} --lr-period-updates ${LR_UPDATE} \
  --clip-norm ${CLIP} --lr ${LR} --lr-shrink ${LR_SHRINK} --shrink-min \
  --dropout ${DOUT} --relu-dropout ${RELU_DOUT} \
  --attention-dropout ${ATT_DOUT} --copy-attention-dropout ${CP_ATT_DOUT} \
  --encoder-embed-dim ${ENC_EDIM} --decoder-embed-dim ${DEC_EDIM} \
  --max-target-positions ${M_TGT_POS} --max-source-positions ${M_SRC_POS} \
  --encoder-ffn-embed-dim ${ENC_FFN_EDIM} --decoder-ffn-embed-dim ${DEC_FFN_EDIM} \
  --encoder-attention-heads ${ENC_ATT_HEADS} --decoder-attention-heads ${DEC_ATT_HEADS} \
  --copy-attention-heads ${CP_ATT_HEADS} \
  --share-all-embeddings \
  --no-progress-bar \
  --log-interval ${LOG_INTVL} \
  --positive-label-weight ${POS_LBL_W} \
  --copy-attention --copy-attention-heads ${CP_ATT_HEADS} > ${OUT}/log${exp}.out 2>&1 &
fi


tail -f ${OUT}/log${exp}.out
