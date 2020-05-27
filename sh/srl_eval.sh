#!/usr/bin/env bash
# copy from generate.sh

source ./config.sh
. /tmp/work.txt

set -e

EMA=''

rm -rf ${DATA_RAW}/test.src-tgt.src
rm -rf ${DATA_RAW}/test.src-tgt.tgt


echo "### gec_scripts/split.py ###"
python gec_scripts/split.py \
  ${DATA_RAW}/test.src-tgt.src.old \
  ${DATA_RAW}/test.src-tgt.src \
  ${DATA_RAW}/test.idx

cp ${DATA_RAW}/test.src-tgt.src ${DATA_RAW}/test.src-tgt.tgt

echo "### generate.py ###"
EPOCHS=${EPOCHS}
for EPOCH in ${EPOCHS[*]}; do
    if [ -f ${RESULT}/m2score${EMA}${exp}_${EPOCH}.log ]; then
        echo "NotFound ... ${RESULT}/m2score${EMA}${exp}_${EPOCH}.log"
        continue
    fi
    echo ${EPOCH}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python generate.py ${DATA_RAW} \
    --path ${MODELS}/checkpoint${EMA}${EPOCH}.pt \
    --beam ${BEAM} \
    --nbest ${NBEST} \
    --gen-subset test \
    --max-tokens ${M_TOKENS_} \
    --raw-text \
    --batch-size ${BATCH} \
    --print-alignment \
    --max-len-a ${M_LEN_A} \
    --no-early-stop \
    --copy-ext-dict --replace-unk \
    > ${RESULT}/output${EMA}${EPOCH}.nbest.txt 

    cat ${RESULT}/output${EMA}${EPOCH}.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 ${RESULT}/output${EMA}${EPOCH}.txt.split

    echo "## gec_scripts/revert_split.py ##"
    python ./gec_scripts/revert_split.py ${RESULT}/output${EMA}${EPOCH}.txt.split $DATA_RAW/test.idx > ${RESULT}/output${EMA}${EPOCH}.txt

    python2 ./gec_scripts/m2scorer/m2scorer -v ${RESULT}/output${EMA}${EPOCH}.txt ./data/test.m2 > ${RESULT}/m2score${EMA}${exp}_${EPOCH}.log
    tail -n 1 ${RESULT}/m2score${EMA}${exp}_${EPOCH}.log
done

python gec_scripts/show_m2.py ${RESULT}/m2score${EMA}${exp}_{}.log
