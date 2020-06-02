#!user/bin/bash
source config.sh

bash config.sh
SRC="sh/src"
CoNLL="datasets/conll05"                    # path of downloaded CoNLL-datasets
CoNLL_to_JSON="datasets/json"               #         JSONed CoNLL05-datasets
PARALLEL="datasets/parallel"                #         parallel corpus
TRAIN_PREF="${PARALLEL}/train"    # prefix of parallel train set
DEV_PREF="${PARALLEL}/dev"        #                    dev set
TEST_PREF="${PARALLEL}/test.wsj"  #                    test set
# ALIGN="datasets/data_align"
SMALL_RATE=10
SMALL="datasets/small"

### CoNLL_to_JSON ###
if [ ! -d ${CoNLL_to_JSON} ] ; then
  mkdir -p ${CoNLL_to_JSON}
  for DATA in "train" "dev" "test.wsj" "test.brown"
  do
    RAW="${CoNLL}/conll05.${DATA}.txt"
    OUT="${CoNLL_to_JSON}/conll05.${DATA}.json"
    python ${SRC}/CoNLL_to_JSON.py \
      --source_file ${RAW} \
      --output_file ${OUT} \
      --dataset_type mono \
      --src_lang "<EN>" \
      --token_type CoNLL05
  done
else
  echo "Already Existed: ${CoNLL_to_JSON}"
  ls ${CoNLL_to_JSON}
fi


### create parallel corpus ###
if [ ! -d ${PARALLEL} ] ; then
  python ${SRC}/create_parallel_corpus.py \
    --ddir ${CoNLL_to_JSON} \
    --outdir ${PARALLEL}
else
  echo "Already Existed: ${PARALLEL}"
  ls ${PARALLEL}
fi

### small set for develop ###
if [ ${OUT} = 'small' ] ; then
  DSETS=`ls ${PARALLEL} --ignore='small_*'`
  for DSET in ${DSETS[@]}
  do
    DSET=${PARALLEL}/${DSET}
    WC=`wc -l ${DSET} | awk '{print $1}'`
    DEST=${PARALLEL}/small_${DSET##*/}
    NUM=$((${WC}*${SMALL_RATE}/100))
    head -${NUM} ${DSET} > ${DEST}
  done
  TRAIN_PREF="${PARALLEL}/small_train"    # prefix of parallel train set
  DEV_PREF="${PARALLEL}/small_dev"        #                    dev set
  TEST_PREF="${PARALLEL}/small_test.wsj"  #                    test set
fi

### preprocess ###
if [ ! -f ${DATA_RAW}/test.src-tgt.src.old ] ; then

  rm -rf ${DATA_BIN} ${DATA_RAW}

  echo "### /fairseq-gec/preprocess.sh ###"
  python preprocess.py \
    --source-lang src --target-lang tgt \
    --padding-factor 1 \
    --joined-dictionary \
    --copy-ext-dict \
    --trainpref ${TRAIN_PREF} \
    --validpref ${DEV_PREF} \
    --destdir ${DATA_BIN} \
    --output-format binary \
    | tee ${OUT}/data_bin.log
    #--srcdict dicts/dict.src.txt \
    #--alignfile ${TRAIN_PREF}.src \
  echo "create log file: ${OUT}/data_bin.log"

  python preprocess.py \
    --source-lang src --target-lang tgt \
    --padding-factor 1 \
    --srcdict ${DATA_BIN}/dict.src.txt \
    --joined-dictionary \
    --copy-ext-dict \
    --testpref ${TEST_PREF} \
    --destdir ${DATA_RAW} \
    --output-format raw \
    | tee ${OUT}/data_raw.log
  echo "create log file: ${OUT}/data_raw.log"

  mv ${DATA_RAW}/test.src-tgt.src ${DATA_RAW}/test.src-tgt.src.old


  #echo "### /fairseq-gec/align.sh ###"
  #mkdir -p ${ALIGN}
  #python scripts/build_sym_alignment.py \
  #  --fast_align_dir ~/software/fast_align/build/ \
  #  --mosesdecoder_dir fakkk \
  #  --source_file ${TRAIN_PREF}.src \
  #  --target_file ${TRAIN_PREF}.tgt \
  #  --output_dir data_align

  #cp ${ALIGN}/align.forward ${TRAIN_PREF}.forward
  #cp ${ALIGN}/align.backward ${TRAIN_PREF}.backward

else
  echo "Already Existed: ${DATA_BIN}"
  ls ${DATA_BIN}
fi

echo ""
echo "=== DONE! ==="
echo "preprocessed against dataset ... ${OUT}/"
echo "next ..."
echo "$ bash run.sh -g [GPU_ID] -m [MODE]"
