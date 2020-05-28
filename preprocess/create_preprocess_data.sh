#!user/bin/bash
source config.sh

bash config.sh
SRC="preprocess/src"
CoNLL="datasets/conll05"                    # path of downloaded CoNLL-datasets
CoNLL_to_JSON="datasets/json"               #         JSONed CoNLL05-datasets
PARALLEL="datasets/parallel"                #         parallel corpus
TRAIN_PREF="${PARALLEL}/train"    # prefix of parallel train set
DEV_PREF="${PARALLEL}/dev"        #                    dev set
TEST_PREF="${PARALLEL}/test.wsj"  #                    test set
# ALIGN="datasets/data_align"

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


### preprocess ###
if [ ! -f ${DATA_BIN}/hoge ] ; then
  #mkdir -p ${DATA_BIN}
  #fairseq-preprocess --source-lang src --target-lang tgt \
  #  --trainpref ${TRAIN_PREF} \
  #  --validpref ${DEV_PREF} \
  #  --testpref ${TEST_PREF} \
  #  --destdir ${OUT}  #SRL-S2S/conll05

  cp datasets/preprocessed/conll05/dict* dicts/
  rm -f ${DATA_BIN}/dict

  echo "### /fairseq-gec/preprocess.sh ###"
  python preprocess.py \
    --source-lang src --target-lang tgt \
    --srcdict dicts/dict.src.txt \
    --padding-factor 1 \
    --joined-dictionary \
    --copy-ext-dict \
    --trainpref ${TRAIN_PREF} \
    --validpref ${DEV_PREF} \
    --destdir ${DATA_BIN} \
    --output-format binary \
    | tee ${OUT}/data_bin.log
    #--alignfile ${TRAIN_PREF}.src \
  echo "create log file: ${OUT}/data_bin.log"

  python preprocess.py \
    --source-lang src --target-lang tgt \
    --padding-factor 1 \
    --srcdict dicts/dict.src.srl.txt \
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

echo "=== DONE! ==="
echo "preprocessed against dataset ... ${OUT}/"
echo "next ..."
echo "$ bash run.sh -g [GPU_ID] -m [MODE]"
