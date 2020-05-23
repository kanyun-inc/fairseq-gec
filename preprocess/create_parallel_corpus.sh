#!user/bin/bash

SRC="preprocess/src"
CoNLL="datasets/conll05"          # path of downloaded CoNLL-datasets
CoNLL_to_JSON="datasets/json"     #         JSONed CoNLL05-datasets
PARALLEL="datasets/parallel"      #         parallel corpus
FAIRDDIR="SRL-S2S/data/conll05"   #         preprocessed data for using fairseq

TRAIN_PREF="${PARALLEL}/train"    # prefix of parallel train set
DEV_PREF="${PARALLEL}/dev"        #                    dev set
TEST_PREF="${PARALLEL}/test.wsj"  #                    test set


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


### fairseq preprocess ###
if [ ! -d ${FAIRDDIR} ] ; then
  mkdir -p ${FAIRDDIR}
  fairseq-preprocess --source-lang src --target-lang tgt \
      --trainpref ${TRAIN_PREF} \
      --validpref ${DEV_PREF} \
      --testpref ${TEST_PREF} \
      --destdir ${FAIRDDIR}
else
  echo "Already Existed: ${FAIRDDIR}"
  ls ${FAIRDDIR}
fi
