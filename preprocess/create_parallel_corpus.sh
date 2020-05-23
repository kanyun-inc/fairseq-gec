#!user/bin/bash

SRC="preprocess/src"
CoNLL="datasets/conll05"
CoNLL_to_JSON="datasets/json"
PARALLEL="datasets/parallel"

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

if [ ! -d ${PARALLEL} ] ; then
  python ${SRC}/create_parallel_corpus.py \
    --ddir ${CoNLL_to_JSON} \
    --outdir ${PARALLEL}
else
  echo "Already Existed: ${PARALLEL}"
  ls ${PARALLEL}
fi
