#!/bin/bash
# Reproduce the released universal chopper (wiiiktor/chopper-0.31B = wiiiktor/chopper_mmbert/mmbert-base) from the public data.
# Same commands and flags as the original runs (chopper_basemix54k -> chopper_fullc -> chopper_final), with the paths made relative.
#   stage 1  base mix, 54,000 rows: benchmark chains (MuSiQue, HotpotQA, TriviaQA) + Wikipedia chains     ~6 h on an RTX 3090
#   stage 2  rerank set, 54,923 rows: 8 languages, business documents, 30% "no answer" rows               ~6 h
#   stage 3  final mix, 102,654 rows: 10 sources, flat LR with a linear anneal over the last 25%          ~5 h
#   stage 4  (Polish variants only) PoQuAD + PolQA + replay of the final mix
# Another base (e.g. --base BAAI/bge-reranker-v2-m3): first fit the rows to its tokenizer with scripts/chopper/fit_splits.py.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${D:-data/chopper_demo}; OUT=${OUT:-runs}; BASE=${BASE:-jhu-clsp/mmBERT-base}; PY=${PY:-python}
SH="--pred-thr 0.5 --merge-gap 5 --min-run 4"

# data: the public dataset repo, flattened into the ctx_<split>/qw_<split> layout the trainer reads
[ -d $D ] || huggingface-cli download wiiiktor/chopper_demo --repo-type dataset --local-dir $D
mkdir -p $D/flat $OUT
for f in $D/benchmark_chains/* $D/wikipedia_chains/* $D/rerank_set/* $D/final_mix/train/* $D/final_mix/val/* $D/polish/train/* $D/polish/val/*; do
  ln -sf "$(realpath "$f")" $D/flat/
done
T="$PY scripts/chopper/mm_train.py $D/flat"

# stage 1: the 54k base mix (keep-w default 0.5)
$T $OUT/s1 --base $BASE --train-splits train,train2,train3,wtrain --accum 4 --eval-every 4500
# stage 2: the rerank set, continued from stage 1
$T $OUT/s2 --base $BASE --init $OUT/s1/chopper.pt --train-splits rrfull --val-split val --keep-w 0.15 --accum 4 --eval-every 9000
# stage 3: the final mix, continued from stage 2 (flat mix: a curriculum tied it)
$T $OUT/s3 --base $BASE --init $OUT/s2/chopper.pt --train-splits cm8k,rrfull20,rrfull20b,mqu,mhtrain2,mhret,hpq,w2,qasper,msc \
  --lr-shape flat-anneal --anneal 0.25 --val-split hs --keep-w 0.15 --accum 4 --eval-every 26000 $SH
# -> $OUT/s3/chopper.pt is the released universal chopper

# stage 4 (only for a Polish chopper)
if [ "${POLISH:-0}" = 1 ]; then
  $T $OUT/s4 --base $BASE --init $OUT/s3/chopper.pt --train-splits plpq,plqa,plreplay --val-split hs \
    --lr-shape flat-anneal --anneal 0.25 --keep-w 0.15 --accum 4 --eval-every 1000000 $SH
fi
