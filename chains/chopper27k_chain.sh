#!/bin/bash
# 2026-10-30 (user: "at the end, train also the mmbert chopper on the full dataset"): after the keeper chain, mmBERT chopper
# on all 27,000 labelled contexts (train+train2+train3; evals at 9k/18k/27k), shaped decode, search step
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm; C=$O/chopper27k
sleep 120; while systemctl --user is-active --quiet keeper || systemctl --user is-active --quiet mm_more2 || systemctl --user is-active --quiet mm_subq17; do sleep 60; done
[ "$(wc -l < $O/qw_train3.jsonl)" = 18000 ] || { echo "$(date +%T) CHOPPER27K: qw_train3 incomplete" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_train.py $O $C --train-splits train,train2,train3 --accum 4 --eval-every 9000 > $C.out 2>&1 || { echo "$(date +%T) CHOPPER27K FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_train.py $O ${C}_shape --eval-only $C/reader.pt --pred-thr 0.8 --merge-gap 5 --min-run 4 > ${C}_shape.out 2>&1
.venv-gen/bin/python scripts/mm_search_loop.py $O ${C}_shape/preds_val.jsonl > $O/search_chopper27k.log 2>&1
echo "$(date +%T) CHOPPER27K CHAIN DONE" >> $O/chain.log
