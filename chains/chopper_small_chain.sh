#!/bin/bash
# 2026-10-30 (user: "comparison of 10% qwen-1.7B with mmbert 0.15B would be very interesting"): mmBERT-small (0.14B, MIT)
# chopper on the SAME 9k rows as the keeper and mmBERT-base reader 2 (train+train2), val eval every 1500 rows (curve)
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm; C=$O/chopper_small9k
sleep 60; while systemctl --user is-active --quiet chopper27k || systemctl --user is-active --quiet keeper; do sleep 60; done
.venv-gen/bin/python scripts/mm_train.py $O $C --base jhu-clsp/mmBERT-small --train-splits train,train2 --accum 4 --eval-every 1500 > $C.out 2>&1 || { echo "$(date +%T) CHOPPER-SMALL FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_train.py $O ${C}_shape --base jhu-clsp/mmBERT-small --eval-only $C/reader.pt --pred-thr 0.8 --merge-gap 5 --min-run 4 > ${C}_shape.out 2>&1
echo "$(date +%T) CHOPPER-SMALL DONE" >> $O/chain.log
