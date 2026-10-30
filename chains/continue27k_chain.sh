#!/bin/bash
# 2026-10-30 (user: "when queue finishes, continue with training using the rest from 27k dataset samples, with qwen 1.7B
# and mmbert small"; "add these missing follow-up questions before continue27k"; "log ce tok etc … training curve"):
# the 9k models CONTINUE on train3 (the other 18,000 rows; fresh schedule), in parallel, with frequent val evals:
#   10% Qwen3-1.7B chopper (eval every 1500) | mmBERT-small chopper (eval every 1500) | Qwen3-1.7B answerer (eval every 4500)
# then character-metric comparisons (keeper eval-only with each mmBERT-small's preds as the baseline line)
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm; K=/media/wii/bigdata/keeper
sleep 60; while systemctl --user is-active --quiet chopper_small || systemctl --user is-active --quiet chopper27k || systemctl --user is-active --quiet keeper || systemctl --user is-active --quiet subq3; do sleep 60; done
.venv-rwkv7/bin/python scripts/keeper_train.py $O $K/perms_q17.pt $K/w10_q17_27k --frac 0.1 --train-splits train3 --init $K/w10_q17/keeper.pt --eval-every 1500 --baseline $O/chopper27k_shape/preds_val.jsonl > $K/w10_q17_27k.out 2>&1 &
P1=$!
( .venv-gen/bin/python scripts/mm_train.py $O $O/chopper_small27k --base jhu-clsp/mmBERT-small --train-splits train3 --init $O/chopper_small9k/reader.pt --accum 4 --eval-every 1500 > $O/chopper_small27k.out 2>&1 && \
  .venv-gen/bin/python scripts/mm_train.py $O $O/chopper_small27k_shape --base jhu-clsp/mmBERT-small --eval-only $O/chopper_small27k/reader.pt --pred-thr 0.8 --merge-gap 5 --min-run 4 > $O/chopper_small27k_shape.out 2>&1 ) || echo "$(date +%T) SMALL27K FAILED" >> $O/chain.log &
P2=$!
if [ -s $O/subq_train3.jsonl ]; then
  TRAIN_SPLITS=train3 .venv-rwkv7/bin/python scripts/mm_qwen_answer.py $O $O/reader2_shape/preds_val.jsonl $O/qans17_27k --subq $O/subq_train3.jsonl --init $O/qans17/lora --eval-every 4500 > $O/qans17_27k.out 2>&1 || echo "$(date +%T) ANSWERER27K FAILED" >> $O/chain.log
fi
wait $P1 || echo "$(date +%T) KEEPER27K FAILED" >> $K/chain.log
wait $P2
.venv-rwkv7/bin/python scripts/keeper_train.py $O $K/perms_q17.pt $K/cmp_9k --eval-only $K/w10_q17/keeper.pt --baseline $O/chopper_small9k_shape/preds_val.jsonl > $K/cmp_9k.out 2>&1
.venv-rwkv7/bin/python scripts/keeper_train.py $O $K/perms_q17.pt $K/cmp_27k --eval-only $K/w10_q17_27k/keeper.pt --baseline $O/chopper_small27k_shape/preds_val.jsonl > $K/cmp_27k.out 2>&1
echo "$(date +%T) CONTINUE27K DONE" >> $K/chain.log
