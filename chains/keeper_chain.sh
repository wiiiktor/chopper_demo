#!/bin/bash
# 2026-10-29 (user: "start working on the 10% width concept (only, no depth, no recompute) with small qwen"):
# after the train3 labels free the GPU: score + rewire Qwen3-1.7B on 128 train contexts (with the exactness checks),
# then train the 10% width keeper on train+train2 (9k, = mmBERT reader 2's data) and evaluate against reader 2
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm; K=/media/wii/bigdata/keeper; mkdir -p $K
while ! grep -q "mm_more2 stopped\|QWEN3 FAILED" $O/chain.log || systemctl --user is-active --quiet mm_more2; do sleep 60; done
.venv-rwkv7/bin/python scripts/keeper_rewire.py Qwen/Qwen3-1.7B $O/ctx_train.jsonl $K/perms_q17.pt --n 128 > $K/rewire_q17.log 2>&1 || { echo "$(date +%T) REWIRE FAILED" >> $K/chain.log; exit 1; }
echo "$(date +%T) rewire done" >> $K/chain.log
.venv-rwkv7/bin/python scripts/keeper_train.py $O $K/perms_q17.pt $K/w10_q17 --frac 0.1 --train-splits train,train2 > $K/w10_q17.out 2>&1 || { echo "$(date +%T) TRAIN FAILED" >> $K/chain.log; exit 1; }
echo "$(date +%T) KEEPER CHAIN DONE" >> $K/chain.log
