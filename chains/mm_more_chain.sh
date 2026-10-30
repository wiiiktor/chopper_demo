#!/bin/bash
# 2026-10-29 overnight follow-up: the bottleneck is the reader's fragment quality ⇒ 6,000 MORE train questions
# (next rows of each permutation) → contexts → Qwen3-14B fragments → mmBERT reader 2 on 9,000 → shaped preds →
# search step again
cd /home/wii/Projects/certain
O=/media/wii/bigdata/mm
# (labeller comparison finished 07:55; the first try of this chain failed on a variable named TAG shadowing the trainer)
F="--reread --tags --cl 256 --overlap 50 --articles --corpus triviaqa --frozen-index --think 4 --a-lora 64 --ce-w 0 --reselect --slot-merge rank --mix musique:0.3,hotpot:0.2,triviaqa:0.3 --fair-pools triviaqa:6000/5000,musique:4000/6000,hotpot:2000/1500 --mh-batch 1"
FAIR_TOKS_ONLY=1 .venv-rwkv7/bin/python scripts/mm_dump_pools.py $O $F --split train --tag train2 --start musique:1200,hotpot:800,triviaqa:1000 --n musique:2400,hotpot:1600,triviaqa:2000 > $O/dump_train2.log 2>&1 || { echo "$(date +%T) DUMP2 FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_contexts.py $O train2 > $O/ctx_train2.log 2>&1 || { echo "$(date +%T) CTX2 FAILED" >> $O/chain.log; exit 1; }
echo "$(date +%T) ctx train2 done" >> $O/chain.log
VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python scripts/mm_qwen_labels.py $O/ctx_train2.jsonl $O/qw_train2.jsonl > $O/qw_train2.log 2>&1 || { echo "$(date +%T) QWEN2 FAILED" >> $O/chain.log; exit 1; }
echo "$(date +%T) qwen train2 done" >> $O/chain.log
.venv-gen/bin/python scripts/mm_train.py $O $O/reader2 --train-splits train,train2 --accum 4 --eval-every 3000 > $O/reader2.out 2>&1 || { echo "$(date +%T) READER2 FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_train.py $O $O/reader2_shape --eval-only $O/reader2/reader.pt --pred-thr 0.8 --merge-gap 5 --min-run 4 > $O/reader2_shape.out 2>&1
.venv-gen/bin/python scripts/mm_search_loop.py $O $O/reader2_shape/preds_val.jsonl > $O/search2.log 2>&1
echo "$(date +%T) MORE CHAIN DONE" >> $O/chain.log
