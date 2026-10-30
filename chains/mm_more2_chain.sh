#!/bin/bash
# 2026-10-29 (user: "we definitely need more data for reader training — queue more generation after testing 7b 13b"):
# 18,000 more train questions (next rows: MuSiQue 3,600→9,600, HotpotQA 2,400→8,400, TriviaQA 3,000→9,000) →
# contexts → Qwen3-14B fragments (~9 h) → mmBERT reader 3 on all 27,000 → shaped preds → search step
cd /home/wii/Projects/certain
O=/media/wii/bigdata/mm
sleep 120; while systemctl --user is-active --quiet mm_more || systemctl --user is-active --quiet mm_bigrwkv || systemctl --user is-active --quiet mm_subq; do sleep 60; done
F="--reread --tags --cl 256 --overlap 50 --articles --corpus triviaqa --frozen-index --think 4 --a-lora 64 --ce-w 0 --reselect --slot-merge rank --mix musique:0.3,hotpot:0.2,triviaqa:0.3 --fair-pools triviaqa:6000/5000,musique:4000/6000,hotpot:2000/1500 --mh-batch 1"
FAIR_TOKS_ONLY=1 .venv-rwkv7/bin/python scripts/mm_dump_pools.py $O $F --split train --tag train3 --start musique:3600,hotpot:2400,triviaqa:3000 --n musique:6000,hotpot:6000,triviaqa:6000 > $O/dump_train3.log 2>&1 || { echo "$(date +%T) DUMP3 FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_contexts.py $O train3 > $O/ctx_train3.log 2>&1 || { echo "$(date +%T) CTX3 FAILED" >> $O/chain.log; exit 1; }
echo "$(date +%T) ctx train3 done" >> $O/chain.log
VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python scripts/mm_qwen_labels.py $O/ctx_train3.jsonl $O/qw_train3.jsonl > $O/qw_train3.log 2>&1 || { echo "$(date +%T) QWEN3 FAILED" >> $O/chain.log; exit 1; }
echo "$(date +%T) qwen train3 done" >> $O/chain.log
.venv-gen/bin/python scripts/mm_train.py $O $O/reader3 --train-splits train,train2,train3 --accum 4 --eval-every 9000 > $O/reader3.out 2>&1 || { echo "$(date +%T) READER3 FAILED" >> $O/chain.log; exit 1; }
.venv-gen/bin/python scripts/mm_train.py $O $O/reader3_shape --eval-only $O/reader3/reader.pt --pred-thr 0.8 --merge-gap 5 --min-run 4 > $O/reader3_shape.out 2>&1
.venv-gen/bin/python scripts/mm_search_loop.py $O $O/reader3_shape/preds_val.jsonl > $O/search3.log 2>&1
echo "$(date +%T) MORE2 CHAIN DONE" >> $O/chain.log
