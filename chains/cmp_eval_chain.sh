#!/bin/bash
# for the RAG head-to-head: our retrievers evaluated on EXACTLY the pools the RAG baseline used (spec 0/5000,0/6000,0/1500):
# (a) --reselect step 2000 (A's final state ≈ 15 snippets), (b) the accumulating retriever ab_01ret step 3000 (3 rounds).
cd /home/wii/Projects/certain
sleep 2
O=/media/wii/bigdata/ab_cmp; mkdir -p $O/reselect $O/accum
C="--reread --tags --cl 256 --overlap 50 --articles --corpus triviaqa --frozen-index --think 4 --mh-lr 1e-4 --a-lora 64 --ce-w 0 --mix musique:0.3,hotpot:0.2,triviaqa:0.3 --fair-pools triviaqa:0/5000,musique:0/6000,hotpot:0/1500 --mh-batch 1 --mh-steps 0 --mh-eval 1000:100 --mh-eval-big 4000:40"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-rwkv7/bin/python scripts/train_ab_multihop.py $C --reselect --resume4 /media/wii/bigdata/ab_reselect/mh_step2000.pt --out4 $O/reselect >> $O/reselect/eval.log 2>&1
.venv-rwkv7/bin/python scripts/train_ab_multihop.py $C --resume4 /media/wii/bigdata/ab_01ret/mh_step3000.pt --out4 $O/accum >> $O/accum/eval.log 2>&1
echo "$(date +%T) CMP DONE" >> $O/chain.log
