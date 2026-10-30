#!/bin/bash
# 2026-10-30: the 1.7B answerer's sub-question for EVERY val question (forced "[search]"), then the search step —
# separates its question quality from its search decision. Runs after the keeper, alongside chopper27k.
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm
sleep 60; while systemctl --user is-active --quiet keeper; do sleep 60; done
for n in reader qwen; do
  N=$([ $n = qwen ] && echo qwen || echo $O/reader2_shape/preds_val.jsonl)
  .venv-rwkv7/bin/python scripts/mm_qwen_forcesubq.py $O $O/qans17/lora $N $O/qans17/subq17force_$n.jsonl > $O/forcesubq_$n.log 2>&1 || { echo "$(date +%T) FORCESUBQ $n FAILED" >> $O/chain.log; continue; }
  .venv-gen/bin/python scripts/mm_search_loop.py $O $O/reader2_shape/preds_val.jsonl --subq $O/qans17/subq17force_$n.jsonl > $O/search_subq17force_$n.log 2>&1
done
echo "$(date +%T) FORCESUBQ DONE" >> $O/chain.log
