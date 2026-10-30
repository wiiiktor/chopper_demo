#!/bin/bash
# 2026-10-29: search step with the Qwen3-1.7B answerer's OWN sub-questions (vs Qwen3-14B's .87); waits for the train3 labels to free the GPU
cd /home/wii/Projects/certain; O=/media/wii/bigdata/mm
while ! grep -q "qwen train3 done\|QWEN3 FAILED" $O/chain.log; do sleep 60; done
for n in reader qwen; do
  .venv-gen/bin/python scripts/mm_search_loop.py $O $O/reader2_shape/preds_val.jsonl --subq $O/qans17/subq17_$n.jsonl > $O/search_subq17_$n.log 2>&1
done
echo "$(date +%T) SUBQ17 SEARCH DONE" >> $O/chain.log
