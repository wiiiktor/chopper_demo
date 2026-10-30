#!/bin/bash
# 2026-10-29 (user: "switch to mmbert with conversation tail and top 16 snippets, and decisions search / answer"):
# pools as text → clean top-16 contexts (Qwen3-Embedding + bge-reranker) → Qwen3-14B fragment labels (char spans)
cd /home/wii/Projects/certain
O=/media/wii/bigdata/mm
while systemctl --user is-active --quiet mm_dump; do sleep 20; done
grep -q "DUMP DONE" $O/dump_val.log || { echo "$(date +%T) DUMP FAILED" >> $O/chain.log; exit 1; }
for S in val train; do
  .venv-gen/bin/python scripts/mm_contexts.py $O $S > $O/ctx_$S.log 2>&1 || { echo "$(date +%T) CTX $S FAILED" >> $O/chain.log; exit 1; }
  echo "$(date +%T) ctx $S done" >> $O/chain.log
done
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/.venv-vllm/bin:$PATH
for S in val train; do
  .venv-vllm/bin/python scripts/mm_qwen_labels.py $O/ctx_$S.jsonl $O/qw_$S.jsonl > $O/qw_$S.log 2>&1 || { echo "$(date +%T) QWEN $S FAILED" >> $O/chain.log; exit 1; }
  echo "$(date +%T) qwen $S done" >> $O/chain.log
done
echo "$(date +%T) MM DATA DONE" >> $O/chain.log
