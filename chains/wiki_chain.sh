#!/bin/bash
# 2026-10-30 (user: "at the end, start generating 27k training data samples using wikipedia"): benchmark-free dataset B.
# wk_chains (CPU, after the download) → [after continue27k] rounds of wk_gen (Qwen3-14B-AWQ) + wk_judge (Qwen3-4B) until
# 18,000 2-hop + 9,000 1-hop train (+200/+100 val) are accepted → wk_pools → mm_contexts → mm_qwen_labels (parts of
# 4,500: it writes only at the end) → mm_subq --train. Splits are tagged wtrain / wval in /media/wii/bigdata/mm.
cd /home/wii/Projects/certain; W=/media/wii/bigdata/wiki; O=/media/wii/bigdata/mm; L=$W/chain.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/.venv-vllm/bin:$PATH
say() { echo "$(date +%T) $*" >> $L; }
while systemctl --user is-active --quiet wikidl; do sleep 60; done
if [ ! -s $W/cand.jsonl ]; then
  .venv-rwkv7/bin/python scripts/wk_chains.py $W/raw/20231101.en $W --n2 150000 --n1 30000 > $W/chains.log 2>&1 || { say "CHAINS FAILED"; exit 1; }
fi
say "chains: $(tail -1 $W/chains.log)"
sleep 60; while systemctl --user is-active --quiet continue27k; do sleep 120; done
count() { .venv-rwkv7/bin/python -c "
import json,glob,sys
c={}
for f in glob.glob('$W/jud_r*.jsonl'):
    for r in map(json.loads, open(f)):
        if r['ok']: c[(r['split'],r['type'])]=c.get((r['split'],r['type']),0)+1
print(c.get(('train','2hop'),0), c.get(('train','1hop'),0), c.get(('val','2hop'),0), c.get(('val','1hop'),0))"; }
s2=0; s1=0
for r in 1 2 3 4 5 6 7 8; do
  read t2 t1 v2 v1 <<< "$(count)"
  n2=20000; n1=6000
  [ $t2 -ge 18000 ] && [ $v2 -ge 200 ] && n2=0
  [ $t1 -ge 9000 ] && [ $v1 -ge 100 ] && n1=0
  say "round $r: accepted so far 2-hop $t2 (+$v2 val), 1-hop $t1 (+$v1 val) → generating $n2 + $n1"
  [ $n2 -eq 0 ] && [ $n1 -eq 0 ] && break
  if [ ! -s $W/jud_r$r.jsonl ]; then
    .venv-vllm/bin/python scripts/wk_gen.py $W $W/gen_r$r.jsonl --n2 $n2 --n1 $n1 --skip2 $s2 --skip1 $s1 > $W/gen_r$r.log 2>&1 || { say "GEN r$r FAILED"; exit 1; }
    .venv-vllm/bin/python scripts/wk_judge.py $W/gen_r$r.jsonl $W/jud_r$r.jsonl > $W/jud_r$r.log 2>&1 || { say "JUDGE r$r FAILED"; exit 1; }
    say "$(grep -h 'accepted' $W/jud_r$r.log | tr '\n' ' ')"
  fi
  s2=$((s2 + n2)); s1=$((s1 + n1))
done
.venv-rwkv7/bin/python scripts/wk_pools.py $W $O $W/jud_r*.jsonl > $W/pools.log 2>&1 || { say "POOLS FAILED"; exit 1; }
say "$(grep POOLS $W/pools.log | tr '\n' ' ')"
for t in wval wtrain; do
  .venv-gen/bin/python scripts/mm_contexts.py $O $t > $O/ctx_$t.log 2>&1 || { say "CTX $t FAILED"; exit 1; }
  say "$(tail -1 $O/ctx_$t.log)"
done
.venv-vllm/bin/python scripts/mm_qwen_labels.py $O/ctx_wval.jsonl $O/qw_wval.jsonl > $O/qw_wval.log 2>&1 || { say "LABELS wval FAILED"; exit 1; }
split -d -l 4500 $O/ctx_wtrain.jsonl $W/ctxpart_
for p in $W/ctxpart_*; do
  b=$(basename $p); [ -s $W/qw_$b.jsonl ] && continue
  .venv-vllm/bin/python scripts/mm_qwen_labels.py $p $W/qw_$b.jsonl > $W/qw_$b.log 2>&1 || { say "LABELS $b FAILED"; exit 1; }
  say "labels $b: $(tail -2 $W/qw_$b.log | head -1)"
done
cat $W/qw_ctxpart_*.jsonl > $O/qw_wtrain.jsonl; say "labels wtrain: $(wc -l < $O/qw_wtrain.jsonl) rows"
.venv-vllm/bin/python scripts/mm_subq.py --train $O wtrain $O/subq_wtrain.jsonl > $O/subq_wtrain.log 2>&1 || { say "SUBQ FAILED"; exit 1; }
say "WIKI CHAIN DONE — dataset B: ctx/qw/subq_wtrain + _wval in $O"
