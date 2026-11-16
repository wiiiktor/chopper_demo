"""Package a trained chopper.pt (scripts/mm_train.py) into the chopper_rag / HF folder layout:
<out>/encoder/{model.safetensors (bf16), config.json, tokenizer*}, <out>/heads.safetensors (fp32 keep + decision), <out>/chopper_config.json.
usage: package_chopper.py <chopper.pt> <out dir>"""
import json, os, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

ck, out = sys.argv[1], sys.argv[2]
c = torch.load(ck, map_location="cpu", weights_only=False)
a = c["args"]; a = a if isinstance(a, dict) else vars(a)
base = a["base"]
enc = AutoModel.from_pretrained(base, dtype=torch.float32)
missing, unexpected = enc.load_state_dict(c["enc"], strict=False)
assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
os.makedirs(f"{out}/encoder", exist_ok=True)
enc.to(torch.bfloat16).save_pretrained(f"{out}/encoder")
AutoTokenizer.from_pretrained(base).save_pretrained(f"{out}/encoder")
save_file({"keep.weight": c["keep"]["weight"].float().contiguous(), "keep.bias": c["keep"]["bias"].float().contiguous(),
           "decision.weight": c["dec"]["weight"].float().contiguous(), "decision.bias": c["dec"]["bias"].float().contiguous()},
          f"{out}/heads.safetensors")
ta = {k: a[k] for k in ("base", "lr", "head_lr", "accum", "max_len", "eval_every", "seed", "hop_w", "keep_w", "train_splits", "val_split",
                        "curriculum", "lr_shape", "anneal", "recycle", "rec_w1", "merge_gap", "min_run", "pred_thr") if k in a}
json.dump({"base": base, "step": c["step"], "rows": c["step"], "train_args": ta,
           "decode": {"pred_thr": a.get("pred_thr", 0.5), "merge_gap": a.get("merge_gap", 5), "min_run": a.get("min_run", 4)},
           "decision": {"0": "SEARCH", "1": "ANSWER"}}, open(f"{out}/chopper_config.json", "w"), indent=1)
print("packaged", base, "step", c["step"], "->", out, sorted(os.listdir(f"{out}/encoder")))
