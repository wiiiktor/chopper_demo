"""int8 ONNX vs PyTorch fp32 on rrval_notag rows: max |dp|, decision agreement, identical chops (via the package decoder)."""
import json, sys
import numpy as np, onnxruntime as ort, torch
sys.path.insert(0, "release/integrations/chopper-rag")
from chopper_rag.core import Chopper, shape, kept_text

variant, onnx_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ch = Chopper.from_pretrained(variant=variant, device="cpu", **({"repo": sys.argv[4]} if len(sys.argv) > 4 else {}))
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
rows = [json.loads(l) for l in open("/media/wii/bigdata/rr/pack/ctx_rrval_notag.jsonl")][::100][:n]


def chops(s, off, p, head):
    body = np.array([a >= len(head) and b > a for a, b in off])
    q = shape((p > ch.threshold) & body, ch.merge_gap, ch.min_run) & body
    return [s[a:b].strip() for a, b in kept_text(s, off, q) if s[a:b].strip()]


dmax, same, dec, tot, kept_t, kept_o = 0, 0, 0, 0, 0, 0
for r in rows:
    head = f"Question: {r['question']}"; s = head
    for p in r["passages"][:8]:
        s += ch.sep + p["text"].strip()
    e = ch.tokenizer(s, return_offsets_mapping=True, truncation=True, max_length=8192, return_tensors="pt")
    off = e.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        h = ch.encoder(**e).last_hidden_state[0].float()
    pt = torch.sigmoid(h @ ch.heads["keep.weight"].T + ch.heads["keep.bias"])[:, 0].numpy()
    at = float(torch.softmax(h[0] @ ch.heads["decision.weight"].T + ch.heads["decision.bias"], -1)[1])
    po, ao = sess.run(None, {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()})
    po = po[0]; ao = float(ao[0])
    dmax = max(dmax, float(np.abs(po - pt).max()))
    ct, co = chops(s, off, pt, head), chops(s, off, po, head)
    same += ct == co; dec += (at >= .5) == (ao >= .5); tot += 1
    kept_t += sum(map(len, ct)); kept_o += sum(map(len, co))
    if ct != co:
        print("DIFF", len(ct), len(co), set(ct) ^ set(co))
print(f"{variant}: rows {tot} | max |dp| {dmax:.4f} | identical chops {same}/{tot} | decision agree {dec}/{tot} | kept chars torch {kept_t} onnx {kept_o}")
