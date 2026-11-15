"""Export a chopper variant (encoder + KEEP head + decision head) to one ONNX graph, then int8-quantize it.
outputs: p_keep [batch, seq] (sigmoid of KEEP logit per token), p_answer [batch] (softmax[1] of decision on token 0).
usage: export.py <variant folder, e.g. mmbert-small> <out dir>"""
import os, sys, shutil
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel

folder, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
d = os.path.join(snapshot_download("wiiiktor/chopper_mmbert", allow_patterns=[f"{folder}/*"]), folder)
enc = AutoModel.from_pretrained(f"{d}/encoder", torch_dtype=torch.float32, attn_implementation="eager",
                                reference_compile=False).eval()
H = load_file(f"{d}/heads.safetensors")


class Wrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = enc
        self.kw = torch.nn.Parameter(H["keep.weight"].float(), requires_grad=False)
        self.kb = torch.nn.Parameter(H["keep.bias"].float(), requires_grad=False)
        self.dw = torch.nn.Parameter(H["decision.weight"].float(), requires_grad=False)
        self.db = torch.nn.Parameter(H["decision.bias"].float(), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        p_keep = torch.sigmoid(h @ self.kw.T + self.kb)[..., 0]
        p_ans = torch.softmax(h[:, 0] @ self.dw.T + self.db, -1)[:, 1]
        return p_keep, p_ans


m = Wrap().eval()
ids = torch.randint(10, 1000, (1, 300)); am = torch.ones_like(ids)
f32 = f"{out}/model_fp32.onnx"
torch.onnx.export(m, (ids, am), f32, input_names=["input_ids", "attention_mask"], output_names=["p_keep", "p_answer"],
                  dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "p_keep": {0: "b", 1: "t"},
                                "p_answer": {0: "b"}}, opset_version=17, dynamo=False)
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(f32, f"{out}/model_int8.onnx", weight_type=QuantType.QInt8)
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(f"{d}/encoder/{f}", f"{out}/{f}")
shutil.copy(f"{d}/chopper_config.json", f"{out}/chopper_config.json")
print({f: round(os.path.getsize(f"{out}/{f}") / 1e6, 1) for f in os.listdir(out)})
