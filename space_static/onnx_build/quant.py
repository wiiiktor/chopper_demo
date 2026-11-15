"""Weight-only int8 with standard ONNX ops only (runs in onnxruntime-web): activations stay fp32.
 - embedding table: int8 per row + fp32 row scale; Gather(int8) -> Cast -> Mul(Gather(scale))
 - every MatMul with a constant weight [K, N]: int8 per output column, DequantizeLinear(axis=1) -> MatMul
(onnxruntime's dynamic quantization also quantizes the activations and broke the chops: kept text x70.)
usage: quant.py <fp32.onnx> <out.onnx> [matmul: int8 (default) | fp16]"""
import sys
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

MM = sys.argv[3] if len(sys.argv) > 3 else "int8"
m = onnx.load(sys.argv[1]); g = m.graph
init = {i.name: i for i in g.initializer}
new_nodes, drop = [], set()
for n in g.node:
    if n.op_type == "Gather" and n.input[0] in init and len(init[n.input[0]].dims) == 2 and init[n.input[0]].dims[0] > 10000:
        E = numpy_helper.to_array(init[n.input[0]]).astype(np.float32)
        s = np.abs(E).max(1) / 127 + 1e-12
        q = np.round(E / s[:, None]).clip(-127, 127).astype(np.int8)
        b = n.input[0]; drop.add(b)
        g.initializer.extend([numpy_helper.from_array(q, b + "_q"), numpy_helper.from_array(s.astype(np.float32), b + "_s")])
        o = n.output[0]
        new_nodes += [helper.make_node("Gather", [b + "_q", n.input[1]], [o + "_gq"], axis=0),
                      helper.make_node("Cast", [o + "_gq"], [o + "_gf"], to=TensorProto.FLOAT),
                      helper.make_node("Gather", [b + "_s", n.input[1]], [o + "_gs"], axis=0),
                      helper.make_node("Unsqueeze", [o + "_gs", o + "_ax"], [o + "_gs2"]),
                      helper.make_node("Mul", [o + "_gf", o + "_gs2"], [o])]
        g.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), o + "_ax"))
        print("embedding", b, E.shape)
        continue
    if n.op_type == "MatMul" and n.input[1] in init and len(init[n.input[1]].dims) == 2:
        W = numpy_helper.to_array(init[n.input[1]]).astype(np.float32)
        if MM == "fp16":                                   # weights stored fp16, cast to fp32 at run time
            b = n.input[1]; drop.add(b)
            g.initializer.append(numpy_helper.from_array(W.astype(np.float16), b + "_h"))
            new_nodes += [helper.make_node("Cast", [b + "_h"], [b + "_dq"], to=TensorProto.FLOAT),
                          helper.make_node("MatMul", [n.input[0], b + "_dq"], list(n.output))]
            continue
        s = np.abs(W).max(0) / 127 + 1e-12
        q = np.round(W / s[None, :]).clip(-127, 127).astype(np.int8)
        b = n.input[1]; drop.add(b)
        g.initializer.extend([numpy_helper.from_array(q, b + "_q"), numpy_helper.from_array(s.astype(np.float32), b + "_s"),
                              numpy_helper.from_array(np.zeros(W.shape[1], np.int8), b + "_z")])
        new_nodes += [helper.make_node("DequantizeLinear", [b + "_q", b + "_s", b + "_z"], [b + "_dq"], axis=1),
                      helper.make_node("MatMul", [n.input[0], b + "_dq"], list(n.output))]
        continue
    new_nodes.append(n)
used = {i for n in new_nodes for i in n.input}
keep = [i for i in g.initializer if i.name not in drop or i.name in used]
del g.initializer[:]; g.initializer.extend(keep)
del g.node[:]; g.node.extend(new_nodes)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print("saved", sum(1 for n in new_nodes if n.op_type == "DequantizeLinear"), "matmuls quantized")
