import numpy as np
import torch
from uxils.torch_ext.vis import show_netron_gui_onnx
from uxils.multimodal_fusion.torch import get_fusion_module

sizes = [100, 200]
mod = get_fusion_module(sizes, 10, alg="mutan")
args = [torch.zeros(1, size) for size in sizes]

print(mod(args).shape)

torch.onnx.export(mod, args, "1.onnx", opset_version=12)
show_netron_gui_onnx("1.onnx")
