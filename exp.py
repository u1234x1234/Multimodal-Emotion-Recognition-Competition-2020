import numpy as np
import torch
from uxils.torch_ext.vis import show_netron_gui_onnx
from uxils.multimodal_fusion.torch import get_fusion_module

sizes = [4, 8]
mod = get_fusion_module(sizes, 10, alg="concat_max")
args = [torch.zeros(1, size) for size in sizes]

torch.onnx.export(mod, args, "1.onnx")
show_netron_gui_onnx("1.onnx")
