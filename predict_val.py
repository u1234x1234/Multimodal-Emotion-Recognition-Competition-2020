import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.torch_ext.trainer import test_loop
from scipy.stats import gmean
from common_utils import MM, MM2, get_test, ids_to_class, prepare_auido, prepare_data, get_split, preload_model

dataset = get_split()[1]
# dataset = dataset[:1000]

torch.set_grad_enabled(False)


models = [
    # preload_model(MM2, "arti/m001/55c424/15.pt"),
    preload_model(MM, "arti/m001/f3456b/18.pt"),  # 0.562
    preload_model(MM2, "arti/m001/2430bc/30.pt"),  # 0.571
    preload_model(MM, "arti/m001/0b624d/22.pt"),  # 0.59
    preload_model(MM2, "arti/m001/1837a9/38.pt"),  # 0.607
]


def read(pv, pa, pt, y, *args):
    x_audio = prepare_auido(pa, postprocess=None, n_seconds=7, offset=1)
    x_text, x_image = prepare_data(
        v_path=pv,
        t_path=pt,
        image_preprocess=models[0].im_prep,
        image_aug=None,
    )
    return x_audio, x_text, x_image, y


data_iter = TorchIterator(dataset, read=read)
y_true = []
y_pred = []
for b in data_iter:
    probabilities = []
    for model in models:
        prob = F.softmax(model(*b[:3]), dim=1)
        probabilities.append(prob)

    prob = np.stack([x.cpu().numpy() for x in probabilities], axis=2)
    prob = gmean(prob, axis=2)
    pred = prob.argmax(axis=1)
    # pred = prob.argmax(dim=1).cpu().numpy()

    y_true.append(b[-1].cpu().numpy())
    y_pred.append(pred)


y_pred = np.hstack(y_pred)
y_true = np.hstack(y_true)

print((y_pred == y_true).mean())
