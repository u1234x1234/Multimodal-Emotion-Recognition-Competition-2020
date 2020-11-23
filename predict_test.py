import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.image.processing import imagenet_normalization
from uxils.torch_ext.trainer import test_loop
from uxils.video.io import read_video_cv2

from common_utils import MM, MM2, get_test, ids_to_class, prepare_auido, prepare_data, preload_model

dataset = get_test("data/2020-1/test1/")
torch.set_grad_enabled(False)


models = [
    preload_model(MM2, "arti/m001/eaad1d/15.pt"),
    preload_model(MM, "arti/m001/0b624d/22.pt"),  # 0.59
    preload_model(MM2, "arti/m001/1837a9/38.pt"),  # 0.607
]


def read(pv, pa, pt, fid):
    x_audio = prepare_auido(pa, postprocess=None, n_seconds=7, offset=1)
    x_text, x_image = prepare_data(
        v_path=pv,
        t_path=pt,
        image_preprocess=models[0].im_prep,
        image_aug=None,
    )
    return x_audio, x_text, x_image, fid


data_iter = TorchIterator(dataset, read=read)
file_ids = []
y_pred = []
for b in data_iter:
    probabilities = []
    for model in models:
        prob = F.softmax(model(*b[:3]), dim=1)
        probabilities.append(prob)

    prob = sum(probabilities)
    pred = prob.argmax(dim=1).cpu().numpy()

    y_pred.append(pred)
    file_ids.append(b[-1])

y_pred = np.hstack(y_pred)
file_ids = np.hstack(file_ids)
predicted_classes = ids_to_class(y_pred)

file_ids = np.array(file_ids, dtype=np.int)
df = pd.DataFrame(zip(file_ids, predicted_classes), columns=["FileID", "Emotion"])
df.to_csv("sub_test1_004.csv", index=False)
