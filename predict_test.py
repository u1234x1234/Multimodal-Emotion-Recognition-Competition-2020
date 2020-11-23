import numpy as np
import pandas as pd
import torch
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.image.processing import imagenet_normalization
from uxils.torch_ext.trainer import test_loop
from uxils.video.io import read_video_cv2

from common_utils import MM, get_test, ids_to_class, prepare_auido, prepare_data

dataset = get_test("data/2020-1/test1/")

model = MM()
model.load_state_dict(torch.load("arti/m001/0b624d/22.pt"))


def read(pv, pa, pt, fid):
    x_audio = prepare_auido(pa, postprocess=None, n_seconds=7, offset=0)
    x_text, x_image = prepare_data(
        v_path=pv,
        t_path=pt,
        image_preprocess=model.im_prep,
    )
    return x_audio, x_text, x_image, fid


model.eval()
torch.set_grad_enabled(False)

data_iter = TorchIterator(dataset, read=read)
y_pred, file_ids = test_loop(
    model,
    data_iter,
    forward_fn=lambda model, batch: (model(*batch[:3]), batch[-1]),
)
predicted_classes = ids_to_class(y_pred.argmax(axis=1))


file_ids = np.array(file_ids, dtype=np.int)
df = pd.DataFrame(zip(file_ids, predicted_classes), columns=["FileID", "Emotion"])
df.to_csv("sub_test1_002.csv", index=False)
