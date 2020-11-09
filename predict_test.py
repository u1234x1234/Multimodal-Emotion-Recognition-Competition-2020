import numpy as np
import pandas as pd
import torch
from uxils.data_iterator import seq_iterator_ctx
from uxils.image.processing import imagenet_normalization
from uxils.torch_ext.trainer import test_loop
from uxils.video.io import read_video_cv2

from utils import Model, get_test, ids_to_class, prepare_auido

dataset = get_test()

model = Model()
model.load_state_dict(torch.load("arti/8cb10e/25.pt"))


def read(pv, pa, pt, fid):
    x_text = np.load(pt)["word_embed"].mean(axis=0)
    xa = prepare_auido(pa)

    try:
        frames = read_video_cv2(pv)
        image = frames[6]
    except Exception:
        import traceback

        exc = traceback.format_exc()
        print(exc)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

    image = imagenet_normalization(image)

    return x_text, xa, image, fid


model.eval()
torch.set_grad_enabled(False)

with seq_iterator_ctx(dataset, read=read) as data_iter:

    y_pred, file_ids = test_loop(
        model,
        data_iter,
        forward_fn=lambda model, batch: (model(*batch[:3]), batch[-1]),
    )

    predicted_classes = ids_to_class(y_pred.argmax(axis=1))


file_ids = np.array(file_ids, dtype=np.int)
df = pd.DataFrame(zip(file_ids, predicted_classes), columns=["FileID", "Emotion"])
df.to_csv("sub_test1_001.csv", index=False)
