import numpy as np
import torch
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import experiment_dir_with_table
from uxils.image.processing import imagenet_normalization
from uxils.metrics import init_metric
from uxils.pprint_ext import print_obj
from uxils.profiling import Profiler
from uxils.timer import Timer
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop
from uxils.video.io import read_video_cv2

from utils import Model, get_split, prepare_auido

train_dataset, val_dataset = get_split()

model = Model()
model.load_state_dict(torch.load("arti/8cb10e/25.pt"))

optimizer = init_optimizer_with_model("sgd,0.01", model)


def read(pv, pa, pt, y, yf, ys):
    x_text = np.load(pt)["word_embed"].mean(axis=0)
    xa = prepare_auido(pa)

    try:
        frames = read_video_cv2(pv)
        image = frames[6]
    except IndexError:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
    except Exception:
        import traceback

        exc = traceback.format_exc()
        print(exc)

    image = imagenet_normalization(image)

    return x_text, xa, image, y


metric = init_metric("accuracy")
out_dir, table = experiment_dir_with_table("arti")

with seq_iterator_ctx(
    train_dataset, read=read, subsample=5000
) as train_iter, seq_iterator_ctx(val_dataset, read=read, subsample=1000) as val_iter:

    for epoch_idx in range(1000):
        training_loop(
            model,
            optimizer,
            train_iter,
            loss_fn="ce",
            forward_fn=lambda model, batch: model(*batch[:3]),
        )

        y_pred, y_true = test_loop(
            model,
            val_iter,
            forward_fn=lambda model, batch: (model(*batch[:3]), batch[-1]),
        )

        acc = metric(y_true, y_pred.argmax(axis=1))
        table.print_row(epoch=epoch_idx, acc=acc)

        torch.save(model.state_dict(), f"{out_dir}/{epoch_idx}.pt")
