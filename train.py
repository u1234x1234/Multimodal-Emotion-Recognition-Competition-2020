import numpy as np
from uxils.data_iterator import seq_iterator_ctx
from uxils.metrics import init_metric
from uxils.pprint_ext import print_obj
from uxils.profiling import Profiler
from uxils.timer import Timer
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop

from utils import Model, get_split, prepare_auido

train_dataset, val_dataset = get_split()

model = Model()
optimizer = init_optimizer_with_model("adam", model)


def read(pv, pa, pt, y, yf, ys):
    x_text = np.load(pt)["word_embed"]
    xa = prepare_auido(pa)

    return x_text, xa, y


metric = init_metric("accuracy")


with seq_iterator_ctx(
    train_dataset, read=read, subsample=10000
) as train_iter, seq_iterator_ctx(val_dataset, read=read, subsample=100) as val_iter:

    for epoch_idx in range(1000):
        training_loop(
            model,
            optimizer,
            train_iter,
            loss_fn="ce",
            forward_fn=lambda model, batch: model(*batch[:2]),
        )
        y_pred, y_true = test_loop(
            model,
            val_iter,
            forward_fn=lambda model, batch: (model(*batch[:2]), batch[-1]),
        )

        acc = metric(y_true, y_pred.argmax(axis=1))
        print(acc)
