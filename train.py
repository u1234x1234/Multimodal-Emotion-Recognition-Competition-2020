import numpy as np
import torch
from uxils.audio.augmentation import augmentation_pipeline
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import experiment_dir_with_table, torch_experiment
from uxils.metrics import init_metric
from uxils.pprint_ext import print_obj
from uxils.profiling import Profiler
from uxils.timer import Timer
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop

from utils import Model, get_split, prepare_auido, prepare_data

train_dataset, val_dataset = get_split()

n_seconds = 3
model = Model()
# model.load_state_dict(torch.load("arti/8cb10e/25.pt"))

optimizer = init_optimizer_with_model("adam", model)

audio_aug = augmentation_pipeline(
    [
        "gain,-30,10",
        "stretch",
        "shift",
        "pitch",
        "gaussian",
    ]
)


def read_train(pv, pa, pt, y, yf, ys):
    x_audio = prepare_auido(pa, audio_aug)
    x_text, x_image = prepare_data(v_path=pv, t_path=pt)
    return x_text, x_audio, x_image, y


def read_val(pv, pa, pt, y, yf, ys):
    x_audio = prepare_auido(pa)
    x_text, x_image = prepare_data(v_path=pv, t_path=pt)
    return x_text, x_audio, x_image, y


metric = init_metric("accuracy")
exp = torch_experiment("arti", model, "acc")

with seq_iterator_ctx(
    train_dataset, read=read_train, subsample=5000
) as train_iter, seq_iterator_ctx(
    val_dataset, read=read_val, subsample=1000
) as val_iter:

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
        exp.update(acc=acc)
