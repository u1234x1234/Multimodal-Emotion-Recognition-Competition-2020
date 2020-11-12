from functools import partial

import torch
from uxils.audio.augmentation import augmentation_pipeline
from uxils.data_iterator import seq_iterator_ctx
from uxils.evolution.pbt import start_pbt
from uxils.metrics import init_metric
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop
from uxils.torch_ext.utils import load_state_partial

from common_utils import Model, get_split, prepare_auido, prepare_data


def train_model(
    trajectory,
    in_path,
    out_path,
    fusion_alg="concat",
    audio_aug=None,
    n_seconds=3,
    optimizer="adam",
    audio_freeze_first_n=0,
    audio_freeze_last_n=0,
):
    if audio_aug:
        audio_aug = augmentation_pipeline(audio_aug)

    def read_val(pv, pa, pt, y, yf, ys, aug=None):
        x_audio = prepare_auido(pa, postprocess=aug)
        x_text, x_image = prepare_data(v_path=pv, t_path=pt)
        return x_text, x_audio, x_image, y

    read_train = partial(read_val, aug=audio_aug)

    train_dataset, val_dataset = get_split()
    model = Model(
        fusion_alg=fusion_alg,
        audio_freeze_first_n=audio_freeze_first_n,
        audio_freeze_last_n=audio_freeze_last_n,
    )

    if in_path:
        load_state_partial(model, f"{in_path}/model.pt", verbose=1)

    optimizer = init_optimizer_with_model(optimizer, model)
    metric = init_metric("accuracy")

    with seq_iterator_ctx(
        train_dataset,
        read=read_train,
        subsample=5000,
        batch_size=20,
    ) as train_iter, seq_iterator_ctx(
        val_dataset,
        read=read_val,
        subsample=1000,
        batch_size=20,
    ) as val_iter:
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

    torch.save(model.state_dict(), f"{out_path}/model.pt")
    return acc


search_space = {
    # "fusion_alg": ["mfh", "mfb", "mutan", "mlb", "concat", "linear_sum"],
    "fusion_alg": ["mfb"],
    "audio_aug": [None],
    "n_seconds": [3],
    "optimizer": ["adam"],
    "audio_freeze_first_n": [0, 0.1, 0.25, 0.4],
    "audio_freeze_last_n": [0, 0.1, 0.25, 0.4],
}

start_pbt(
    objective_func=train_model,
    search_space=search_space,
    root_path="arti_pbt001",
    weight_from_result=lambda x: x * x * x,
    n_workers=4,
    gpu_per_worker=0.5,
    cpu_per_worker=8,
    reinit_storage=True,
    debug=False,
)
