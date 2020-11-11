from functools import partial

from uxils.audio.augmentation import stretch_shift_pitch
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import torch_experiment
from uxils.metrics import init_metric
from uxils.ray_ext.hpo_executor import execute_search_space
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop

from utils import Model, get_split, prepare_auido, prepare_data

n_seconds = 3
# model.load_state_dict(torch.load("arti/8cb10e/25.pt"))


def train_model(fusion_alg="concat", audio_aug=None):
    def read_val(pv, pa, pt, y, yf, ys, aug=None):
        x_audio = prepare_auido(pa, postprocess=aug)
        x_text, x_image = prepare_data(v_path=pv, t_path=pt)
        return x_text, x_audio, x_image, y

    read_train = partial(read_val, aug=audio_aug)

    train_dataset, val_dataset = get_split()
    model = Model(fusion_alg=fusion_alg)
    optimizer = init_optimizer_with_model("adam", model)

    metric = init_metric("accuracy")
    exp = torch_experiment("arti/fus004", model, "acc")

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

        for epoch_idx in range(20):
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

    return acc


search_space = {
    # "fusion_alg": ["mfh", "mfb", "mutan", "mlb", "concat", "linear_sum"],
    "fusion_alg": ["mlb"],
    "audio_aug": [None, stretch_shift_pitch],
}

execute_search_space(
    search_space,
    train_model,
    n_workers=4,
    n_cpus=40,
    n_gpus=2,
    gpu_per_trial=0.5,
    cpu_per_trial=8,
)
