from functools import partial

from uxils.audio.augmentation import augmentation_pipeline
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import torch_experiment
from uxils.metrics import init_metric
from uxils.ray_ext.hpo_executor import execute_search_space
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop

from common_utils import Model, get_split, prepare_auido, prepare_data


def train_model(
    fusion_alg="concat",
    audio_aug=None,
    n_seconds=7,
    offset=2,
    audio_freeze_first_n=0.3,
    audio_freeze_last_n=0,
    image_freeze_first_n=0,
):
    model = Model(
        fusion_alg=fusion_alg,
        audio_freeze_first_n=audio_freeze_first_n,
        audio_freeze_last_n=audio_freeze_last_n,
        image_freeze_first_n=image_freeze_first_n,
    )

    def read_val(pv, pa, pt, y, yf, ys, aug=None, frame=None):
        x_audio = prepare_auido(pa, postprocess=aug, n_seconds=n_seconds, offset=offset)
        x_text, x_image = prepare_data(
            v_path=pv, t_path=pt, frame=frame, image_preprocess=model.image_preprocess
        )
        return x_text, x_audio, x_image, y

    read_train = partial(read_val, aug=audio_aug, frame="random")

    train_dataset, val_dataset = get_split()
    optimizer = init_optimizer_with_model("adam", model)

    metric = init_metric("accuracy")
    exp = torch_experiment("arti/fus003", model, "acc", print_diff=True)

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


train_model()
# qwe

search_space = {
    # "fusion_alg": ["mfh", "mfb", "mutan", "mlb", "concat", "linear_sum"],
    "fusion_alg": ["mfb"],
    "audio_aug": [None],
    "n_seconds": [7],
    "offset": [2],
    "audio_freeze_first_n": [0.3],
    "audio_freeze_last_n": [0],
}

execute_search_space(
    search_space,
    train_model,
    n_workers=6,
    gpu_per_worker=0.3,
    cpu_per_worker=6,
    debug=True,
)
