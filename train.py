import numpy as np
from functools import partial
from uxils.torch_ext.utils import freeze_layers
from uxils.audio.augmentation import augmentation_pipeline
from uxils.ray_ext.hpo_executor import execute_search_space
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.torch_ext.trainer import ModelTrainer
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.torch_ext.sequence_modules import create_sequence_model
from uxils.multimodal_fusion.torch import FusionModule
from common_utils import Model, get_split, prepare_auido, prepare_data, get_speech_model


def train_model(
    fusion_alg="concat",
    audio_aug=None,
    n_seconds=7,
    offset=2,
    audio_freeze_first_n=0.3,
    audio_freeze_last_n=0,
    image_freeze_first_n=0.5,
    n_images=4,
):
    speech_model = get_speech_model("v1")
    freeze_layers(speech_model, audio_freeze_first_n, audio_freeze_last_n)

    back, im_prep = get_face_recognition_model("imagenet_regnetx002", num_classes=0)
    freeze_layers(back, image_freeze_first_n)
    image_model = create_sequence_model(back, "128", n_images, alg="concat")

    model = FusionModule([speech_model, None, image_model], [512, 200, 128], 7, alg="concat", mm_alg="sum")

    def read_val(pv, pa, pt, y, yf, ys, aug=None, frame=None):
        x_audio = prepare_auido(pa, postprocess=aug, n_seconds=n_seconds, offset=offset)
        x_text, x_image = prepare_data(
            v_path=pv,
            t_path=pt,
            frame=frame,
            image_preprocess=im_prep,
            n_images=n_images,
        )
        return x_audio, x_text, x_image, y

    read_train = partial(read_val, aug=audio_aug, frame="random")

    train_dataset, val_dataset = get_split()

    train_iter = TorchIterator(
        train_dataset, read=read_train, epoch_size=5000, batch_size=20
    )
    val_iter = TorchIterator(val_dataset, read=read_val, epoch_size=1000, batch_size=20)

    trainer = ModelTrainer(
        model=model,
        optimizer="adam",
        loss="ce",
        metric="accuracy",
        forward=lambda model, batch: model(*batch[:3]),
        forward_val=lambda model, batch: (batch[-1], model(*batch[:3])),
        exp_prefix="arti/m001",
    )

    return trainer.train(train_iter, val_iter, n_epochs=20)


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
