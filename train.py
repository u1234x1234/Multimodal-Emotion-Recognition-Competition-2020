from functools import partial

import numpy as np
import torch
from uxils.audio.augmentation import augmentation_pipeline
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.multimodal_fusion.torch import FusionModule
from uxils.ray_ext.hpo_executor import execute_search_space
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.torch_ext.sequence_modules import create_sequence_model
from uxils.torch_ext.trainer import ModelTrainer
from uxils.torch_ext.utils import freeze_layers, load_state_partial
from uxils.audio.augmentation import augmentation_pipeline
from common_utils import MM, MM2, get_split, prepare_auido, prepare_data
from uxils.image.aug import augmentation_pipeline as iap


def train_model(
    offset=1,
    n_seconds=7,
):

    model = MM2()
    # load_state_partial(model, "arti/m001/1837a9/38.pt", verbose=1)

    def read_val(pv, pa, pt, y, yf, ys, aug=None, image_aug=None):
        x_audio = prepare_auido(pa, postprocess=aug, n_seconds=n_seconds, offset=offset)
        x_text, x_image = prepare_data(
            v_path=pv,
            t_path=pt,
            image_preprocess=model.im_prep,
            image_aug=image_aug
        )

        return x_audio, x_text, x_image, y

    aug = None
    # aug = augmentation_pipeline("gain_gaussian")
    image_aug = None
    # image_aug = iap(operations=["hflip", "rgb_shift", "brightness_contrast"])

    read_train = partial(read_val, aug=aug, image_aug=image_aug)
    train_dataset, val_dataset = get_split()

    train_iter = TorchIterator(
        train_dataset, read=read_train, epoch_size=4000, batch_size=32, n_workers=12
    )
    val_iter = TorchIterator(
        val_dataset, read=read_val, epoch_size=2000, batch_size=32, n_workers=8
    )

    trainer = ModelTrainer(
        model=model,
        optimizer="adamw",
        loss="ce",
        metric="accuracy",
        forward=lambda model, batch: model(*batch[:3]),
        forward_val=lambda model, batch: (batch[-1], model(*batch[:3])),
        exp_prefix="arti/m001",
        exp_kwargs={"print_diff": True, "save_models": True},
        # scheduler="CosineAnnealingLR",
    )

    return trainer.train(train_iter, val_iter)


train_model()
