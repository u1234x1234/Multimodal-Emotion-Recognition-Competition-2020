import os

import numpy as np
import pandas as pd
import torch
from uxils.audio.io import Audio, read_audio
from uxils.audio.processing import fixed_window
from uxils.cache import cached_persistent
from uxils.file_system import glob_audio, glob_files, glob_videos
from uxils.pandas_ext import merge_dataframes
from uxils.torch_ext.sequential_model import init_sequential

from speech_models import PretrainedSpeakerEmbedding

CLASSES = [
    "neu",
    "fea",
    "dis",
    "ang",
    "sad",
    "hap",
    "sur",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def class_to_id(name):
    return CLASS_TO_ID[name]


@cached_persistent
def get_split():
    """pv, pa, pt, y_global, y_face, y_speech"""
    id_to_vpath = {
        int(os.path.basename(p).split("-")[0]): p for p in glob_videos("face_images/")
    }
    id_to_apath = {
        int(os.path.basename(p).split("-")[0]): p for p in glob_audio("data/audio")
    }
    id_to_tpath = {
        int(os.path.basename(p).split(".")[0]): p
        for p in glob_files("data/", extensions=["npz"])
    }

    def _prep(p1, p2, p3):
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        df3 = pd.read_csv(p3)
        df = merge_dataframes(df1, df2, df3, on="FileID")
        for i in range(1, 4):
            df.iloc[:, i] = df.iloc[:, i].apply(class_to_id)
        paths = [
            (id_to_vpath.get(path), id_to_apath.get(path), id_to_tpath.get(path))
            for path in df.FileID.values
        ]

        train = list(zip(*paths))
        train = list(zip(*train, *[df.values[:, idx] for idx in range(1, 4)]))
        return train

    train = _prep(
        "data/2020-1/train.csv",
        "data/2020-1/train_face.csv",
        "data/2020-1/train_speech.csv",
    )
    val = _prep(
        "data/2020-1/val.csv", "data/2020-1/val_face.csv", "data/2020-1/val_speech.csv"
    )

    return train, val


def prepare_auido(path):
    try:
        audio = read_audio(path, sr=16000)
    except:
        audio = Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    xa = fixed_window(audio, 4)[16000:]
    return xa


from uxils.torch_ext.utils import freeze_layers
import timm
from uxils.image.face.pretrained import get_face_recognition_model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.speech_model = PretrainedSpeakerEmbedding("models/baseline_lite_ap.model")
        freeze_layers(self.speech_model, freeze=0.5, invert=0, matchers=(".*attention", "model[.]fc", ".*sap_linear"), verbose=1)
        # freeze_layers(self.speech_model, freeze=0.5)

        self.vision_model = get_face_recognition_model("imagenet_regnetx002", num_classes=100)[0]
        # self.text_model = init_sequential(200, [100])

        emb_size = 512 + 200 + 100
        self.out_nn = init_sequential(emb_size, [128, "relu", 7])

    def forward(self, xt, xa, xim):
        # xt = torch.stack([self.text_model(x).mean(dim=0) for x in xt], dim=0)
        xa = self.speech_model(xa)
        xim = self.vision_model(xim)

        x = torch.cat([xa, xt, xim], dim=1)
        x = self.out_nn(x)

        return x
