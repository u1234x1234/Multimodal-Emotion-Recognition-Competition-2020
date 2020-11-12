import glob
import os

import numpy as np
import pandas as pd
import torch
from uxils.audio.io import Audio, read_audio
from uxils.audio.processing import fixed_window
from uxils.cache import cached_persistent
from uxils.file_system import glob_audio, glob_files, glob_videos
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.image.processing import imagenet_normalization
from uxils.pandas_ext import merge_dataframes
from uxils.torch_ext.sequential_model import init_sequential
from uxils.torch_ext.utils import freeze_layers
from uxils.video.io import read_video_cv2, read_random_frame

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


def ids_to_class(indices):
    return [CLASSES[i] for i in indices]


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


def get_test():
    root_dir = "data/2020-1/test1/"
    paths = []
    for vpath in glob.glob(f"{root_dir}/*.mp4"):
        bn = os.path.basename(vpath)
        v_id = bn.split("-")[0]
        tpath = f"{root_dir}/{v_id}.npz"
        apath = f"data/audio/test1/{bn.split('.')[0]}.wav"
        vpath = f"face_images/{bn}"

        paths.append((vpath, apath, tpath, v_id))

    return paths


def prepare_auido(path, postprocess=None, n_seconds=3, offset=0):
    try:
        audio = read_audio(path, sr=16000)
    except:
        audio = Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    if postprocess is not None:
        audio = postprocess(audio)

    xa = fixed_window(audio, n_seconds)[16000 * offset :]
    return xa


def prepare_data(v_path, t_path, frame, image_preprocess):
    x_text = np.load(t_path)["word_embed"].mean(axis=0)

    try:
        if frame is None:  # validation
            frames = read_video_cv2(v_path)
            image = frames[6]
        else:
            image = read_random_frame(v_path)
    except Exception:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        import traceback

        exc = traceback.format_exc()

    image = image_preprocess(image)
    return x_text, image


def get_speech_model(model):
    if model == "v1":
        from speech_models import PretrainedSpeakerEmbedding

        return PretrainedSpeakerEmbedding("models/baseline_lite_ap.model")
    else:
        from clovaai_ResNetSE34V2 import PretrainedSpeakerEmbedding

        return PretrainedSpeakerEmbedding("models/baseline_v2_ap.model")


class Model(torch.nn.Module):
    def __init__(self, fusion_alg, audio_freeze_first_n, audio_freeze_last_n, image_freeze_first_n):
        super().__init__()

        self.speech_model = get_speech_model("v1")

        freeze_layers(
            self.speech_model,
            freeze_first_n=audio_freeze_first_n,
            freeze_last_n=audio_freeze_last_n,
            verbose=0,
        )

        self.vision_model, self.image_preprocess = get_face_recognition_model(
            # "imagenet_regnetx002",
            num_classes=128,
        )
        freeze_layers(self.vision_model, freeze_first_n=0.95)

        from uxils.multimodal_fusion.torch import get_fusion_module

        self.fusion = get_fusion_module([512, 200, 128], 128, fusion_alg)
        self.out_nn = init_sequential(128, [128, "relu", 7])

    def forward(self, xt, xa, xim):
        # xt = torch.stack([self.text_model(x).mean(dim=0) for x in xt], dim=0)

        xa = self.speech_model(xa)
        xim = self.vision_model(xim)
        x = self.fusion([xa, xt, xim])

        # x = torch.cat([xa, xt, xim], dim=1)
        # x = self.out_nn(xa)

        return x
