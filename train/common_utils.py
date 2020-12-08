import glob
import os
import cv2
import numpy as np
import pandas as pd
import torch
from uxils.audio.io import Audio, read_audio
from uxils.audio.processing import fixed_window
from uxils.file_system import glob_audio, glob_files, glob_videos
from uxils.pandas_ext import merge_dataframes
from uxils.torch_ext.image_modules import create_image_module
from uxils.torch_ext.sequential_model import init_sequential
from uxils.torch_ext.utils import freeze_layers


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


def read_video_cv2(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        frames.append(frame)

    return frames


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


def prepare_auido(path, postprocess=None, n_seconds=3, offset=0):
    try:
        audio = read_audio(path, sr=16000)
    except:
        audio = Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    if postprocess is not None:
        audio = postprocess(audio)

    xa = fixed_window(audio, n_seconds)[16000 * offset :]
    return xa


def prepare_data(v_path, t_path, image_preprocess, image_aug):
    x_text = np.load(t_path)["word_embed"]
    n2 = x_text.shape[0] // 2
    x_text = np.hstack(
        (
            x_text.mean(axis=0),
            x_text[: max(n2, 1)].mean(axis=0),
            x_text[n2:].mean(axis=0),
            x_text[0],
            x_text[n2],
            x_text[-1],
        )
    )

    images = read_im(v_path, image_preprocess, image_aug).transpose(2, 0, 1).astype(np.float32)
    return x_text, images


def get_speech_model(model):
    from speech_models import PretrainedSpeakerEmbedding
    return PretrainedSpeakerEmbedding("models/baseline_lite_ap.model")


def read_im(path, image_preprocess, image_aug):
    try:
        frames = read_video_cv2(path)
        # if image_aug is not None:
            # frames = take_n(frames, 4, alg="random_sorted", to_np=False)
        # else:
        frames = [frames[3], frames[5], frames[7], frames[9]]

        image = np.hstack(frames)
    except Exception:
        image = np.zeros((200, 200 * 4, 3), dtype=np.uint8)

    if image_aug is not None:
        image = image_aug(image)

    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    image = image_preprocess(image)
    return image


class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.speech_model = get_speech_model("v1")
        freeze_layers(self.speech_model, 0.7)

        self.image_model, self.im_prep = create_image_module(
            "regnetx_002", pretrained=True, num_classes=0, global_pool=""
        )
        # freeze_layers(self.image_model, 0.2, 0)

        self.text_nn = init_sequential(200 * 6, [200, "relu"])
        self.image_nn = init_sequential(368 * 25, [200, "relu"])

        self.out_nn = init_sequential(512 + 200 + 200, [256, "relu", 7])

    def forward(self, xa, xt, xi):
        xa = self.speech_model(xa)
        xi = self.image_model(xi)

        xi = xi.mean(dim=2).view(xi.shape[0], -1)
        xi = self.image_nn(xi)

        xt = self.text_nn(xt)
        return self.out_nn(torch.cat([xa, xt, xi], dim=1))


class MM2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.speech_model = get_speech_model("v1")
        # freeze_layers(self.speech_model, 0.7)

        self.image_model, self.im_prep = create_image_module(
            "regnetx_002", pretrained=True, num_classes=0, global_pool="avg"
        )
        # freeze_layers(self.image_model, 0.5, 0)

        self.text_nn = init_sequential(200 * 6, [200, "relu"])
        self.image_nn = init_sequential(368 * 1, [200, "relu"])

        self.out_nn = init_sequential(512 + 200 + 200, [256, "relu", 7])

    def forward(self, xa, xt, xi):
        xa = self.speech_model(xa)
        xi = self.image_model(xi)

        xi = self.image_nn(xi)

        xt = self.text_nn(xt)
        return self.out_nn(torch.cat([xa, xt, xi], dim=1))


def preload_model(cls, path):
    model = cls()
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.eval()
    return model
