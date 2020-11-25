import glob
import os
from functools import partial

import cv2
import librosa
import numpy as np
import timm
import torch

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


def ids_to_class(indices):
    return [CLASSES[i] for i in indices]


def mean_std_normalization(img, mean, std):
    "(image - mean) / std"
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img -= np.array(mean)
    img /= np.array(std)
    return img


def create_image_module(name, num_classes=None, pretrained=True, global_pool="avg"):
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, global_pool=global_pool
    )
    cfg = model.default_cfg
    preprocessing = partial(mean_std_normalization, mean=cfg["mean"], std=cfg["std"])
    return model, preprocessing


def get_test(prefix, audio_dir, face_dir):
    paths = []
    for vpath in glob.glob(f"{prefix}/**/*.mp4", recursive=True):
        bn = os.path.basename(vpath)
        v_id = bn.split("-")[0]
        tpath = f"{prefix}/{v_id}.npz"
        apath = f"{audio_dir}/{bn.split('.')[0]}.wav"
        vpath = f"{face_dir}/{bn}"

        assert os.path.exists(apath)
        assert os.path.exists(vpath)
        assert os.path.exists(tpath)

        paths.append((vpath, apath, tpath, v_id))

    return paths


class Audio:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def __str__(self):
        return (
            f"Audio: {self.data.shape} SR: {self.sample_rate}, Sec: {self.n_seconds()}"
        )

    def n_seconds(self):
        return self.data.shape[0] / self.sample_rate


def read_audio(path, sr=None, resampling_type=0):
    audio, sr = librosa.load(path, sr=sr)
    return Audio(audio, sr)


def prepare_auido(path, postprocess=None, n_seconds=3, offset=0):
    try:
        audio = read_audio(path, sr=16000)
    except:
        audio = Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    if postprocess is not None:
        audio = postprocess(audio)

    xa = fixed_window(audio, n_seconds)[16000 * offset :]
    return xa


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


def read_im(path, image_preprocess, image_aug):
    try:
        frames = read_video_cv2(path)
        frames = [frames[3], frames[5], frames[7], frames[9]]
        image = np.hstack(frames)
    except Exception:
        image = np.zeros((200, 200 * 4, 3), dtype=np.uint8)

    if image_aug is not None:
        image = image_aug(image)

    image = image_preprocess(image)
    return image


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

    images = (
        read_im(v_path, image_preprocess, image_aug)
        .transpose(2, 0, 1)
        .astype(np.float32)
    )
    return x_text, images


def n_dims(x):
    try:
        return n_dims(x[0]) + 1
    except Exception:
        return 0


def apply_to_row_1d(X, func, to_numpy=True, dtype=np.float32):
    "apply func to rows if X is 2d or to 1d"

    d = n_dims(X)

    if d == 1:
        X = func(X)
    elif d == 2:
        X = [func(x) for x in X]
    else:
        raise ValueError("Incorrect shapes")

    if to_numpy:
        X = np.array(X, dtype=dtype)

    return X


def _pad_single(x, n, alg):
    if len(x) < n:
        if alg == "tile":
            n_rep = np.math.ceil(n / len(x))
            x = np.tile(x, n_rep)
        elif alg == "pad":
            x = np.pad(x, (0, n - len(x)))
        else:
            raise ValueError(f'alg "{alg}" is not implemented yet')
    return x[:n]


def pad_1d(X, n=32000, alg="tile", to_numpy=True, dtype=np.float32):
    "Transform sequence[or list of sequences] of variable len to the fixed len"
    func = partial(_pad_single, n=n, alg=alg)
    return apply_to_row_1d(X, func)


def _to_fixed_1d(x, n, pad_alg="tile"):
    x = x[:n]
    if len(x) < n:
        x = pad_1d(x, n)
    return x


def to_fixed_1d(X, n=32000, pad_alg="pad"):
    func = partial(_to_fixed_1d, n=n, pad_alg=pad_alg)
    return apply_to_row_1d(X, func)


def fixed_window(audio: Audio, seconds, pad_alg="pad"):
    "Pad or crop"
    assert 0 < seconds < 30
    assert isinstance(audio, Audio)
    out_size = int(audio.sample_rate * seconds)
    return to_fixed_1d(audio.data, out_size, pad_alg=pad_alg)


class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.speech_model = PretrainedSpeakerEmbedding(None)
        self.image_model, self.im_prep = create_image_module(
            "regnetx_002", pretrained=True, num_classes=0, global_pool=""
        )

        self.text_nn = torch.nn.Sequential(
            torch.nn.Linear(200 * 6, 200), torch.nn.ReLU()
        )
        self.image_nn = torch.nn.Sequential(
            torch.nn.Linear(368 * 25, 200), torch.nn.ReLU()
        )
        self.out_nn = torch.nn.Sequential(
            torch.nn.Linear(512 + 200 + 200, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 7),
        )

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

        self.speech_model = PretrainedSpeakerEmbedding(None)
        self.image_model, self.im_prep = create_image_module(
            "regnetx_002", pretrained=True, num_classes=0, global_pool="avg"
        )

        self.text_nn = torch.nn.Sequential(
            torch.nn.Linear(200 * 6, 200), torch.nn.ReLU()
        )
        self.image_nn = torch.nn.Sequential(
            torch.nn.Linear(368 * 1, 200), torch.nn.ReLU()
        )
        self.out_nn = torch.nn.Sequential(
            torch.nn.Linear(512 + 200 + 200, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 7),
        )

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
