import torch
import cv2
import numpy as np
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.torch_ext.data_iterator import TorchIterator
from uxils.torch_ext.feedforward_modules import (
    init_sequential,
    init_sequential_from_model,
)
from uxils.torch_ext.trainer import ModelTrainer
from uxils.torch_ext.utils import freeze_layers
from uxils.video.io import read_random_frame, read_video_cv2
from uxils.torch_ext.sequence_modules import create_sequence_model
from common_utils import Model, get_split

train_dataset, val_dataset = get_split()

from model_irse import IR_50

# model = IR_50([112, 112])
# model.load_state_dict(torch.load("models/ms1m_ir50/backbone_ir50_ms1m_epoch63.pth"))

# def preprocess_image(image):
#     image = cv2.resize(image, (112, 112)).astype(np.float32)
#     image /= 255.
#     image = (image - 0.5) / 0.5
#     return image

# model, preprocess_image = get_face_recognition_model("facenet_pytorch_vggface2", num_classes=7)
backbone, preprocess_image = get_face_recognition_model(
    "imagenet_regnetx002", num_classes=0
)
# freeze_layers(backbone, 0.5, strict=False)


def read(pv, pa, pt, y, yf, ys):

    try:
        images = read_video_cv2(pv)
        images = [images[idx] for idx in sorted(np.random.randint(len(images), size=1))]
        # images = [images[0], images[3], images[6], images[-1]]
        images = [preprocess_image(image) for image in images]
    except Exception:
        images = [np.zeros((200, 200, 3), dtype=np.float32) for _ in range(1)]

    images = np.array(images).transpose(0, 3, 1, 2)
    return images, yf


model = create_sequence_model(backbone, "7", seq_size=1, alg="concat")
train_iter = TorchIterator(
    train_dataset, read=read, epoch_size=5000, batch_size=20, n_workers=8
)
val_iter = TorchIterator(val_dataset, read=read, epoch_size=1000, batch_size=20, n_workers=8)

trainer = ModelTrainer(
    model=model,
    optimizer="adam",
    loss="ce",
    metric="accuracy",
    forward=lambda model, batch: model(batch[0]),
    forward_val=lambda model, batch: (batch[-1], model(batch[0])),
    exp_prefix="arti/m001",
)
trainer.train(train_iter, val_iter)
