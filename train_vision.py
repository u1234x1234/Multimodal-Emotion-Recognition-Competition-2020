import torchvision.transforms as transforms
import torch
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5],
                         std = [0.5, 0.5, 0.5]),
])
import numpy as np
import cv2
from uxils.torch_ext.feedforward_modules import init_sequential_from_model, init_sequential
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import torch_experiment
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.metrics import init_metric
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop
from uxils.torch_ext.utils import freeze_layers
from uxils.video.io import read_random_frame, read_video_cv2

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
model, preprocess_image = get_face_recognition_model("imagenet_regnetx002", num_classes=0)
freeze_layers(model, 0.5, strict=False)

# model = init_sequential_from_model(model, [128, "tanh", 7])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model
        self.out_nn = init_sequential(368, [128, "relu", 7])

    def forward(self, xs):
        bs = xs.shape[0]
        # xs = [self.backbone(x).mean(dim=0) for x in xs]
        xs = xs.view(xs.shape[0] * xs.shape[1], *xs.shape[2:])
        xs = self.backbone(xs).view(bs, 2, -1).mean(dim=1)
        # xs = torch.stack(xs, dim=0)

        return self.out_nn(xs)



model = Model()
optimizer = init_optimizer_with_model("adam", model)


def read(pv, pa, pt, y, yf, ys):

    try:
        images = read_video_cv2(pv)
        images = [images[idx] for idx in np.random.randint(len(images), size=2)]
        images = [preprocess_image(image) for image in images]
    except Exception:
        images = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(2)]

    images = np.array(images).transpose(0, 3, 1, 2)
    return images, yf


metric = init_metric("accuracy")
exp = torch_experiment("arti_face", model, "acc", print_diff=True)

with seq_iterator_ctx(
    train_dataset, read=read, subsample=1000, batch_size=16, memory_gb=32
) as train_iter, seq_iterator_ctx(val_dataset, read=read, batch_size=16, memory_gb=32) as val_iter:

    for epoch_idx in range(1000):
        training_loop(
            model,
            optimizer,
            train_iter,
            loss_fn="ce",
            forward_fn=lambda model, batch: model(*batch[:-1]),
        )

        y_pred, y_true = test_loop(
            model,
            val_iter,
            forward_fn=lambda model, batch: (model(*batch[:-1]), batch[-1]),
        )

        acc = metric(y_true, y_pred.argmax(axis=1))
        exp.update(acc=acc)
