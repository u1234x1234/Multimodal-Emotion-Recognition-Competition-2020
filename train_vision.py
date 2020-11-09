import numpy as np
from uxils.data_iterator import seq_iterator_ctx
from uxils.experiment import experiment_dir_with_table
from uxils.metrics import init_metric
from uxils.torch_ext.module_utils import init_optimizer_with_model
from uxils.torch_ext.trainer import test_loop, training_loop
from uxils.video.io import read_random_frame
from uxils.image.face.pretrained import get_face_recognition_model
from uxils.torch_ext.utils import freeze_layers
from utils import Model, get_split
from uxils.image.processing import imagenet_normalization

train_dataset, val_dataset = get_split()

model, preprocess_image = get_face_recognition_model(num_classes=7)
# model, preprocess_image = get_face_recognition_model("imagenet_regnetx002", num_classes=7)
# freeze_layers(model, 0.9)
# model.load_state_dict(torch.load("arti/"))

optimizer = init_optimizer_with_model("adam", model)


def read(pv, pa, pt, y, yf, ys):

    try:
        image = read_random_frame(pv)
    except Exception:
        image = np.zeros((200, 200, 3), dtype=np.uint8)

    image = preprocess_image(image)

    return image, yf


metric = init_metric("accuracy")
out_dir, table = experiment_dir_with_table("arti")

with seq_iterator_ctx(
    train_dataset, read=read, subsample=10000
) as train_iter, seq_iterator_ctx(val_dataset, read=read, subsample=100) as val_iter:

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
        table.print_row(epoch=epoch_idx, acc=acc)
