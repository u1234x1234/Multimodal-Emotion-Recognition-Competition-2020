import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from predict_utils import (
    MM,
    MM2,
    get_test,
    ids_to_class,
    preload_model,
    prepare_auido,
    prepare_data,
)


class MultimodalDataset(Dataset):
    def __init__(self, data, read):
        self.data = data
        self.read = read

    def __getitem__(self, x):
        return self.read(*self.data[x])

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    dataset = get_test(sys.argv[1], sys.argv[2], sys.argv[3])
    torch.set_grad_enabled(False)

    models = [
        preload_model(MM, "f3456b_18.pt"),  # 0.562
        preload_model(MM, "0b624d_22.pt"),  # 0.59
        preload_model(MM2, "1837a9_38.pt"),  # 0.607
    ]

    def read(pv, pa, pt, fid):
        x_audio = prepare_auido(pa, postprocess=None, n_seconds=7, offset=1)
        x_text, x_image = prepare_data(
            v_path=pv,
            t_path=pt,
            image_preprocess=models[0].im_prep,
            image_aug=None,
        )
        return x_audio, x_text, x_image, fid

    data_iter = DataLoader(
        MultimodalDataset(dataset, read), batch_size=16, num_workers=4
    )

    file_ids = []
    y_pred = []
    for b in data_iter:
        probabilities = []
        for model in models:
            prob = F.softmax(model(b[0].cuda(), b[1].cuda(), b[2].cuda()), dim=1)
            probabilities.append(prob)

        prob = np.stack([x.cpu().numpy() for x in probabilities], axis=2)
        pred = gmean(prob, axis=2).argmax(axis=1)
        y_pred.append(pred)
        file_ids.append(b[-1])

    y_pred = np.hstack(y_pred)
    file_ids = np.hstack(file_ids)
    predicted_classes = ids_to_class(y_pred)

    file_ids = np.array(file_ids, dtype=np.int)
    df = pd.DataFrame(zip(file_ids, predicted_classes), columns=["FileID", "Emotion"])
    df.to_csv(sys.argv[4], index=False)
