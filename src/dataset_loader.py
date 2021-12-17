import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from torch.utils.data import Subset, ConcatDataset, Dataset, DataLoader
import numpy as np
import torch

TOTAL_REAL_TRAIN = 4000
TOTAL_SYNTHETIC_TRAIN = 2234


def create_dataloader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=2)


def create_dataset(from_real,
                   from_synthetic,
                   stage,
                   spacial_transforms=[],
                   transform_x=[],
                   dataset_dir='/content/dataset'):

    # create mixture of synthetic and real dataset

    mix = []

    for count, data_type in zip([from_real, from_synthetic], ['real', 'synthetic']):

        if count != 0:

            path = os.path.join(dataset_dir, data_type, stage)

            dset = DatasetSegmentation(
                path, spacial_transforms=spacial_transforms, transform_x=transform_x)

            mix.append(dset if count == -1 else Subset(dset,
                       np.random.choice(len(dset), count, replace=False)))

    dset = ConcatDataset(mix)

    assert len(dset) > 0

    return dset


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, spacial_transforms=[], transform_x=[]):

        super(DatasetSegmentation, self).__init__()

        self.folder_path = folder_path
        self.img_files = sorted(
            glob.glob(os.path.join(folder_path, "images", "*")))

        self.spacial_transforms = transforms.Compose(spacial_transforms)
        self.transform_x = transforms.Compose(
            [transforms.ToTensor()] + transform_x)
        self.transform_y = transforms.Compose(
            [
                transforms.ToTensor(),
                # binarize the images, sometimes they are not 0 or 1
                transforms.Lambda(lambda t: (t > 0.5).float()),
            ]
        )

    def _maskname(self, index):
        return os.path.join(
            self.folder_path, "masks", os.path.basename(self.img_files[index])
        )

    def __getitem__(self, index):

        x = Image.open(self.img_files[index])
        y = Image.open(self._maskname(index))

        x_t = self.transform_x(x)
        y_t = self.transform_y(y)

        # apply spacial transformation to both x and y simultaneously
        xy_t = self.spacial_transforms(torch.cat([x_t, y_t], dim=0))
        x_t = xy_t[:3]
        y_t = xy_t[3:]

        return x_t, y_t

    def __len__(self):
        return len(self.img_files)
