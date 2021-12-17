import torch
import glob
import os
from pathlib import Path
from contextlib import contextmanager


def get_quantiles(IOU, quantiles=[.001, .005, .01, .05, .1]):
    return np.quantile(IOU, quantiles)


@contextmanager
def in_dir(directory):
    before = Path.cwd()
    Path(directory).mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(directory)
        print(f"now in {directory}")
        yield None
    finally:
        os.chdir(before)


def move_ims(basepath, count, set_name='val'):
    os.makedirs(f"{basepath}/{set_name}/images/")
    os.makedirs(f"{basepath}/{set_name}/masks/")
    imset = os.listdir(f"{basepath}/train/images")
    for im in imset[:count]:
        os.rename(f"{basepath}/train/images/{im}",
                  f"{basepath}/{set_name}/images/{im}")
        os.rename(f"{basepath}/train/masks/{im}",
                  f"{basepath}/{set_name}/masks/{im}")
