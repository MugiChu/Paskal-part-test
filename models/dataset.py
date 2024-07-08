import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Tuple, Optional

sys.path.append(os.path.join(os.getcwd(), ".."))
from utils.transforms import Compose

images_path: str = "../data/JPEGImages"
masks_path: str = "../data/gt_masks"


class PascalPartDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        masks_path: str,
        transform: Any = None,
        mode: str = "train",
    ) -> None:
        self.images_path: str = images_path
        self.masks_path: str = masks_path
        self.transform: Any = transform
        self.mode: str = mode

        self.image_files: list = sorted(os.listdir(images_path))
        self.mask_files: list = sorted(os.listdir(masks_path))

        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of images and masks does not match"

        if mode == "train":
            with open("../data/train_id.txt", "r") as f:
                self.image_ids: list = [line.strip() for line in f]
        elif mode == "val":
            with open("../data/val_id.txt", "r") as f:
                self.image_ids: list = [line.strip() for line in f]
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_id: str = self.image_ids[index]
        image_file: str = os.path.join(self.images_path, image_id + ".jpg")
        mask_file: str = os.path.join(self.masks_path, image_id + ".npy")

        image: np.ndarray = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask: np.ndarray = np.load(mask_file)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_ids)
