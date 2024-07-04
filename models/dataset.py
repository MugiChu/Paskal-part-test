import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Tuple, Optional

sys.path.append(os.path.join(os.getcwd(), ".."))
from utils.transforms import Compose


images_path = "../data/JPEGImages"
masks_path = "../data/gt_masks"


class PascalPartDataset(Dataset):
    def __init__(self, images_path: str, masks_path: str, transform: Optional[Any] = None) -> None:
        self.images_path: str = images_path
        self.masks_path: str = masks_path
        self.transform: Optional[Any] = transform

        self.image_files: list = sorted(os.listdir(images_path))
        self.mask_files: list = sorted(os.listdir(masks_path))

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks does not match"

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_file: str = os.path.join(self.images_path, self.image_files[index])
        mask_file: str = os.path.join(self.masks_path, self.mask_files[index])

        image: np.ndarray  = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask: np.ndarray = np.load(mask_file)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)
