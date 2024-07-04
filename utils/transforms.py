import numpy as np
import cv2
import random
from typing import List, Tuple

class Compose:
    def __init__(self, transforms: List) -> None:
        self.transforms: List = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class Resize:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size: Tuple[int, int] = size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask

class Normalize:
    def __init__(self, mean: float, std: float) -> None:
        self.mean: float = mean
        self.std: float = std

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = (image / 255.0 - self.mean) / self.std
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob: float = prob

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask
