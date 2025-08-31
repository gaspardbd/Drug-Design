from pathlib import Path
from typing import Tuple

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.utils import make_grid, save_image


def get_fashion_mnist_dataloader(data_path: str, batch_size: int, test: bool = False) -> tuple:
    """Load Fashion-MNIST via datasets, apply transforms, and return a DataLoader.

    If data_path is a directory that exists, uses load_from_disk; otherwise uses load_dataset.
    """
    path = Path(data_path)
    if path.exists() and path.is_dir():
        dataset = load_from_disk(str(path))
    else:
        # allow shorthand like "fashion_mnist"
        dataset = load_dataset("fashion_mnist") if data_path == "fashion_mnist" else load_dataset(data_path)

    transform = Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2.0) - 1.0),
        ]
    )

    def transforms_im(examples):
        examples["pixel_values"] = [transform(image) for image in examples["image"]]
        if "image" in examples:
            del examples["image"]
        if "label" in examples:
            del examples["label"]
        return examples

    dataset = dataset.with_transform(transforms_im)
    channels, image_size, _ = dataset["train"][0]["pixel_values"].shape

    dataloader = (
        DataLoader(dataset["test"], batch_size=batch_size)
        if test
        else DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    )

    return dataloader, channels, image_size, len(dataloader)


def normalize_im(images: torch.Tensor) -> torch.Tensor:
    shape = images.shape
    images_flat = images.view(shape[0], -1)
    images_flat = images_flat - images_flat.min(1, keepdim=True)[0]
    images_flat = images_flat / (images_flat.max(1, keepdim=True)[0] + 1e-8)
    return images_flat.view(shape)


def save_image_grid(images: torch.Tensor, out_path: str) -> None:
    grid = make_grid(normalize_im(images))
    save_image(grid, out_path)


