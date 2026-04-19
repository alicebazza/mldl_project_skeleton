import torch
from typing import Optional, Callable

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters of a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def permute_pixels(images: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    """Permutes the pixels in each image in the batch."""
    if perm is None:
        return images

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]

    images = images.view(batch_size, n_channels, -1)
    images = images[..., perm]
    images = images.view(batch_size, n_channels, w, h)
    return images


def make_averager() -> Callable[[Optional[float]], float]:
    """Returns a function that maintains a running average."""
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        nonlocal count, total

        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager
