import torch
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int = 64):
    image_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=True,
            download=True,
            transform=image_transforms
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=False,
            transform=image_transforms
        ),
        batch_size=1000,
        shuffle=True,
    )

    return train_loader, test_loader
