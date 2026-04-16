import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return train and test DataLoaders for CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def labels_to_onehot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert integer labels to one-hot vectors (required for MSE loss)."""
    return F.one_hot(labels, num_classes=num_classes).float()
