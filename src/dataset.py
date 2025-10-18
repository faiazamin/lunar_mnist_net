from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.MNIST(root="data", train=True, transform=tf, download=True)
    test  = datasets.MNIST(root="data", train=False, transform=tf, download=True)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test,  batch_size=batch_size)
    )