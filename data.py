from torchvision import transforms, datasets


def get_emnist_dataset():
    compose = transforms.Compose([
        transforms.RandomRotation(degrees=(-90, -90)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    out_dir = './dataset'
    return datasets.EMNIST(root=out_dir, split='bymerge', train=True, transform=compose, download=True)
