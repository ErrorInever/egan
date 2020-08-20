from torchvision import transforms, datasets


def get_emnist_dataset():
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    out_dir = './dataset'
    return datasets.EMNIST(root=out_dir, train=True, transforms=compose, download=True)
