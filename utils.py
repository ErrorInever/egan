import torch
from torch import nn
import matplotlib.pyplot as plt


def activation_func(activation, inplace=False):
    """
    Gets instance of function
    :param activation: name function
    :return: instance of function
    """
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.2, inplace=inplace)],
        ['selu', nn.SELU(inplace=inplace)],
        ['none', nn.Identity()]
    ])[activation]


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def latent_space(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = torch.randn(size, 100)
    return n


def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = torch.ones(size, 1)
    return data


def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = torch.zeros(size, 1)
    return data


def show_batch(batch, max_rows=4, max_cols=5):
    images = batch[0]

    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(6, 6))
    for idx, image in enumerate(images):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis("off")
        axes[row, col].imshow(image.squeeze(0).cpu(), cmap="gray")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    fig.tight_layout()
    plt.show()


