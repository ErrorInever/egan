import torch
from torch import nn


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
