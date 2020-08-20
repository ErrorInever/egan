from torch import nn
from utils import activation_func


class Block(nn.Module):
    """Block layer"""
    def __init__(self, in_channels, out_channels, repeats, act_type, dropout=False):
        """
        :param in_channels: ``int``, in channels
        :param out_channels: ``int``, out channels
        :param repeats: `int``, count of repeat block
        :param act_type: ``str``, type of activation function
        """
        super(Block, self).__init__()
        self.act = activation_func(act_type)

        layers = []
        for i in range(repeats):
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(self.act)
            if dropout:
                layers.append(nn.Dropout(0.3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):

    def __init__(self, in_features=100, out_features=784, act_type='leaky_relu'):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.block1 = Block(256, 512, 1, act_type)
        self.block2 = Block(512, 1024, 1, act_type)
        self.tail = nn.Sequential(
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.tail(x)
        return x
