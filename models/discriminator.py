from torch import nn
from models.generator import Block


class Discriminator(nn.Module):

    def __init__(self, in_features=784, out_features=1, act_type='leaky_relu'):
        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.head = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.block1 = Block(1024, 512, 1, act_type, dropout=True)
        self.block2 = Block(512, 256, 1, act_type, dropout=True)
        self.tail = nn.Sequential(
            nn.Linear(256, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.tail(x)
        return x
