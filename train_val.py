import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from config.conf import cfg
from data import get_emnist_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from metric_logger import Logger
from train.train import train_one_epoch
from utils import latent_space


def parse_args():
    parser = argparse.ArgumentParser(description='Emnist')
    parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
    parser.add_argument('--save_models', dest='save_models', help='save model', action='store_true')
    parser.add_argument('--tensorboard', dest='tensorboard', help='use tensorboard', action='store_true')
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = get_emnist_dataset()
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.BATCH_SIZE)

    generator = Generator()
    discriminator = Discriminator()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE)

    criterion = torch.nn.BCELoss()

    logger = Logger(model_name='EGAN', dataset_name='EMNIST', ls_api_key=args.api_key, tensorboard=args.tensorboard)

    num_samples = 16
    static_z = latent_space(num_samples)
    for epoch in range(cfg.NUM_EPOCH):
        train_one_epoch(generator, discriminator, dataloader, d_optimizer, g_optimizer, criterion, epoch, logger,
                        static_z, num_samples, freq=100)
        if args.save_models:
            logger.save_models(generator, discriminator, epoch)
