import torch
from metric_logger import Logger
from utils import latent_space, images_to_vectors, ones_target, zeros_target, vectors_to_images
from config.conf import cfg


def train_generator(discriminator, g_optimizer, criterion, device, gen_data):
    m = gen_data.size(0)
    ones_labels = ones_target(m).to(device)

    g_optimizer.zero_grad()

    prediction = discriminator(gen_data)
    loss = criterion(prediction, ones_labels)
    loss.backward()
    g_optimizer.step()
    return loss


def train_discriminator(discriminator, d_optimizer, criterion, device, real_data, gen_data):
    m = real_data.size(0)
    ones_labels = ones_target(m).to(device)
    zeros_labels = zeros_target(m).to(device)

    d_optimizer.zero_grad()

    # 1. train on real data
    prediction_real = discriminator(real_data)
    loss_real = criterion(prediction_real, ones_labels)
    loss_real.backward()

    # 2. train on fake data
    prediction_gen = discriminator(gen_data)
    loss_gen = criterion(prediction_gen, zeros_labels)
    loss_real.backward()

    # update weights
    d_optimizer.step()

    loss = loss_real + loss_gen

    return loss, prediction_real, prediction_gen


def train_one_epoch(generator, discriminator, dataloader, d_optimizer, g_optimizer,
                    criterion, epoch, device, logger, static_z_vector, num_samples, freq=100):
    for n_batch, (real_batch, _) in enumerate(dataloader):
        m = real_batch.size(0)

        real_batch = real_batch.to(device)

        # 1. train discriminator
        z_vector = latent_space(m).to(device)
        pz_distribution = generator(z_vector).detach()
        px_distribution = images_to_vectors(real_batch)
        dis_loss, prediction_real, prediction_gen = train_discriminator(
            discriminator, d_optimizer, criterion, device, real_data=px_distribution, gen_data=pz_distribution)

        # 2. train generator
        z_vector = latent_space(m).to(device)
        pz_distribution = generator(z_vector)
        gen_loss = train_generator(discriminator, g_optimizer, criterion, device, gen_data=pz_distribution)

        logger.log(dis_loss, gen_loss, prediction_real, prediction_gen, epoch, n_batch, len(dataloader))

        if n_batch % freq == 0:
            static_pz = generator(static_z_vector)
            static_images = vectors_to_images(static_pz)
            logger.log_images(static_images, num_samples, epoch, n_batch, len(dataloader))
            logger.display_status(epoch, cfg.NUM_EPOCH, n_batch, len(dataloader),
                                  dis_loss, gen_loss, prediction_real, prediction_gen)
