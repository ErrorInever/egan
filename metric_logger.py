import losswise
import torch
import os
import errno
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from config.conf import cfg
from matplotlib import pyplot as plt
from IPython import display


class Logger:

    def __init__(self, model_name, dataset_name, ls_api_key, tensorboard=False):
        """
        :param model_name: ``str``, model name
        :param dataset_name: ``str``, dataset name
        :param ls_api_key:  ``str``, losswise API key
        :param tensorboard: if True - save tensorboard metrics
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_subdir = '{}/{}'.format(model_name, dataset_name)
        self.comment = '{}_{}'.format(self.model_name, self.dataset_name)
        self.ls_api_key = ls_api_key
        self.tensorboard = tensorboard

        if self.ls_api_key:
            losswise.set_api_key(ls_api_key)
            self.session = losswise.Session(
                tag='egan',
                max_iter=cfg.NUM_EPOCH,
                track_git=False
            )
            self.gen_graph_loss = self.session.graph('gen_loss', kind='min', display_interval=1)
            self.dis_graph_loss = self.session.graph('dis_loss', kind='min', display_interval=1)
            self.dis_acc_real = self.session.graph('acc_real', kind='max', display_interval=1)
            self.dis_acc_gen = self.session.graph('acc_gen', kind='max', display_interval=1)

        if self.tensorboard:
            self.metric_logger = SummaryWriter(comment=self.comment)

    def log(self, dis_loss, gen_loss, dis_pred_real, dis_pred_gen, epoch, n_batch, num_batches):
        """
        Logging training values
        :param dis_loss: discriminator loss
        :param gen_loss: generator loss
        :param dis_pred_real: accuracy predict real image
        :param dis_pred_gen: accuracy predict generated image
        :param epoch: current epoch
        :param n_batch: current batch
        :param num_batches: count batches
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss.data.cpu().numpy()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)

        if self.ls_api_key:
            self.dis_graph_loss.append(step, {'train_dis_loss:': dis_loss})
            self.gen_graph_loss.append(step, {'train_gen_loss:': gen_loss})
            # D(x)
            #self.dis_acc_real.append(step, {'train_real_acc:': dis_pred_real})
            # D(G(x))
            #self.dis_acc_gen.append(step, {'train_gen_acc:': dis_pred_gen})

        if self.tensorboard:
            self.metric_logger.add_scalar('loss/dis', dis_loss, step)
            self.metric_logger.add_scalar('loss/gen', gen_loss, step)
            # self.metric_logger.add_scalar('acc/D(x)', dis_pred_real, step)
            # self.metric_logger.add_scalar('acc/D(G(X))', dis_pred_gen, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):

        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if format == 'NCHW':
            images = images.transpose(1, 3).detach().cpu().numpy().astype(np.uint8)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)
        # Add horizontal images to tensorboard
        self.metric_logger.add_image(img_name, horizontal_grid, step)
        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.metric_logger.close()
        self.session.done()

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, dis_loss, gen_loss, dis_pred_real, gen_pred_real):
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.data().cpu().numpy()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.data().cpu().numpy()
        if isinstance(dis_pred_real, torch.autograd.Variable):
            dis_pred_real = dis_pred_real.data().cpu().numpy()
        if isinstance(gen_pred_real, torch.autograd.Variable):
            gen_pred_real = gen_pred_real.data().cpu().numpy()

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(dis_pred_real.mean(), gen_pred_real.mean()))

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
