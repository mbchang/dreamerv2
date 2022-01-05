import os.path
import argparse
import math
import torch
import torchvision.utils as vutils
from datetime import datetime
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataloading
# import dvae as dv
import slate
import slot_attn
import transformer
import utils

from absl import app
from absl import flags
from einops import rearrange
import hashlib
import h5py
from loguru import logger as lgr
import ml_collections
from ml_collections.config_flags import config_flags
import pathlib
import pprint
import shutil
import sys
import time
import wandb

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = ml_collections.ConfigDict(dict(
    seed=0,
    epochs=20,
    patience=4,

    checkpoint_path='checkpoint.pt.tar',
    log_path='logs',
    data_path='../slate_data/3dshapes.h5',
    # data_path='../ball_data/whiteballpush/U-Dk4s0n2000t10_ab',

    slate=slate.SLATE.defaults(),

    cpu=False,
    headless=False,
    debug=False,
    jit=False,
    eval=True,
    ))
FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('args', args)


def main(argv):
    if args.debug:
        args.epochs = 2

        args.slate = slate.SLATE.defaults_debug()

        args.cpu = True
        args.headless = False

        prefix = 'db_'
    else:
        prefix = ''

    torch.manual_seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    tf.config.run_functions_eagerly(not args.jit)
    if not args.cpu:
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    def create_expdir(data_name, prefix, args):
        exp_name = f'{prefix}'
        if args.jit:
            exp_name += 'jit_'
        if args.slate.mono_train:
            exp_name += 'mt_'
        if args.slate.slot_model.perceiver_output:
            exp_name += 'po'
        exp_name += f'esd{args.slate.slot_model.einsum_dict}_'
        exp_name += f'sgi{args.slate.stop_gradient_input}_'
        exp_name += f'sgo{args.slate.stop_gradient_output}_'
        exp_name += f'{args.slate.dvae.cnn_type}_'
        exp_name += f'{datetime.now():%Y%m%d%H%M%S}'
        exp_dir = pathlib.Path(os.path.join(data_name, exp_name))
        return exp_dir

    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    data_name = pathlib.Path(args.data_path).with_suffix('').name
    exp_dir = create_expdir(data_name, prefix, args)
    log_dir = pathlib.Path(os.path.join(args.log_path, exp_dir))
    writer = SummaryWriter(log_dir)
    writer.add_text('hparams', arg_str)

    wandb.init(
        config=args.to_dict(),
        project='slate_pytorch',
        dir=log_dir,
        group=f'{prefix}{pathlib.Path(args.log_path).name}_{data_name}',
        job_type='train',
        id=f'slate_{data_name}_{log_dir.name}')

    lgr.remove()   # remove default handler
    lgr.add(os.path.join(log_dir, 'debug.log'))
    if not args.headless:
        lgr.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
        lgr.info(f'Logdir: {log_dir}')

    lgr.info(f'\n{args}')

    # save source code
    code_dir = os.path.join(log_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for src_file in [x for x in os.listdir('.') if '.py' in x]:
        shutil.copy2(src_file, code_dir)

    lgr.info('Building dataloaders...')
    if '3dshapes' in args.data_path:
        databuilder = dataloading.DebugShapes3D if args.debug else dataloading.Shapes3D
        train_loader = dataloading.DataLoader(
            dataset = databuilder(root=args.data_path, phase='train'), batch_size=args.slate.batch_size)
        val_loader = dataloading.DataLoader(
            dataset = databuilder(root=args.data_path, phase='val'), batch_size=args.slate.batch_size)
        train_epoch_size = len(train_loader)
        val_epoch_size = len(val_loader)
    # elif 'ball' in args.data_path:
    #     args.eval = False
    #     assert not args.eval
    #     train_loader = dataloading.WhiteBallDataLoader(h5=h5py.File(f'{args.data_path}.h5', 'r'), batch_size=args.slate.batch_size)
    #     if args.debug:
    #         train_loader.num_batches = 80
    #         train_epoch_size = 80
    #     else:
    #         train_loader.num_batches = 8000
    #         train_epoch_size = 8000
    elif 'dmc' in args.data_path:
        args.eval = False
        assert not args.eval
        train_loader = dataloading.DMCDatLoader(dataroot=args.data_path, batch_size=args.slate.batch_size)
        if args.debug:
            train_loader.num_batches = 80
            train_epoch_size = 80
        else:
            train_loader.num_batches = 8000
            train_epoch_size = 8000
    else:
        raise NotImplementedError

    # log_interval = train_epoch_size // 10

    lgr.info('Building model...')
    model = slate.SLATE(args.slate)
    lgr.info(model)

    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        stagnation_counter = checkpoint['stagnation_counter']
        lr_decay_factor = checkpoint['lr_decay_factor']
        model.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0
        stagnation_counter = 0
        lr_decay_factor = args.slate.lr_decay_factor


    lgr.info('initialize with input...')
    image = train_loader.get_batch()
    if isinstance(image, dict):
        image = image['image']
    model(image, tau=tf.constant(1.0), hard=True)

    lgr.info('Begin training.')
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        
        model.train()

        for batch in range(train_loader.num_batches):
            image = train_loader.get_batch()
            if isinstance(image, dict):
                image = image['image']

            global_step = epoch * train_epoch_size + batch


            t0 = time.time()

            if args.slate.mono_train:
                if args.slate.smooth_input:
                    loss, outputs, metrics = model.monolithic_train_step_smooth_input(image)
                else:
                    loss, outputs, metrics = model.monolithic_train_step(image)
            else:
                loss, outputs, metrics = model.train_step(image)

            recon = outputs['dvae']['recon']
            z_hard = outputs['dvae']['z_hard']
            attns = outputs['slot_model']['attns']
            tau = outputs['iterates']['tau']

            mse = metrics['dvae']['mse']
            cross_entropy = metrics['slot_model']['cross_entropy']

            if global_step % args.slate.log_interval == 0:
                lgr.info('Train Step: {:3} \t Loss: {:F} \t MSE: {:F} \t Time: {:F}'.format(
                      global_step, loss.numpy(), mse.numpy(), time.time()-t0))

                wandb.log({
                    'train/loss': loss.numpy(),
                    'train/cross_entropy': cross_entropy.numpy(),
                    'train/mse': mse.numpy(),
                    'train/tau': tau,
                    'train/lr': model.dvae_optimizer.lr.numpy(),
                    'train/lr_main': model.main_optimizer.lr.numpy(),
                    'train/itr': global_step
                    }, step=global_step)

        t0 = time.time()
        n = 32
        gen_img, _, _ = model.reconstruct_autoregressive(image[:n])
        vis_recon = rearrange(slate.SLATE.visualize(image[:n], attns[:n], recon[:n], gen_img, val_loader.unnormalize_obs), 'b n c h w -> c (b h) (n w)')
        lgr.info(f"TRAIN: Autoregressive generation took {time.time() - t0} seconds. Image hash: {utils.hash_sha1(vis_recon)}")

        writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), vis_recon.numpy())
        plt.imsave(f'{log_dir}/train_{epoch+1}.png', rearrange(vis_recon.numpy(), 'c h w -> h w c'))
        
        if args.eval:
            model.eval()
            
            val_cross_entropy_relax = 0.
            val_mse_relax = 0.
            
            val_cross_entropy = 0.
            val_mse = 0.
            
            t0 = time.time()
            for batch in range(val_loader.num_batches):
                image = val_loader.get_batch()
                if isinstance(image, dict):
                    image = image['image']

                _, outs_relax, mets_relax = model(image, tf.constant(tau), False)
                
                _, outs, mets = model(image, tf.constant(tau), True)

                val_cross_entropy_relax += mets_relax['slot_model']['cross_entropy'].numpy()
                val_mse_relax += mets_relax['dvae']['mse'].numpy()
                
                val_cross_entropy += mets['slot_model']['cross_entropy'].numpy()
                val_mse += mets['dvae']['mse'].numpy()

            val_cross_entropy_relax /= (val_epoch_size)
            val_mse_relax /= (val_epoch_size)
            
            val_cross_entropy /= (val_epoch_size)
            val_mse /= (val_epoch_size)
            
            val_loss_relax = val_mse_relax + val_cross_entropy_relax
            val_loss = val_mse + val_cross_entropy

            lgr.info('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t Time: {:F}'.format(
                epoch+1, val_loss, val_mse, time.time() - t0))

            if val_loss < best_val_loss:
                stagnation_counter = 0
                best_val_loss = val_loss
                best_epoch = epoch + 1

                # if 50 <= epoch:
                t0 = time.time()
                n = 32
                gen_img, _, _ = model.reconstruct_autoregressive(image[:n])
                vis_recon = rearrange(slate.SLATE.visualize(image[:n], outs['slot_model']['attns'][:n], outs['dvae']['recon'][:n], gen_img, val_loader.unnormalize_obs), 'b n c h w -> c (b h) (n w)')
                lgr.info(f"VAL: Autoregressive generation took {time.time() - t0} seconds. Image hash: {utils.hash_sha1(vis_recon)}")

                writer.add_image('VAL_recon/epoch={:03}'.format(epoch + 1), vis_recon.numpy())
                plt.imsave(f'{log_dir}/val_{epoch+1}.png', rearrange(vis_recon.numpy(), 'c h w -> h w c'))

            else:
                stagnation_counter += 1
                if stagnation_counter >= args.patience:
                    lr_decay_factor = lr_decay_factor / 2.0
                    stagnation_counter = 0

            # writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

            wandb.log({
                'val/loss_relax': val_loss_relax,
                'val/cross_entropy_relax': val_cross_entropy_relax,
                'val/mse_relax': val_mse_relax,
                'val/loss': val_loss,
                'val/cross_entropy': val_cross_entropy,
                'val/mse': val_mse,
                'val/best_loss': best_val_loss,
                'val/epoch': epoch+1,
                }, step=global_step)

            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'best_val_loss': best_val_loss,
            #     'best_epoch': best_epoch,
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'stagnation_counter': stagnation_counter,
            #     'lr_decay_factor': lr_decay_factor,
            # }

            # torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

            lgr.info('====> Best Loss = {:F} @ Epoch {} \t Time per epoch: {:F}'.format(best_val_loss, best_epoch, time.time() - t_epoch))

    writer.close()

if __name__ == "__main__":
    app.run(main)