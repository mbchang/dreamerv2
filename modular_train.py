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

import shapes_3d
from slate import SLATE, create_tokens, overlay_attention

from einops import rearrange
from loguru import logger as lgr
import pathlib
import pprint
import shutil
import sys
import time
import wandb

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='logs')
parser.add_argument('--data_path', default='../slate_data/3dshapes.h5')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_main', type=float, default=1e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)

parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=1024)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--num_slot_heads', type=int, default=1)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--jit', action='store_true')

args = parser.parse_args()

if args.debug:
    args.num_workers = 0
    args.batch_size = 5
    args.epochs = 2

    args.lr_warmup_steps = 3

    args.num_dec_blocks = 2
    args.vocab_size = 32
    args.d_model = 16
    args.num_heads = 2

    args.num_iterations = 2
    args.slot_size = 16
    args.mlp_hidden_size = 16
    args.tau_steps = 3

    args.cpu = True
    args.headless = False
    # args.jit = False

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

pprint.pprint(arg_str_list)

wandb.init(
    config=vars(args),
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

# save source code
code_dir = os.path.join(log_dir, 'code')
os.makedirs(code_dir, exist_ok=True)
for src_file in [x for x in os.listdir('.') if '.py' in x]:
    shutil.copy2(src_file, code_dir)

lgr.info('Building dataset...')
databuilder = shapes_3d.DebugShapes3D if args.debug else shapes_3d.Shapes3D
train_dataset = databuilder(root=args.data_path, phase='train')
val_dataset = databuilder(root=args.data_path, phase='val')

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

lgr.info('Building dataloaders...')
train_loader = shapes_3d.DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = shapes_3d.DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 10

lgr.info('Building model...')
model = SLATE(args)
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
    lr_decay_factor = 1.0


lgr.info('initialize with input...')
model(train_loader.get_batch(), tau=tf.constant(1.0), hard=args.hard)

dvae_optimizer = tf.keras.optimizers.Adam(args.lr_dvae, epsilon=1e-08)
main_optimizer = tf.keras.optimizers.Adam(args.lr_main, epsilon=1e-08)

def linear_warmup(step, start_value, final_value, start_step, final_step):
    
    assert start_value <= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    
    return value


def cosine_anneal(step, start_value, final_value, start_step, final_step):
    
    assert start_value >= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    
    return value


def visualize(image, recon_orig, gen, attns, N=8):
    unsqueeze = lambda x: rearrange(x, 'b c h w -> b 1 c h w')
    image, recon_orig, gen = map(unsqueeze, (image[:N], recon_orig[:N], gen[:N]))
    attns = attns[:N]
    return rearrange(tf.concat((image, recon_orig, gen, attns), axis=1), 'b n c h w -> c (b h) (n w)')


def f32(x):
    return tf.cast(x, tf.float32)


@tf.function
def dvae_train_step(dvae, image, tau, hard):
    with tf.GradientTape() as tape:
        recon, z_hard, mse = dvae(image, tau, hard)
    gradients = tape.gradient(mse, dvae.trainable_weights)
    return tf.clip_by_value(recon, 0., 1.), z_hard, mse, gradients


@tf.function
def slot_model_train_step(slot_model, z_transformer_input, z_transformer_target):
    with tf.GradientTape() as tape:
        attns, cross_entropy = slot_model(z_transformer_input, z_transformer_target)
    gradients = tape.gradient(cross_entropy, slot_model.trainable_weights)
    return attns, cross_entropy, gradients



lgr.info('Begin training.')
for epoch in range(start_epoch, args.epochs):
    t_epoch = time.time()
    
    model.train()

    for batch in range(train_loader.num_batches):
        image = train_loader.get_batch()

        global_step = epoch * train_epoch_size + batch
        
        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        dvae_optimizer.lr = f32(lr_decay_factor * args.lr_dvae)
        main_optimizer.lr = f32(lr_decay_factor * lr_warmup_factor * args.lr_main)

        t0 = time.time()

        recon, z_hard, mse, gradients = dvae_train_step(model.dvae, image, tf.constant(tau), args.hard)
        dvae_optimizer.apply_gradients(zip(gradients, model.dvae.trainable_weights))

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(z_hard))

        attns, cross_entropy, gradients = slot_model_train_step(model.slot_model, z_transformer_input, z_transformer_target)
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        main_optimizer.apply_gradients(zip(gradients, model.slot_model.trainable_weights))

        loss = mse + cross_entropy

        _, _, H_enc, W_enc = z_hard.shape
        attns = overlay_attention(attns, image, H_enc, W_enc)

        with torch.no_grad():
            if batch % log_interval == 0:
                lgr.info('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F} \t Time: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.numpy(), mse.numpy(), time.time()-t0))
                
                writer.add_scalar('TRAIN/loss', loss.numpy(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.numpy(), global_step)
                writer.add_scalar('TRAIN/mse', mse.numpy(), global_step)
                
                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', dvae_optimizer.lr.numpy(), global_step)
                writer.add_scalar('TRAIN/lr_main', main_optimizer.lr.numpy(), global_step)

                wandb.log({
                    'train/loss': loss.numpy(),
                    'train/cross_entropy': cross_entropy.numpy(),
                    'train/mse': mse.numpy(),
                    'train/tau': tau,
                    'train/lr_dvae': dvae_optimizer.lr.numpy(),
                    'train/lr_main': main_optimizer.lr.numpy(),
                    'train/itr': global_step
                    }, step=global_step)

    with torch.no_grad():
        t0 = time.time()
        gen_img = model.reconstruct_autoregressive(image[:32])
        lgr.info(f'TRAIN: Autoregressive generation took {time.time() - t0} seconds.')
        vis_recon = visualize(image, recon, gen_img, attns, N=32)
        writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), vis_recon.numpy())
    
    with torch.no_grad():
        model.eval()
        
        val_cross_entropy_relax = 0.
        val_mse_relax = 0.
        
        val_cross_entropy = 0.
        val_mse = 0.
        
        t0 = time.time()
        for batch in range(val_loader.num_batches):
            image = val_loader.get_batch()

            (recon_relax, cross_entropy_relax, mse_relax, attns_relax) = model(image, tf.constant(tau), False)
            
            (recon, cross_entropy, mse, attns) = model(image, tf.constant(tau), True)
            
            val_cross_entropy_relax += cross_entropy_relax.numpy()
            val_mse_relax += mse_relax.numpy()
            
            val_cross_entropy += cross_entropy.numpy()
            val_mse += mse.numpy()

        val_cross_entropy_relax /= (val_epoch_size)
        val_mse_relax /= (val_epoch_size)
        
        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        
        val_loss_relax = val_mse_relax + val_cross_entropy_relax
        val_loss = val_mse + val_cross_entropy

        writer.add_scalar('VAL/loss_relax', val_loss_relax, epoch+1)
        writer.add_scalar('VAL/cross_entropy_relax', val_cross_entropy_relax, epoch + 1)
        writer.add_scalar('VAL/mse_relax', val_mse_relax, epoch+1)

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)

        lgr.info('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t Time: {:F}'.format(
            epoch+1, val_loss, val_mse, time.time() - t0))

        if val_loss < best_val_loss:
            stagnation_counter = 0
            best_val_loss = val_loss
            best_epoch = epoch + 1

            if 50 <= epoch:
                t0 = time.time()
                gen_img = model.reconstruct_autoregressive(image)
                lgr.info(f'VAL: Autoregressive generation took {time.time() - t0} seconds.')
                vis_recon = visualize(image, recon, gen_img, attns, N=32)
                writer.add_image('VAL_recon/epoch={:03}'.format(epoch + 1), vis_recon.numpy())

        else:
            stagnation_counter += 1
            if stagnation_counter >= args.patience:
                lr_decay_factor = lr_decay_factor / 2.0
                stagnation_counter = 0

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

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
