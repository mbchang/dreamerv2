import argparse
import copy
from collections import OrderedDict
import numpy as np
import os
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def drop_keys(dictionary, keys):
    new_dict = copy.deepcopy(dictionary)
    for key in keys:
        del new_dict[key]
    return new_dict


class Runner():
    def __init__(self, command='', gpus=[]):
        self.gpus = gpus
        self.command = command
        self.flags = {}

    def add_flag(self, flag_name, flag_values=''):
        self.flags[flag_name] = flag_values

    def add_flags(self, flag_dict):
        for flag_name, flag_values in flag_dict.items():
            self.add_flag(flag_name, flag_values)

    def append_flags_to_command(self, command, flag_dict):
        for flag_name, flag_value in flag_dict.items():
            if type(flag_value) == bool:
                if flag_value == True:
                    command += f' --{flag_name}'
            elif type(flag_value) == list:
                command += f" --{flag_name} {' '.join(element for element in flag_value)}"
            else:
                command += f' --{flag_name} {flag_value}'
        return command

    def command_prefix(self, i):
        prefix = f'CUDA_VISIBLE_DEVICES={self.gpus[i]} DISPLAY=:0 ' if len(self.gpus) > 0 else 'DISPLAY=:0 '
        command = prefix+self.command
        return command

    def command_suffix(self, command):
        if len(self.gpus) == 0:
            command += ' --cpu'
        # command += ' --printf'
        command += ' &'
        return command

    def generate_commands(self, execute=False):
        i = 0
        j = 0
        for flag_dict in product_dict(**self.flags):
            command = self.command_prefix(i)
            command = self.append_flags_to_command(command, flag_dict)
            command = self.command_suffix(command)

            print(command)
            if execute:
                os.system(command)
            if len(self.gpus) > 0:
                i = (i + 1) % len(self.gpus)
            j += 1

            time.sleep(1e-3)

        print('Launched {} jobs'.format(j))

class RunnerWithIDs(Runner):
    def __init__(self, command, gpus):
        Runner.__init__(self, command, gpus)

    def product_dict(self, **kwargs):
        ordered_kwargs_dict = OrderedDict()
        for k, v in kwargs.items():
            if not k == 'seed':
                ordered_kwargs_dict[k] = v

        keys = ordered_kwargs_dict.keys()
        vals = ordered_kwargs_dict.values()

        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def generate_commands(self, execute=False):
        if 'seed' not in self.flags:
            Runner.generate_commands(self, execute)
        else:
            i = 0
            j = 0

            for flag_dict in self.product_dict(**self.flags):
                command = self.command_prefix(i)
                command = self.append_flags_to_command(command, flag_dict)

                # add exp_id: one exp_id for each flag_dict.
                exp_id = ''.join(str(s) for s in np.random.randint(10, size=7))
                command += ' --expid {}'.format(exp_id)

                # command doesn't get modified from here on
                for seed in self.flags['seed']:
                    seeded_command = command
                    seeded_command += ' --seed {}'.format(seed)

                    seeded_command = self.command_suffix(seeded_command)

                    print(seeded_command)
                    if execute:
                        os.system(seeded_command)
                    if len(self.gpus) > 0:
                        i = (i + 1) % len(self.gpus)
                    j += 1

                    time.sleep(1e-3)

            print('Launched {} jobs'.format(j))


def get_it_to_segregate1_10_26_21():
    """
        try: 
            1. warm up until 1e-4, batch size 16, decay 0.5
            2. no warm up, but start at 1e-4
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [16])
    r.add_flag('pred_horizon', [1])
    r.add_flag('expname', ['t3_ph1_b16_lr4e-4_dr5e-1_geb'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [16])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('decay_rate', [0.8])
    r.add_flag('warmup_steps', [0])
    r.add_flag('expname', ['t3_ph1_b16_lr1e-4_dr8e-1_wu0_geb'])
    r.generate_commands(args.for_real)


def get_it_to_segregate2_10_26_21():
    """
        maybe let's increase the batch size?
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('expname', ['t3_ph1_b32_lr4e-4_dr5e-1_geb'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('decay_rate', [0.8])
    r.add_flag('warmup_steps', [0])
    r.add_flag('expname', ['t3_ph1_b32_lr1e-4_dr8e-1_wu0_geb'])
    r.generate_commands(args.for_real)

def get_it_to_segregate3_10_26_21():
    """
        change slot temp

        Conclusion: yes changing slot temp helps
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('slot_temp', [1e-1])
    r.add_flag('expname', ['t3_ph1_b32_lr4e-4_dr5e-1_st1e-1_geb'])
    r.generate_commands(args.for_real)


def get_it_to_segregate4_10_26_21():
    """
        now I need to lower the learning rate from 4e-4. I had tried 2e-4 and that was also too much. Let's try 1e-4.
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('slot_temp', [1e-1])
    r.add_flag('expname', ['t3_ph1_b32_lr1e-4_dr5e-1_st1e-1_geb'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [1e-1])
    r.add_flag('expname', ['t3_ph1_b32_lr1e-4_dr5e-1_st1e-1_wu0_geb'])
    r.generate_commands(args.for_real)


def get_it_to_segregate5_10_27_21():
    """
        seed the imagination with the posterior instead of the prior actually

        turns out that a slot temp of 0.5 instead of 0.1 made it so much better
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [5e-1])
    r.add_flag('expname', ['t3_ph1_b32_lr1e-4_dr5e-1_st5e-1_wu0_imagpost_geb'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [1e-1])
    r.add_flag('expname', ['t3_ph1_b32_lr1e-4_dr5e-1_st1e-1_wu0_imagpost_geb'])
    r.generate_commands(args.for_real)



def longer_pred_horizon_10_27_21():
    """
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [5e-1])
    r.add_flag('pred_horizon', [7])
    r.add_flag('expname', ['t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_wu0_imagpost_geb'])
    r.generate_commands(args.for_real)

def more_frequent_decay_10_27_21():
    """
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('decay_steps', [25000])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [5e-1])
    r.add_flag('pred_horizon', [7])
    r.add_flag('expname', ['t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_wu0_imagpost_ds25e3_geb'])
    r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('model_type', ['factorized_world_model'])
    r.add_flag('num_frames', [3])
    r.add_flag('batch_size', [32])
    r.add_flag('pred_horizon', [1])
    r.add_flag('learning_rate', [1e-4])
    r.add_flag('decay_steps', [50000])
    r.add_flag('warmup_steps', [0])
    r.add_flag('slot_temp', [5e-1])
    r.add_flag('pred_horizon', [7])
    r.add_flag('expname', ['t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_wu0_imagpost_ds50e3_geb'])
    r.generate_commands(args.for_real)


def train_on_dreamer_data_11_1_21():
    """
        test 
            1) whether you can segregate on place_cradle (because pytorch could)
            2) whether the loss curve on balls looks like the loss curve on balls with dreamer launcher
                if so, then the difference is in the data
                if not, then the difference is in the code
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('cfg.dataroot', ['dmc_data/data/balls_whiteball_push/wmoFalse_20211101152738', 'dmc_data/data/dmc_manip_place_cradle/wmoFalse_20211101152629'])
    r.add_flag('cfg.subroot', ['runs/train_on_dreamer_data'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/train_on_offline_data'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)


def find_good_hyperparms_for_slim_11_4_21():
    # first just check that if we do not have slim then the original hyperparameters work
    """

    This was the best:
        T3_H7_tp0.5_B32_etslim_dtslim_lr0.0005_plTrue_20211104121941

    If the learning rate was too high (aka 5e-3), it ended up just not learning at all --> that could potentially be alleviated by warm-up. 
    But lr5e-4 is sufficient. 

    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    r.add_flag('cfg.lnr.model.encoder_type', ['default'])
    r.add_flag('cfg.lnr.model.decoder_type', ['default'])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type'
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1, 2, 3])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [5e-4, 1e-3, 5e-3])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

    # running this after seeing that the default one with the posterior loss does not segregate
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    r.add_flag('cfg.lnr.model.encoder_type', ['default'])
    r.add_flag('cfg.lnr.model.decoder_type', ['default'])
    r.add_flag('cfg.lnr.model.posterior_loss', [True])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)


def find_better_hyperparms_for_slim_11_4_21():
    """
    Given that this was the best from the last iteration,
        T3_H7_tp0.5_B32_etslim_dtslim_lr0.0005_plTrue_20211104121941

    let's find better hyperparameters.

    It seems like having the posterior loss helps break the symmetry better

    Conclusion:
        lr0.0005 still is the best
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [5e-4])
    r.add_flag('cfg.lnr.optim.warmup_steps', [0])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        'lnr.optim.warmup_steps',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[2, 3])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [1e-4])
    r.add_flag('cfg.lnr.optim.warmup_steps', [0, 10000])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        'lnr.optim.warmup_steps',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

def find_better_hyperparms_for_slim2_11_4_21():
    """
    Given that lr0.0005 still is the best, let's figure out what is the right decay rate. warmup_steps=0 seems fine here. 

    on gauss1
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1,2,3])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [5e-4])
    r.add_flag('cfg.lnr.optim.warmup_steps', [0])
    r.add_flag('cfg.lnr.optim.decay_steps', [5000, 10000, 15000])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        'lnr.optim.warmup_steps',
        'lnr.optim.decay_steps',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

def find_better_hyperparms_for_slim3_11_4_21():
    """
    Given that lr0.0005 still is the best, let's figure out what is the right decay rate. warmup_steps=0 seems fine here. 

    on gauss1
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[1,2,3])
    r.add_flag('cfg.dataroot', ['ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_slim'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [5e-4])
    r.add_flag('cfg.lnr.optim.decay_steps', [5000, 10000, 15000])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        'lnr.optim.warmup_steps',
        'lnr.optim.decay_steps',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)

def find_good_hyperparms_for_dmc_cradle_11_5_21():
    """
    first take the best hyperparams you found for balls, and then search around there for manip_place_cradle
    """
    r = RunnerWithIDs(command='python train_slot_attention.py', gpus=[0, 1])
    r.add_flag('cfg.dataroot', ['dmc_data/data/dmc_manip_place_cradle/wmoFalse_20211101152629'])
    r.add_flag('cfg.subroot', ['runs/good_hyperparms_for_dmc_cradle'])
    r.add_flag('cfg.lnr.sess.num_frames', [3])
    r.add_flag('cfg.lnr.sess.pred_horizon', [7])
    r.add_flag('cfg.lnr.optim.batch_size', [32])
    r.add_flag('cfg.lnr.optim.learning_rate', [5e-4])
    r.add_flag('cfg.lnr.optim.decay_steps', [5000, 10000])
    r.add_flag('cfg.lnr.model.posterior_loss', [True, False])
    r.add_flag('cfg.lnr.model.temp', [5e-1])
    to_watch = [
        'lnr.sess.num_frames', 
        'lnr.sess.pred_horizon',
        'lnr.model.temp',
        'lnr.optim.batch_size',
        'lnr.model.encoder_type',
        'lnr.model.decoder_type',
        'lnr.optim.learning_rate',
        'lnr.model.posterior_loss',
        'lnr.optim.warmup_steps',
        'lnr.optim.decay_steps',
        ]
    r.add_flag('cfg.watch', [f'\"{str(tuple(to_watch))}\"'])
    r.generate_commands(args.for_real)




if __name__ == '__main__':
    # get_it_to_segregate_10_26_21()
    # get_it_to_segregate2_10_26_21()
    # get_it_to_segregate3_10_26_21()
    # get_it_to_segregate4_10_26_21()
    # get_it_to_segregate5_10_27_21()
    # longer_pred_horizon_10_27_21()
    # more_frequent_decay_10_27_21()
    # train_on_dreamer_data_11_1_21()
    # find_good_hyperparms_for_slim_11_4_21()
    # find_better_hyperparms_for_slim_11_4_21()
    # find_better_hyperparms_for_slim2_11_4_21()
    # find_better_hyperparms_for_slim3_11_4_21()
    find_good_hyperparms_for_dmc_cradle_11_5_21()


