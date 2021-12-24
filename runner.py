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
        self.i = 0

    def add_flag(self, flag_name, flag_values=''):
        self.flags[flag_name] = flag_values

    def add_flags(self, flag_dict):
        for flag_name, flag_values in flag_dict.items():
            self.add_flag(flag_name, flag_values)

    def append_flags_to_command(self, command, flag_dict):
        for flag_name, flag_value in flag_dict.items():
            if type(flag_value) == bool:
                # if flag_value == True:
                #     command += f' --{flag_name}'
                command += f' --{flag_name}={flag_value}'
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
        # i = 0
        j = 0
        for flag_dict in product_dict(**self.flags):
            command = self.command_prefix(self.i)
            command = self.append_flags_to_command(command, flag_dict)
            command = self.command_suffix(command)

            print(command)
            if execute:
                os.system(command)
            if len(self.gpus) > 0:
                self.i = (self.i + 1) % len(self.gpus)
            j += 1

            time.sleep(1)

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
            # i = 0
            j = 0

            for flag_dict in self.product_dict(**self.flags):
                command = self.command_prefix(self.i)
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
                        self.i = (self.i + 1) % len(self.gpus)
                    j += 1

                    time.sleep(1)

            print('Launched {} jobs'.format(j))



def train_model_sanity_10_22_21():
    """
        Just want to test whether the pipeline for training the model works

        I'm manually creating these just because I have not done the "watch" thing yet

        Note though that the buffer has only been prefilled to 1000
    """
    # python dreamerv2/train.py --logdir runs/model/default_cheetah_pf10000 --configs dmc_vision --task dmc_cheetah_run --agent causal

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('logdir', ['runs/model/default_cheetah_pf10000'])
    r.generate_commands(args.for_real)

    # python dreamerv2/train.py --logdir runs/model/sa_ns1 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --encoder_type slimslot 

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('rssm.num_slots', [1])
    r.add_flag('logdir', ['runs/model/sa_ns1_pf10000'])
    r.generate_commands(args.for_real)

    # python dreamerv2/train.py --logdir runs/model/sa_ns2 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --encoder_type slimslot --rssm.num_slots 2

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[2])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('rssm.num_slots', [2])
    r.add_flag('logdir', ['runs/model/sa_ns2_pf10000'])
    r.generate_commands(args.for_real)

    # python dreamerv2/train.py --logdir runs/model/sa_ns4 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --encoder_type slimslot --rssm.num_slots 4

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('rssm.num_slots', [4])
    r.add_flag('logdir', ['runs/model/sa_ns4_pf10000'])
    r.generate_commands(args.for_real)




def train_model_single_step_sanity_10_22_21():
    """
        Just want to test whether the pipeline for training the model works

        I'm manually creating these just because I have not done the "watch" thing yet

        Note though that the buffer has only been prefilled to 1000
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('logdir', ['runs/model/default_cheetah_t1'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('rssm.num_slots', [1])
    r.add_flag('logdir', ['runs/model/sa_ns1_t1'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('rssm.num_slots', [2])
    r.add_flag('logdir', ['runs/model/sa_ns2_t1'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('rssm.num_slots', [4])
    r.add_flag('logdir', ['runs/model/sa_ns4_t1'])
    r.generate_commands(args.for_real)



def train_model_two_step_sanity_10_22_21():
    """
        Just want to test whether the pipeline for training the model works

        I'm manually creating these just because I have not done the "watch" thing yet

        Note though that the buffer has only been prefilled to 1000
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [2])
    r.add_flag('logdir', ['runs/model/default_cheetah_t2'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [2])
    r.add_flag('rssm.num_slots', [1])
    r.add_flag('logdir', ['runs/model/sa_ns1_t2'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [2])
    r.add_flag('rssm.num_slots', [2])
    r.add_flag('logdir', ['runs/model/sa_ns2_t2'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [2])
    r.add_flag('rssm.num_slots', [4])
    r.add_flag('logdir', ['runs/model/sa_ns4_t2'])
    r.generate_commands(args.for_real)



def train_model_balls_sanity_10_23_21():
    """
        Just want to test whether the pipeline for training the model works

        I'm manually creating these just because I have not done the "watch" thing yet

        Note though that the buffer has only been prefilled to 1000
    """
    # r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    # r.add_flag('configs', ['dmc_vision'])
    # r.add_flag('task', ['balls_whiteball_push'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [10000])
    # r.add_flag('log_every', [100])  # ends up being double for some reason
    # r.add_flag('eval_every', [500])  # ends up being double for some reason
    # r.add_flag('dataset.length', [2])
    # r.add_flag('logdir', ['runs/model/balls_df_t2'])
    # r.generate_commands(args.for_real)

    # r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    # r.add_flag('configs', ['dmc_vision'])
    # r.add_flag('task', ['balls_whiteball_push'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    # r.add_flag('rssm.update', ['slot_attention'])
    # r.add_flag('decoder_type', ['slot'])
    # r.add_flag('encoder_type', ['slimslot'])
    # r.add_flag('prefill', [10000])
    # r.add_flag('log_every', [100])  # ends up being double for some reason
    # r.add_flag('eval_every', [500])  # ends up being double for some reason
    # r.add_flag('dataset.length', [2])
    # r.add_flag('rssm.num_slots', [1])
    # r.add_flag('logdir', ['runs/model/balls_sa_ns1_t2'])
    # r.generate_commands(args.for_real)

    # r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    # r.add_flag('configs', ['dmc_vision'])
    # r.add_flag('task', ['balls_whiteball_push'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    # r.add_flag('rssm.update', ['slot_attention'])
    # r.add_flag('decoder_type', ['slot'])
    # r.add_flag('encoder_type', ['slimslot'])
    # r.add_flag('prefill', [10000])
    # r.add_flag('log_every', [100])  # ends up being double for some reason
    # r.add_flag('eval_every', [500])  # ends up being double for some reason
    # r.add_flag('dataset.length', [2])
    # r.add_flag('rssm.num_slots', [5])
    # r.add_flag('logdir', ['runs/model/balls_sa_ns5_t2'])
    # r.generate_commands(args.for_real)

    # one step

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('logdir', ['runs/model/balls_df_t1'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('rssm.num_slots', [1])
    r.add_flag('logdir', ['runs/model/balls_sa_ns1_t1'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('prefill', [10000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [1])
    r.add_flag('rssm.num_slots', [5])
    r.add_flag('rssm.stoch', [40])  # must be divisible by num_slots
    r.add_flag('logdir', ['runs/model/balls_sa_ns5_t1'])
    r.generate_commands(args.for_real)



def train_model_balls_fwm_10_27_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.dynamics', ['slim_cross_attention'])
    r.add_flag('rssm.update', ['slot_attention'])
    r.add_flag('decoder_type', ['slot'])
    r.add_flag('encoder_type', ['slimslot'])
    r.add_flag('dataset.batch', [32])
    r.add_flag('wm', ['fwm'])
    r.add_flag('precision', [32])
    r.add_flag('prefill', [20000])
    r.add_flag('log_every', [100])  # ends up being double for some reason
    r.add_flag('eval_every', [500])  # ends up being double for some reason
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('rssm.num_slots', [5])
    r.add_flag('logdir', ['runs/model/balls_sa_ns5_t3_fwm'])
    r.generate_commands(args.for_real)

def comparison_for_train_on_dreamer_data_11_1_21():
    """
    CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 1.0 &
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [32])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/train_on_dreamer_data'])
    r.generate_commands(args.for_real)

def batch_size_lr_11_1_21():
    """
    does increasing the batch size help it train faster and segregate?
    does increasing the learning rate help it train faster and segregate?

    for balls, it makes a difference
    for manipulation it doesn't
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['balls_whiteball_push', 'dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [64])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [1e-4, 4e-4])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate'])
    r.generate_commands(args.for_real)


def segregrate_manipulation_11_1_21():
    """
    could a difference with the pytorch results also have to do with the normalization? 

    B32_flr0.0004_tp0.05_20211101215458 this worked.
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [32])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [1e-4, 4e-4])
    r.add_flag('fwm.model.temp', [5e-2, 5e-3])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate fwm.model.temp'])
    r.generate_commands(args.for_real)


def push_learning_rate_batch_size_11_2_21():
    """
        b16,lr1e-4,tp0.1: does not segregate
        b16,lr1e-4,tp0.05: does not segregate
        b16,1r1e-4,tp0.005: does not segregate
        b16,lr1e-4,tp0.5: does not segregate

        b16,lr4e-4,tp0.1: does not segregate
        b16,lr4e-4,tp0.05: does not segregate
        b16,lr4e-4,tp0.005: does not segregate
        b16,lr4e-4,tp0.5: does not segregate

        b16,lr8e-4,tp0.1: does not segregate
        b16,lr8e-4,tp0.05: does not segregate
        b16,lr8e-4,tp0.005: segregates, but pretty late in the game
        b16,lr8e-4,tp0.5: does not segregate
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [1e-4, 4e-4, 8e-4])
    r.add_flag('fwm.model.temp', [5e-2, 5e-3])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate fwm.model.temp'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [1e-4, 4e-4, 8e-4])
    r.add_flag('fwm.model.temp', [5e-1, 1e-1])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate fwm.model.temp'])
    r.generate_commands(args.for_real)


def does_prediction_horizon_affect_return_11_2_21():
    """
    
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 1, 0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('dataset.length', [50, 25, 10, 5])
    r.add_flag('logdir', ['runs/does_prediction_horizon_affect_return'])
    r.add_flag('watch', ['dataset.length'])
    r.generate_commands(args.for_real)


def push_learning_rate_batch_size_geb_11_3_21():
    """ dreamer/train_model

    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [16e-4, 32e-4])
    r.add_flag('fwm.model.temp', [5e-1, 5e-2, 5e-3])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate fwm.model.temp'])
    r.generate_commands(args.for_real)

def push_learning_rate_batch_size_gauss1_11_3_21():
    """ dreamer/train_model

    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0,1,2])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('wm_only', [True])
    r.add_flag('precision', [32])
    r.add_flag('dataset.batch', [32])
    r.add_flag('dataset.length', [3])
    r.add_flag('video_pred.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [16e-4, 32e-4])
    r.add_flag('fwm.model.temp', [5e-1, 5e-2, 5e-3])
    r.add_flag('fwm.sess.pred_horizon', [1])  # should be redundant
    r.add_flag('wm', ['fwm'])
    r.add_flag('logdir', ['runs/make_dreamer_train_faster'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate fwm.model.temp'])
    r.generate_commands(args.for_real)


def does_sequence_length_affect_rollout_quality_11_2_21():
    """
    
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0, 1, 1, 0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [50, 25, 10])
    r.add_flag('logdir', ['runs/does_sequence_length_affect_rollout_quality'])
    r.add_flag('watch', ['dataset.length eval_dataset.length eval_dataset.seed_steps'])
    r.generate_commands(args.for_real)

def how_necessary_are_discrete_latents_11_4_21():
    """
    
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.stoch', [1])
    r.add_flag('rssm.discrete', [1])
    r.add_flag('logdir', ['runs/how_necessary_are_discrete_latents'])
    r.add_flag('watch', ['rssm.stoch rssm.discrete'])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('logdir', ['runs/how_necessary_are_discrete_latents'])
    r.add_flag('watch', ['rssm.stoch rssm.discrete'])
    r.generate_commands(args.for_real)

def merge_train_and_train_model_11_4_21():
    """
    I just want to make sure that if I run with the same hyperparameters as 

    mbchang@gauss1:/home/mbchang/shared/counterfactual_dyna_umbrella/baselines/dreamerv2/runs/make_dreamer_train_faster/balls_whiteball_push/B64_flr0.0004_2021110118412, 

    I get the same leaning curve

    failed on gauss, will try again on geb
    """
    r = Runner(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [64])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [3])
    r.add_flag('eval_dataset.seed_steps', [2])
    r.add_flag('fwm.optim.learning_rate', [4e-4])
    r.add_flag('seed', [1, 2])
    r.add_flag('logdir', ['runs/merge_train_and_train_model'])
    r.add_flag('watch', ['dataset.batch fwm.optim.learning_rate dataset.length eval_dataset.length eval_dataset.seed_steps seed'])
    r.generate_commands(args.for_real)

def find_good_hyperparms_for_slim_train_model_11_4_21():
    """
        ok, these died. I encountered this error:


        Traceback (most recent call last):
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/train_model.py", line 295, in <module>
    main()
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/train_model.py", line 280, in main
    report = agnt.report(next(eval_dataset))
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/sandbox/dreamer_wrapper.py", line 161, in report
    rollout_output, rollout_metrics = self.model.rollout(batch=data, seed_steps=seed_steps, pred_horizon=self.config.eval_dataset.length-seed_steps)
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/sandbox/slot_attention_learners.py", line 392, in rollout
    imag_output = self.imagine(recon_output['posterior']['latent'][:, -1], act[:, seed_steps-1:])
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/sandbox/slot_attention_learners.py", line 380, in imagine
    imag_latent = self.generate(slots, actions)
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/dreamerv2/dreamerv2/sandbox/slot_attention_learners.py", line 329, in generate
    latents = rearrange(latents, 't b ... -> b t ...')
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/einops/einops.py", line 450, in rearrange
    raise TypeError("Rearrange can't be applied to an empty list")
TypeError: Rearrange can't be applied to an empty list
    """
    r = RunnerWithIDs(command='python dreamerv2/train_model.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['balls_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.model.temp', [0.05, 0.5])
    r.add_flag('fwm.optim.warmup_steps', [0, 10000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparms_for_slim_balls_train_model'])
    to_watch = [
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def merge_train_and_train_model_manipulation_11_4_21():
    """
    """
    r = Runner(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [32])


    # r.add_flag('dataset.length', [3])
    # r.add_flag('eval_dataset.length', [3])
    # r.add_flag('eval_dataset.seed_steps', [2])
    # r.add_flag('fwm.optim.learning_rate', [4e-4])
    # r.add_flag('seed', [1, 2])


    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.optim.warmup_steps', [10000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/merge_train_and_train_model_manipulation'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def merge_train_and_train_model_cheetah_default_11_5_21():
    """
    """
    # r = Runner(command='python dreamerv2/train.py', gpus=[0])
    # r.add_flag('configs', ['dmc_vision'])
    # r.add_flag('task', ['dmc_cheetah_run'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [20000])
    # r.add_flag('wm_only', [True])
    # r.add_flag('logdir', ['runs/merge_train_and_train_model_cheetah_default'])
    # to_watch = [
    #     'dataset.batch',
    #     'dataset.length',
    #     'eval_dataset.length',
    #     'eval_dataset.seed_steps',
    #     'wm_only',
    # ]
    # r.add_flag('watch', [' '.join(to_watch)])
    # r.generate_commands(args.for_real)

    r = Runner(command='python dreamerv2/train_model.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('logdir', ['runs/merge_train_and_train_model_cheetah_default'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def find_good_hyperparameters_train_mballs_11_5_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.model.temp', [0.05, 0.5])
    r.add_flag('fwm.optim.warmup_steps', [0, 10000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_mballs_train'])
    to_watch = [
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def find_good_hyperparameters_train_cradle_11_5_21():
    """
        conclusion: 5e-4 might be too low of a learning rate
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.model.temp', [0.05, 0.5])
    r.add_flag('fwm.optim.warmup_steps', [5000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_cradle_train'])
    to_watch = [
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparameters_train_cradle2_11_5_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_place_cradle'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-3])
    r.add_flag('fwm.model.temp', [0.05, 0.5])
    r.add_flag('fwm.optim.warmup_steps', [5000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_cradle_train'])
    to_watch = [
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparameters_train_mballs2_11_5_21():
    """
    Conclustion:
        This was best:
        T3_eT10_ss3_flr0.0005_fws5000_plTrue_tp0.5_20211105221011
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [1e-4, 5e-4, 1e-3])
    r.add_flag('fwm.model.temp', [0.5])
    r.add_flag('fwm.optim.warmup_steps', [5000, 15000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_mballs_train'])
    to_watch = [
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_mballs_train_jit_compatible_11_6_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16, 32])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [1e-4, 5e-4])
    r.add_flag('fwm.model.temp', [0.5])
    r.add_flag('fwm.optim.warmup_steps', [10000])
    r.add_flag('fwm.optim.decay_steps', [5000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_mballs_train_jit_compatible'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def find_good_hyperparams_for_mballs_train_jit_compatible2_11_6_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.model.temp', [0.5])
    r.add_flag('fwm.optim.warmup_steps', [10000])
    r.add_flag('fwm.optim.decay_steps', [0, 10000, 15000, 20000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_mballs_train_jit_compatible'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_finger_train_11_6_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.model.temp', [0.5])
    r.add_flag('fwm.optim.warmup_steps', [10000])
    r.add_flag('fwm.optim.decay_steps', [5000, 10000])
    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_finger_train'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def min_lr_balls_11_7_21():
    """
        at this point we have as default
            batch_size=16
            decay_steps=10000
            warmup_steps=10000
            decay_rate=0.5
            posterior_loss=True

        if minimum learning rate doesn't mess things up, then we will keep it, to allow the model to keep training when the actor and critic generate more data
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True, False])
    r.add_flag('fwm.optim.min_lr', [2e-4, 3e-4])

    r.add_flag('logdir', ['runs/min_lr_balls'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.temp',
        'fwm.optim.min_lr'
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_stacker_fish_11_7_21():
    """
        at this point we have as default
            batch_size=16
            decay_steps=10000
            warmup_steps=10000
            decay_rate=0.5
            posterior_loss=True

        if minimum learning rate doesn't mess things up, then we will keep it, to allow the model to keep training when the actor and critic generate more data
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    # r.add_flag('task', ['dmc_fish_swim', 'dmc_stacker_stack2'])
    # r.add_flag('task', ['dmc_fish_swim', 'dmc_swimmer_swimmer6', 'dmc_stacker_stack_2'])
    # r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.update_step.temp', [0.5, 0.1, 0.05, 0.01])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_stacker_fish'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr'
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_dmc_11_7_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_reach_site'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16, 32])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True])
    r.add_flag('fwm.optim.learning_rate', [5e-4])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.update_step.temp', [0.5, 0.1, 0.05, 0.01])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_dmc'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr'
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

# def find_good_hyperparams_for_dmc2_11_7_21():
#     """
#     """
#     r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
#     r.add_flag('configs', ['dmc_vision fwm'])
#     r.add_flag('task', ['dmc_manip_reach_site'])
#     r.add_flag('agent', ['causal'])
#     r.add_flag('prefill', [20000])
#     r.add_flag('dataset.batch', [16, 32])
#     r.add_flag('dataset.length', [3])
#     r.add_flag('eval_dataset.length', [10])
#     r.add_flag('eval_dataset.seed_steps', [3])

#     r.add_flag('fwm.model.posterior_loss', [True])
#     r.add_flag('fwm.optim.learning_rate', [5e-4])
#     r.add_flag('fwm.optim.min_lr', [1e-4])
#     r.add_flag('fwm.model.encoder_type', ['default'])
#     r.add_flag('fwm.model.update_step.temp', [0.5, 0.1, 0.05, 0.01])

#     r.add_flag('logdir', ['runs/find_good_hyperparams_for_dmc'])
#     to_watch = [
#         'dataset.batch',
#         'dataset.length',
#         'eval_dataset.length',
#         'eval_dataset.seed_steps',
#         'fwm.optim.learning_rate',
#         'fwm.optim.warmup_steps',
#         'fwm.optim.decay_steps',
#         'fwm.model.posterior_loss',
#         'fwm.model.update_step.temp',
#         'fwm.optim.min_lr',
#         'fwm.model.encoder_type',
#     ]
#     r.add_flag('watch', [' '.join(to_watch)])
#     r.generate_commands(args.for_real)

def find_good_hyperparams_for_dmc3_11_7_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_reach_site'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16, 32])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True])
    r.add_flag('fwm.optim.learning_rate', [1e-3])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.encoder_type', ['default', 'slim'])
    r.add_flag('fwm.model.update_step.temp', [0.5, 0.05])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_dmc'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def find_good_hyperparams_for_dmc4_11_7_21():
    """
        conclusion: 
            lr 5e-3 too high

    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_reach_site'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True])
    r.add_flag('fwm.optim.learning_rate', [5e-3, 1e-3, 5e-4])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.encoder_type', ['default', 'slim'])
    r.add_flag('fwm.model.update_step.temp', [0.5])
    r.add_flag('dataset.batch', [16, 32, 64])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_dmc'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def make_sure_reset_state_when_isfirstTrue_did_not_break_anything_balls_11_8_21():
    """
        these are the defaults
          batch_size=16,
          decay_rate=0.5,
          decay_steps=10000,
          learning_rate=5e-4,
          num_train_steps=500000,
          warmup_steps=10000,
          min_lr=1e-4,
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])
    r.add_flag('fwm.model.encoder_type', ['default', 'slim'])

    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/make_sure_reset_state_when_isfirstTrue_did_not_break_anything'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def make_sure_reset_state_when_isfirstTrue_did_not_break_anything_fixed_bug_11_8_21():
    """
        these are the defaults
          batch_size=16,
          decay_rate=0.5,
          decay_steps=10000,
          learning_rate=5e-4,
          num_train_steps=500000,
          warmup_steps=10000,
          min_lr=1e-4,

          the resetted_states replace the prior rather than the posterior
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])
    r.add_flag('fwm.model.encoder_type', ['default', 'slim'])

    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/make_sure_reset_state_when_isfirstTrue_did_not_break_anything_fixed_bug'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def can_it_model_the_balls_environment_that_have_reward_11_8_21():
    """
        these are the defaults
          batch_size=16,
          decay_rate=0.5,
          decay_steps=10000,
          learning_rate=5e-4,
          num_train_steps=500000,
          warmup_steps=10000,
          min_lr=1e-4,

          the resetted_states replace the prior rather than the posterior
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 0, 0, 0])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])
    r.add_flag('fwm.model.encoder_type', ['slim'])

    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/can_it_model_the_balls_environment_that_have_reward'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def make_sure_dreamer_can_solve_ball_environments_11_8_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['vmballs_simple_box4', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/make_sure_dreamer_can_solve_ball_environments'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def can_dreamer_solve_manip_reach_site_11_8_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_manip_reach_site'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/can_dreamer_solve_manip_reach_site'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def can_it_model_the_balls_environment_that_have_reward2_11_9_21():
    """
        these are the defaults
          batch_size=16,
          decay_rate=0.5,
          decay_steps=10000,
          learning_rate=5e-4,
          num_train_steps=500000,
          warmup_steps=10000,
          min_lr=1e-4,

          the resetted_states replace the prior rather than the posterior
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 3])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['vmballs_simple_box4', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [32])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])
    r.add_flag('fwm.model.encoder_type', ['slim'])

    r.add_flag('fwm.model.posterior_loss', [True, False])

    r.add_flag('logdir', ['runs/can_it_model_the_balls_environment_that_have_reward'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'fwm.optim.learning_rate',
        'fwm.optim.warmup_steps',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.optim.min_lr',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_dmc5_11_7_21():
    """
        model dim 128

        if we have space we can try scaling up more 
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_manip_reach_site'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True, False])
    r.add_flag('fwm.optim.learning_rate', [1e-3, 5e-4])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.encoder_type', ['default'])
    r.add_flag('fwm.model.update_step.temp', [0.5])
    r.add_flag('dataset.batch', [32])
    r.add_flag('fwm.optim.decay_steps', [10000, 25000])
    r.add_flag('fwm.model.dim', [128])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_dmc'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'fwm.optim.learning_rate',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.model.dim',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def find_good_hyperparams_for_finger_11_7_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision fwm'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.length', [3])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('fwm.model.posterior_loss', [True, False])
    r.add_flag('fwm.optim.learning_rate', [1e-3, 5e-4])
    r.add_flag('fwm.optim.min_lr', [1e-4])
    r.add_flag('fwm.model.encoder_type', ['default'])
    r.add_flag('fwm.model.update_step.temp', [0.5])
    r.add_flag('dataset.batch', [32])
    r.add_flag('fwm.optim.decay_steps', [10000, 25000])
    r.add_flag('fwm.model.dim', [96])

    r.add_flag('logdir', ['runs/find_good_hyperparams_for_finger'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'fwm.optim.learning_rate',
        'fwm.optim.decay_steps',
        'fwm.model.posterior_loss',
        'fwm.model.update_step.temp',
        'fwm.model.dim',
        'fwm.model.encoder_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def can_dreamer_solve_finger_turn_easy_11_8_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/can_dreamer_solve_finger_turn_easy'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def generate_data_dmc_manip_reach_site_11_17_21():
    """
    CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/data --configs debug --task dmc_manip_reach_site --agent causal --prefill 20000 --cpu=False --headless=True
    """
    pass

def generate_data_dmc_finger_turn_easy_11_17_21():
    """
    CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/data --configs debug --task dmc_finger_turn_easy --agent causal --prefill 20000 --cpu=False --headless=True
    """
    pass


def does_slate_wrapper_work_11_19_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision slate'])
    r.add_flag('task', ['dmc_stacker_stack_2', 'dmc_finger_turn_easy', 'dmc_manip_reach_site', 'vmballs_simple_box4', 'dmc_manip_lift_large_box', 'dmc_manip_place_brick'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('slate.slot_model.slot_attn.num_slots', [5])

    r.add_flag('logdir', ['runs/does_slate_wrapper_work'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'slate.slot_model.slot_attn.num_slots',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def does_slate_wrapper_work_with_tffunction_in_causal_agent_11_19_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2, 3])
    r.add_flag('configs', ['dmc_vision slate'])
    r.add_flag('task', ['dmc_stacker_stack_2', 'dmc_finger_turn_easy', 'dmc_manip_reach_site', 'vmballs_simple_box4', 'dmc_manip_lift_large_box', 'dmc_manip_place_brick'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('slate.slot_model.slot_attn.num_slots', [5])

    r.add_flag('logdir', ['runs/does_slate_wrapper_work_with_tffunction_in_causal_agent'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'slate.slot_model.slot_attn.num_slots',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def dynamic_slate_post_loss_only_11_21_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_stacker_stack_2', 'dmc_finger_turn_easy', 'dmc_manip_reach_site', 'vmballs_simple_box4', 'dmc_manip_lift_large_box', 'dmc_manip_place_brick'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])

    r.add_flag('logdir', ['runs/dynamic_slate_post_loss_only'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def dynamic_slate_prior_and_post_loss_11_21_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2, 3, 0, 1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_stacker_stack_2', 'dmc_finger_turn_easy', 'dmc_manip_reach_site', 'vmballs_simple_box4', 'dmc_manip_lift_large_box', 'dmc_manip_place_brick'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])

    r.add_flag('logdir', ['runs/dynamic_slate_prior_and_post_loss'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def does_raising_temp_enable_better_temporal_consistency_in_attn_11_23_21():
    """
        could it also be the batch size?
        learning rate and batch size go together
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])

    r.add_flag('logdir', ['runs/does_raising_temp_enable_better_temporal_consistency_in_attn'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def visrollout_and_consistency_loss_11_23_21():
    """
        does consistency loss help with temporal consistency?

    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2, 3])
    r.add_flag('configs', ['dmc_vision dslate'])
    # r.add_flag('task', ['dmc_finger_turn_easy', 'mballs_whiteball_push'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [10])
    r.add_flag('dataset.length', [5])
    r.add_flag('eval_dataset.length', [10])
    r.add_flag('eval_dataset.seed_steps', [3])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True, False])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0, 2.0])

    r.add_flag('logdir', ['runs/visrollout_and_consistency_loss'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def visrollout_and_consistency_loss_t2_11_24_21():
    """
        see how temperature and batch size affect segregation and consistency with only t=2

        to do this, if seed_steps is the same as the dataset.length, then you won't do imagination
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2, 3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16, 32])
    r.add_flag('dataset.length', [2])
    r.add_flag('eval_dataset.length', [2])
    r.add_flag('eval_dataset.seed_steps', [2])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])

    r.add_flag('logdir', ['runs/visrollout_and_consistency_loss_t2'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def check_that_autoregressive_works_for_static_11_24_21():
    """
        making the maxlen of replay dataset.length will hopefully ensure examples within a batch are not correlated

        the goal here is to check whether static still works
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[3, 3, 3, 0, 1, 2])
    r.add_flag('configs', ['dmc_vision slate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('replay.minlen', [2])
    r.add_flag('replay.maxlen', [2])
    r.add_flag('dataset.batch', [16, 32])
    r.add_flag('dataset.length', [2])
    r.add_flag('eval_dataset.length', [2])
    r.add_flag('eval_dataset.seed_steps', [2])

    r.add_flag('slate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('slate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])

    r.add_flag('logdir', ['runs/check_that_autoregressive_works_for_static'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'slate.slot_model.slot_attn.num_slots',
        'slate.slot_model.slot_attn.temp',
        'slate.slot_model.lr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def dslate_t2_replaymaxlen2_11_24_21():
    """
        see if having replaymaxlen2 will improve diversity within a batch
    """
    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2])
    # r.add_flag('configs', ['dmc_vision dslate'])
    # r.add_flag('task', ['mballs_whiteball_push'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [20000])
    # r.add_flag('replay.minlen', [2])
    # r.add_flag('replay.maxlen', [2])
    # r.add_flag('dataset.batch', [16])
    # r.add_flag('dataset.length', [2])
    # r.add_flag('eval_dataset.length', [2])
    # r.add_flag('eval_dataset.seed_steps', [2])

    # r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    # r.add_flag('dslate.slot_model.consistency_loss', [True])
    # r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])

    # r.add_flag('logdir', ['runs/dslate_t2_replaymaxlen2'])
    # to_watch = [
    #     'replay.minlen',
    #     'replay.maxlen',
    #     'dataset.batch',
    #     'dataset.length',
    #     'dslate.slot_model.slot_attn.num_slots',
    #     'dslate.slot_model.slot_attn.temp',
    #     'dslate.slot_model.lr',
    #     'dslate.slot_model.consistency_loss',
    # ]
    # r.add_flag('watch', [' '.join(to_watch)])
    # r.generate_commands(args.for_real)

    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2, 3, 3, 3])
    # r.add_flag('configs', ['dmc_vision dslate'])
    # r.add_flag('task', ['mballs_whiteball_push'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [20000])
    # r.add_flag('replay.minlen', [2])
    # r.add_flag('replay.maxlen', [2])
    # r.add_flag('dataset.batch', [16])
    # r.add_flag('dataset.length', [2])
    # r.add_flag('eval_dataset.length', [2])
    # r.add_flag('eval_dataset.seed_steps', [2])

    # r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    # r.add_flag('dslate.slot_model.consistency_loss', [True])
    # r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])
    # r.add_flag('dslate.slot_model.lr', [5e-5, 5e-4])

    # r.add_flag('logdir', ['runs/dslate_t2_replaymaxlen2'])
    # to_watch = [
    #     'replay.minlen',
    #     'replay.maxlen',
    #     'dataset.batch',
    #     'dataset.length',
    #     'dslate.slot_model.slot_attn.num_slots',
    #     'dslate.slot_model.slot_attn.temp',
    #     'dslate.slot_model.lr',
    #     'dslate.slot_model.consistency_loss',
    # ]
    # r.add_flag('watch', [' '.join(to_watch)])
    # r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('replay.minlen', [2])
    r.add_flag('replay.maxlen', [2])
    r.add_flag('dataset.batch', [16])
    r.add_flag('dataset.length', [2])
    r.add_flag('eval_dataset.length', [2])
    r.add_flag('eval_dataset.seed_steps', [2])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.1])
    r.add_flag('dslate.slot_model.lr', [5e-5, 1e-4, 5e-4])

    r.add_flag('logdir', ['runs/dslate_t2_replaymaxlen2'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)



def can_it_reconstruct_for_only_two_balls_11_24_21():
    """
        try doing this on geb instead
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('replay.minlen', [2])
    r.add_flag('replay.maxlen', [2])
    r.add_flag('dataset.length', [2])
    r.add_flag('eval_dataset.length', [2])
    r.add_flag('eval_dataset.seed_steps', [2])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0, 2.0])
    r.add_flag('dslate.slot_model.lr', [1e-4, 5e-4])

    r.add_flag('logdir', ['runs/can_it_reconstruct_for_only_two_balls'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)



def can_it_reconstruct_for_only_two_balls_grace_11_24_21():
    """
        try doing this
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box', 'dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('replay.minlen', [2])
    r.add_flag('replay.maxlen', [2])
    r.add_flag('dataset.length', [2])
    r.add_flag('eval_dataset.length', [2])
    r.add_flag('eval_dataset.seed_steps', [2])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.1, 0.5, 1.0, 2.0])
    r.add_flag('dslate.slot_model.lr', [5e-5, 1e-4, 5e-4])

    r.add_flag('logdir', ['runs/can_it_reconstruct_for_only_two_balls_grace'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)



def balls_stress_test_length_11_25_21():
    """
        let's now see if it still works if we increase the length
        grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3,4,5,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0])
    r.add_flag('dslate.slot_model.lr', [1e-4, 3e-4, 5e-4])

    r.add_flag('logdir', ['runs/balls_stress_test_length'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [3,4,5]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def do_the_hyperparameters_work_for_other_tasks_11_25_21():
    """
        gauss1
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'dmc_manip_reach_site', 'dmc_manip_place_cradle', 'dmc_stacker_stack_2'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0])
    r.add_flag('dslate.slot_model.lr', [5e-4])

    r.add_flag('logdir', ['runs/do_the_hyperparameters_work_for_other_tasks'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)



    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'dmc_manip_reach_site', 'dmc_manip_place_cradle', 'dmc_stacker_stack_2'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5])
    r.add_flag('dslate.slot_model.lr', [3e-4])

    r.add_flag('logdir', ['runs/do_the_hyperparameters_work_for_other_tasks'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def balls_stress_test_length_try2_11_25_21():
    """
        let's now see if it still works if we increase the length
        grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3,4,5,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [0.5, 1.0])
    r.add_flag('dslate.slot_model.lr', [1e-4, 3e-4, 5e-4, 7e-4])

    r.add_flag('logdir', ['runs/balls_stress_test_length_try2'])
    to_watch = [
        'replay.minlen',
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.consistency_loss',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [3]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)



def lr_decay_other_tasks_11_26_21():
    """
        gauss1
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'dmc_manip_reach_site', 'mballs_whiteball_push', 'dmc_stacker_stack_2'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [5e-4])
    r.add_flag('dslate.slot_model.decay_steps', [30000, 60000])

    r.add_flag('logdir', ['runs/lr_decay_other_tasks'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.decay_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def balls_stress_test_length_lr_decay_11_26_21():
    """
        do it on geb
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1, 3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [1e-4, 3e-4, 5e-4, 7e-4])
    r.add_flag('dslate.slot_model.decay_steps', [30000])

    r.add_flag('logdir', ['runs/balls_stress_test_length_lr_decay'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.decay_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def balls_stress_test_length_lr_decay_t8_11_27_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [2e-4, 3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1, 0.2])
    r.add_flag('dslate.slot_model.decay_steps', [30000])

    r.add_flag('logdir', ['runs/balls_stress_test_length_lr_decay_t8'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def balls_stress_test_length_lr_decay_t8_nomaskdyn_11_27_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    # r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.lr', [2e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1, 0.2])
    r.add_flag('dslate.slot_model.decay_steps', [30000])

    r.add_flag('logdir', ['runs/balls_stress_test_length_lr_decay_t8_nomaskdyn'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    for t in lengths:
        r.add_flag('replay.minlen', [t])
        r.add_flag('replay.maxlen', [t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)



def balls_test_imagination_11_27_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [2e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1, 0.2])
    r.add_flag('dslate.slot_model.decay_steps', [30000])

    r.add_flag('logdir', ['runs/balls_test_imagination'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    for t in lengths:
        r.add_flag('replay.minlen', [2*t])
        r.add_flag('replay.maxlen', [2*t])
        r.add_flag('dataset.length', [t])
        r.add_flag('eval_dataset.length', [2*t])
        r.add_flag('eval_dataset.seed_steps', [t])

        r.generate_commands(args.for_real)


def balls_curriculum_t8_11_28_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4, 2e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/balls_curriculum_t8'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
        'dslate.curr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    coeffs = [1, 2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def balls_curriculum_t8_currevery_11_28_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3,4,5,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4, 2e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('dslate.curr_every', [30000, 40000])

    r.add_flag('logdir', ['runs/balls_curriculum_t8'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
        'dslate.curr',
        'dslate.curr_every',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1, 2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def dmc_curriculum_t8_11_29_21():
    """
        tried to launch on grace but it crashed
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3,4,5,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'dmc_manip_reach_site', 'dmc_manip_place_cradle', 'dmc_stacker_stack_2'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4, 5e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/dmc_curriculum_t8'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.slot_model.decay_steps',
        'dslate.curr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def balls_test_is_first_11_29_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['mballs_whiteball_push'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.05, 0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('dslate.slot_model.hack_is_first', [True, False])
    r.add_flag('dslate.slot_model.handle_is_first', [True, False])
    # normal is when handle_is_first is False, doesn't matter what hack_is_first is

    r.add_flag('logdir', ['runs/balls_test_is_first'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        # 'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        # 'dslate.slot_model.decay_steps',
        'dslate.curr',
        'dslate.slot_model.hack_is_first',
        'dslate.slot_model.handle_is_first',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)



def test_reward_head_11_30_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,0, 1, 1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5, 1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_reward_head'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def test_randomly_initialized_policy_11_30_21():
    """
        this should perform the same as usual from the perspective of the world model
        The only changes are
            action_encode at each img_step
            the world model is used in the policy()
            tf_cosine_anneal for tau in encoder()
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_randomly_initialized_policy'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def test_with_policy_learning_12_1_21():
    """
        I basically have not touched any of the code in ActorCritic.
        So as long as my world model is good, the ActorCritic part should also be good.
        The only possibe reason it might not work is if I'm not propagating the hidden states across training sequences. But that should only be a problem for non-cyclical behaviors (which is ultimately what I want to work on though), but we will deal with that when we get to it

        put this on grace
    """
    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3,4,5,6,7])
    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,3,5,7])
    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2,6])
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,4])
    r.add_flag('configs', ['dmc_vision dslate'])
    # r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box4', 'vmballs_simple_box', 'dmc_manip_reach_site'])
    # r.add_flag('task', [ 'vmballs_simple_box4', 'dmc_manip_reach_site'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])#5, 1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_with_policy_learning'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def test_with_policy_learning_gauss1_12_1_21():
    """
        I basically have not touched any of the code in ActorCritic.
        So as long as my world model is good, the ActorCritic part should also be good.
        The only possibe reason it might not work is if I'm not propagating the hidden states across training sequences. But that should only be a problem for non-cyclical behaviors (which is ultimately what I want to work on though), but we will deal with that when we get to it
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5, 1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_with_policy_learning_gauss1'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def test_with_policy_learning_delay_learning_behavior_gauss1_12_1_21():
    """
        I basically have not touched any of the code in ActorCritic.
        So as long as my world model is good, the ActorCritic part should also be good.
        The only possibe reason it might not work is if I'm not propagating the hidden states across training sequences. But that should only be a problem for non-cyclical behaviors (which is ultimately what I want to work on though), but we will deal with that when we get to it
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [3])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_with_policy_learning_delay_learning_behavior_gauss1'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def debug_why_it_crashes_gauss1_12_1_21():
    """
        I basically have not touched any of the code in ActorCritic.
        So as long as my world model is good, the ActorCritic part should also be good.
        The only possibe reason it might not work is if I'm not propagating the hidden states across training sequences. But that should only be a problem for non-cyclical behaviors (which is ultimately what I want to work on though), but we will deal with that when we get to it
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [200])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/debug_why_it_crashes'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def test_with_policy_learning_delay_learning_policy_12_4_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,2])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1, 2])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])

    r.add_flag('logdir', ['runs/test_with_policy_learning_delay_learning_policy'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def test_whether_stop_gradient_matters_12_4_21():
    """
    CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train.py --configs dmc_vision dslate --task dmc_finger_turn_easy --agent causal --prefill 200 --dataset.batch 16 --wm_only False --dslate.slot_model.slot_attn.num_slots 1 --dslate.slot_model.consistency_loss=True --dslate.slot_model.slot_attn.temp 1.0 --dslate.slot_model.lr 0.0003 --dslate.slot_model.min_lr_factor 0.1 --dslate.slot_model.decay_steps 30000 --dslate.curr=True --logdir runs/debug_why_it_crashes_jit_components_remove_stopgrad --watch replay.maxlen dataset.batch dataset.length dslate.slot_model.slot_attn.num_slots dslate.slot_model.slot_attn.temp dslate.slot_model.lr dslate.slot_model.min_lr_factor dslate.curr wm_only --replay.minlen 8 --replay.maxlen 8 --dataset.length 4 --eval_dataset.length 8 --eval_dataset.seed_steps 4 --pretrain 2 --log_every 5 --train_steps 10

    conclusion: it does seem to matter
    """
    pass

def does_transformer_behavior_help_12_7_21():
    """
        on grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1, 2])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [True, False])

    r.add_flag('logdir', ['runs/does_transformer_behavior_help'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        'critic_stop_grad',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def does_training_immediately_imply_faster_behavior_efficiency_12_8_21():
    """
        on grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 4, 5])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [2])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [True, False])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('logdir', ['runs/does_training_immediately_imply_faster_behavior_efficiency'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def test_larger_receptive_field_12_8_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('dslate.dvae.weak', [False])

    r.add_flag('logdir', ['runs/test_larger_receptive_field'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def does_larger_receptive_field_help_k1_behavior_12_9_21():
    """
        on grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1,2])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0, 50000])

    r.add_flag('logdir', ['runs/does_larger_receptive_field_help_k1_behavior'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def monolithic_reference_for_finger_12_9_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_finger_turn_easy'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/monolithic_reference'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1, 4])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_finger_turn_easy', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/monolithic_reference'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [2]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def monolithic_reference2_12_9_21():
    """
        reach_site
        cheetah
        simple_box4
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2, 3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_manip_reach_site', 'dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/monolithic_reference'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[5,6,7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_manip_reach_site', 'vmballs_simple_box4', 'dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])

    r.add_flag('logdir', ['runs/monolithic_reference'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def how_does_k1_perform_on_cheetah_12_9_21():
    """
        on grace
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('logdir', ['runs/how_does_k1_perform_on_cheetah'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def does_probabilistic_reward_head_improve_behavior_12_11_21():
    """
        on grace.

        Yes, yes it does.
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1, 3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('logdir', ['runs/does_probabilistic_reward_head_improve_behavior'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def is_slotdim128_sufficient_12_12_21():
    """
        is slotdim128 sufficient compared to slotdim192?

        how much memory does slotdim128 cost and how much memory does slotdim192 cost?

        slotdim192 costs 4706MiB per job

        the process id is not 5119 and is not 5163

        it looks like 
        slotdim128 costs 2557 per job --> wait this is not entirely accurate, because we are using curriculum
        (although if we want to be comparable to monolithic dreamer we should probably use hdim=200).

        But for now we just want to test whether training for sequence lengths of 50 steps will help us
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('logdir', ['runs/is_slotdim128_sufficient'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        # 'dslate.curr',
        # 'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def train_on_longer_sequence_length_no_curr_for_cheetah_12_12_21():
    """
        slotdim128, t4 costs:
        slotdim128, t8, nocurr costs: 8472MiB

        no curriculum here

        it is not process id 2541, 2557
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [False])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('logdir', ['runs/train_on_longer_sequence_length_no_curr_for_cheetah'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        # 'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def train_on_longer_sequence_length_no_curr_for_cheetah2_12_12_21():
    """
        slotdim128, t4 costs:
        slotdim128, t8, nocurr costs: 8472MiB
        slotdim128, t16, nocurr costs:
        slotdim128, t32, nocurr costs:
        slotdim128, t50, nocurr costs:

        no curriculum here

        it is not process id 2541, 2557

        gpu4: 16 
        gpu5: 32
        gpu7: 50

        it gets OOM for 16, 32, and 50
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4, 5, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [False])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('logdir', ['runs/train_on_longer_sequence_length_no_curr_for_cheetah'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        # 'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [16, 32, 50]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def train_on_longer_sequence_length_no_curr_for_cheetah3_12_12_21():
    """
        slotdim128, t4 costs:
        slotdim128, t8, nocurr costs: 8472MiB

        no curriculum here

        it is not process id 2541, 2557
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4,5])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [False])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('logdir', ['runs/train_on_longer_sequence_length_no_curr_for_cheetah'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        # 'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def train_on_longer_sequence_length_no_curr_for_cheetah4_12_12_21():
    """
        slotdim128, t4 costs:
        slotdim128, t8, nocurr costs: 8472MiB

        with curriculum
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['False'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [1])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('logdir', ['runs/train_on_longer_sequence_length_no_curr_for_cheetah'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        'dslate.slot_model.lr',
        'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        # 'wm_only',
        'critic_stop_grad',
        'delay_train_behavior_by',
        'dslate.dvae.weak',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [8]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def discrete_hard_test_stop_gradient_12_12_21():
    """
        what are the differences with what I had in the past?
            - mono_train
            - einsum_dict
            - slot size 128
            - stronger encoder
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1, 3, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('logdir', ['runs/discrete_hard_test_stop_gradient'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        # 'dslate.slot_model.lr',
        # 'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        # 'critic_stop_grad',
        # 'delay_train_behavior_by',
        'dslate.dvae.weak',


        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def discrete_soft_test_stop_gradient_12_13_21():
    """
        what are the differences with what I had in the past?
            - mono_train
            - einsum_dict
            - slot size 128
            - stronger encoder
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1, 3, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('dslate.dvae.sm_hard', [False])

    r.add_flag('logdir', ['runs/discrete_soft_test_stop_gradient'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        # 'dslate.slot_model.slot_attn.temp',
        # 'dslate.slot_model.lr',
        # 'dslate.slot_model.min_lr_factor',
        'dslate.curr',
        'wm_only',
        # 'critic_stop_grad',
        # 'delay_train_behavior_by',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',


        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def make_sure_nothing_has_broken_12_14_21():
    """
        it seems like the worse cross entropy has to do with the stronger encoder. Maybe we should just swap it with the encoder from SAVI?

    """

    # dim = 128
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.slot_model.einsum_dict', [True, False])

    r.add_flag('logdir', ['runs/make_sure_nothing_has_broken'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.d_model',

        'dslate.slot_model.einsum_dict',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


    # dim = 192
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,3,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [192])
    r.add_flag('dslate.slot_model.slot_size', [192])

    r.add_flag('dslate.slot_model.einsum_dict', [True, False])

    r.add_flag('logdir', ['runs/make_sure_nothing_has_broken'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.d_model',

        'dslate.slot_model.einsum_dict',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def how_necessary_are_discrete_latents_balls_12_14_21():
    """
    """
    # r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2, 3])
    # r.add_flag('configs', ['dmc_vision'])
    # r.add_flag('task', ['vmballs_simple_box4', 'vmballs_simple_box'])
    # r.add_flag('agent', ['causal'])
    # r.add_flag('rssm.stoch', [1])
    # r.add_flag('rssm.discrete', [1])

    # r.add_flag('logdir', ['runs/how_necessary_are_discrete_latents_balls'])
    # to_watch = [
    #     'dataset.batch',
    #     'dataset.length',
    #     'eval_dataset.length',
    #     'eval_dataset.seed_steps',
    #     'rssm.stoch',
    #     'rssm.discrete',
    # ]
    # r.add_flag('watch', [' '.join(to_watch)])
    # r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['vmballs_simple_box4', 'vmballs_simple_box'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.stoch', [1])
    r.add_flag('rssm.discrete', [1])

    r.add_flag('logdir', ['runs/how_necessary_are_discrete_latents_balls'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.stoch',
        'rssm.discrete',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [4]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def discrete_hard_test_stop_gradient_try_again_12_14_21():
    """
        what are the differences with what I had in the past?
            - mono_train
            - einsum_dict
            - slot size 128
            - stronger encoder

        what crashed:
            - rmxl4_B16_T4_k5_crT_wmoT_dvwkT_mtT_esdT_sgiF_sgoF_20211214213648
            - rmxl4_B16_T4_k5_crT_wmoT_dvwkF_mtT_esdT_sgiT_sgoF_20211214213642
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 1, 2, 3, 3, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('logdir', ['runs/discrete_hard_test_stop_gradient_try_again'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def discrete_hard_test_stop_gradient_try_again2_12_15_21():
    """
        what did not crash previously:
            dvwkTrue, sgiTrue, sgoFalse
            dvwkFalse, sgiFalse, sgoTrue

        not second 3
        not second 1
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 1, 2, 3, 3, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('logdir', ['runs/discrete_hard_test_stop_gradient_try_again2'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def discrete_soft_test_stop_gradient_try_again_12_15_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 1, 2, 3, 3, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('dslate.dvae.sm_hard', [False])

    r.add_flag('logdir', ['runs/discrete_soft_test_stop_gradient_try_again'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def test_e2e_with_full_loss_12_16_21():
    """
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False, True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True, False])
    r.add_flag('dslate.stop_gradient_output', [True, False])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.e2e', [True])

    r.add_flag('logdir', ['runs/test_e2e_with_full_loss'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.e2e',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def sanity_check_with_dummy_stoch_12_16_21():
    """
        even though dslate.slot_model.distributional=True, the stoch aren't doing anything for now
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [True])

    r.add_flag('logdir', ['runs/sanity_check_with_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [False])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [True])

    r.add_flag('logdir', ['runs/sanity_check_with_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)



def sanity_check_without_dummy_stoch_12_17_21():
    """
        make sure things did not break.

        These were good:
            sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiT_sgoT_latdistF_20211217100312
            sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiF_sgoT_latdistF_20211217100314
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False])

    r.add_flag('logdir', ['runs/sanity_check_without_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [False])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [False])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False])

    r.add_flag('logdir', ['runs/sanity_check_without_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def sanity_check_without_dummy_stoch_grace_12_17_21():
    """
        make sure things did not break.

        These were good:
            sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiT_sgoT_latdistF_20211217100312
            sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiF_sgoT_latdistF_20211217100314
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False, True])

    r.add_flag('logdir', ['runs/sanity_check_without_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [True])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [False])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False, True])

    r.add_flag('logdir', ['runs/sanity_check_without_dummy_stoch'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)



def k1_with_cross_update_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB

        the memory cost is the same as that for monolithic dreamer.
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4,5])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross'])

    r.add_flag('logdir', ['runs/k1_with_cross_update'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def monolithic_check_memory_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[6,7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    # r.add_flag('rssm.update', ['cross'])

    r.add_flag('logdir', ['runs/monolithic_check_memory'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_with_cross_dynamics_and_update_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB as well

        the memory cost is the same as that for monolithic dreamer too, for all variations
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4,5,6,7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross', 'default'])
    r.add_flag('rssm.dynamics', ['cross'])

    r.add_flag('logdir', ['runs/k1_with_cross_dynamics_and_update'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_with_cross_dynamics_and_update_no_action_bias_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB as well

        the memory cost is the same as that for monolithic dreamer too, for all variations
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1,2,3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross', 'default'])
    r.add_flag('rssm.dynamics', ['cross'])

    r.add_flag('logdir', ['runs/k1_with_cross_dynamics_and_update_no_action_bias'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_learnable_fixed_init_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB as well

        the memory cost is the same as that for monolithic dreamer too, for all variations
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4,5,6,7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross', 'default'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['fixed'])

    r.add_flag('logdir', ['runs/k1_learnable_fixed_init'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def k1_slotattn_update_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 8773MiB as well

        the memory cost is the same as that for monolithic dreamer too, for all variations
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['slot'])
    r.add_flag('rssm.dynamics', ['cross', 'default'])
    r.add_flag('rssm.initial', ['iid', 'fixed'])

    r.add_flag('logdir', ['runs/k1_slotattn_update'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_selfattn_behavior_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 10919MiB, probably from the 4x MLP in the transformer?
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2, 3])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['slot'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['iid'])
    r.add_flag('behavior_type', ['selfattn'])

    r.add_flag('logdir', ['runs/k1_selfattn_behavior'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[6, 7])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['fixed'])
    r.add_flag('behavior_type', ['selfattn'])

    r.add_flag('logdir', ['runs/k1_selfattn_behavior'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_crossattn_behavior_12_17_21():
    """
        cheetah
        simple_box4

        memory cost: 10919MiB, probably from the 4x MLP in the transformer?
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['slot'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['iid'])
    r.add_flag('behavior_type', ['crossattn'])

    r.add_flag('logdir', ['runs/k1_crossattn_behavior'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4, 5])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run', 'vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['cross'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['fixed'])
    r.add_flag('behavior_type', ['crossattn'])

    r.add_flag('logdir', ['runs/k1_crossattn_behavior'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_how_much_reduction_do_we_get_with_rssmhidden_instead_of_head_units_12_17_21():
    """
        cheetah

        memory cost: 8886MiB

        ok, this is much better in terms of memory. It would be the same amount of memory usage as monolithic. 
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['slot'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['iid'])
    r.add_flag('behavior_type', ['crossattn', 'selfattn'])

    r.add_flag('logdir', ['runs/k1_how_much_reduction_do_we_get_with_rssmhidden_instead_of_head_units'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)

def k1_test_data_parallel():
    """
        CUDA_VISIBLE_DEVICES=2,3 DISPLAY=:0 python dreamerv2/train.py --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.update slot --rssm.dynamics cross --rssm.initial iid --behavior_type selfattn --logdir runs/k1_test_dataparallel --watch dataset.batch dataset.length eval_dataset.length eval_dataset.seed_steps rssm.update rssm.dynamics rssm.initial behavior_type --data_parallel True
    """
    pass

def k1_test_data_parallel2_12_17_21():
    """
        cheetah

        memory cost: 8886MiB

        ok, this is much better in terms of memory. It would be the same amount of memory usage as monolithic. 
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[4,5])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('rssm.update', ['slot'])
    r.add_flag('rssm.dynamics', ['cross'])
    r.add_flag('rssm.initial', ['iid'])
    r.add_flag('behavior_type', ['selfattn'])
    r.add_flag('dataset.batch', [32])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_test_data_parallel'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_test_data_parallel3_12_17_21():
    """
        cheetah

        memory cost: 8886MiB

        this is for original monolithic


        CUDA_VISIBLE_DEVICES=0,1 DISPLAY=:0 python dreamerv2/train.py --configs dmc_vision --task vmballs_simple_box4 --agent causal --data_parallel=True --logdir runs/k1_test_data_parallel --watch dataset.batch dataset.length eval_dataset.length eval_dataset.seed_steps rssm.update rssm.dynamics rssm.initial behavior_type data_parallel &
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_test_data_parallel'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update',
        'rssm.dynamics',
        'rssm.initial',
        'behavior_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


"""
Here is a configuation of dynamic slate that works best right now.
- 
"""

def can_dslate_train_on_50_steps_12_20_21():
    """
        sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiT_sgoT_latdistF_20211217100312

        What does not fit:
            B=2, T=50

        What fits:
            B=2, T=25: 4601MiB
            B=2, T=40: 8697MiB
            B=2, T=45: 8701MiB

        Trying:
            B=2, T=45:

        wait actually it seems like 
            B=2, T=50 does work 8701
            B=4, T=50 8825
            B=8, T=50, 8701

        no, that didn't work because I had curr=True


        See how much we can push it with batch size of 2.

        When you are waiting for the experiments to finish running, you should be writing and coding.

        NOTE that I got rid of the prefill!

        I should do monotrain=False but somehow it stalls if I do monotrain=False

        ok I can fit B=4, T=50, with mono_train=False

        It turns out that B=4, T=50 failed
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2,3,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [4])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [2e-4, 3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [False, True])
    r.add_flag('dslate.curr_every', [2000])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [False])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False])

    r.add_flag('data_parallel', [False])

    r.add_flag('logdir', ['runs/can_dslate_train_on_50_steps'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'dslate.curr_every',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',

        'dslate.mono_train',
        'dslate.slot_model.einsum_dict',
        'dslate.stop_gradient_input',
        'dslate.stop_gradient_output',
        'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [50]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def can_dslate_train_on_50_steps_12_21_21():
    """
        sanity_check_without_dummy_stoch/t_vmballs_simple_box4/rmxl2_B16_T2_k5_crT_wmoT_dvwkT_dvsmhF_mtT_esdT_sgiT_sgoT_latdistF_20211217100312

        What does not fit:
            B=2, T=50

        What fits:
            B=2, T=25: 4601MiB
            B=2, T=40: 8697MiB
            B=2, T=45: 8701MiB

        Trying:
            B=2, T=45:

        wait actually it seems like 
            B=2, T=50 does work 8701
            B=4, T=50 8825
            B=8, T=50, 8701

        no, that didn't work because I had curr=True


        See how much we can push it with batch size of 2.

        When you are waiting for the experiments to finish running, you should be writing and coding.

        NOTE that I got rid of the prefill!

        I should do monotrain=False but somehow it stalls if I do monotrain=False

        ok I can fit B=4, T=50, with mono_train=False

        It turns out that B=4, T=50 failed

        Ok, so during pretraining, B=2, T=50 works and costs 8725MiB only. So I wonder where the OOM would come from (if it does come) later. 
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[2,3,6,7])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    # r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [2])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [2e-4, 3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [False])
    r.add_flag('dslate.curr_every', [2000])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [False])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False])

    r.add_flag('data_parallel', [False])

    r.add_flag('logdir', ['runs/can_dslate_train_on_50_steps'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'dslate.curr_every',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.lr',

        # 'dslate.mono_train',
        # 'dslate.slot_model.einsum_dict',
        # 'dslate.stop_gradient_input',
        # 'dslate.stop_gradient_output',
        # 'dslate.slot_model.distributional',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [50]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)


def k1_does_dvae_encoder_break_anything_12_21_21():
    """
        8709MiB

    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0, 1, 4, 5])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['grid_dvweak', 'grid_dvstrong'])
    r.add_flag('pos_encode_type', ['slate', 'none'])

    r.add_flag('logdir', ['runs/k1_does_dvae_encoder_break_anything'])
    to_watch = [
        'dataset.batch',
        'dataset.length',
        'eval_dataset.length',
        'eval_dataset.seed_steps',
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_does_dvae_decoder_break_anything_12_21_21():
    """
        n6

        requires two GPUs
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=['0,1', '2,3'])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['default'])
    r.add_flag('decoder_type', ['grid_dvweak', 'grid_dvstrong'])
    r.add_flag('pos_encode_type', ['slate'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_does_dvae_decoder_break_anything'])
    to_watch = [
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
        'decoder_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_does_dvae_decoder_break_anything_geb_12_21_21():
    """
        n6

        requires two GPUs

        two_blocks_eight_heads_defaults

        this fits, but just barely.
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=['0,1'])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['grid_dvweak'])
    r.add_flag('decoder_type', ['grid_dvweak'])
    r.add_flag('pos_encode_type', ['slate'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_does_dvae_decoder_break_anything_2b8h'])
    to_watch = [
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
        'decoder_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)



def shallower_transformer_12_21_21():
    """
        use shallower transformer for decoder

        number of heads does not increase parameter count
        n2 has 6/7 the number of parameters as nb4

        TODO: record the memory usage:
            nb2, nh8 (id 11027): 2557MiB (2939205 parameters)
            nb4, nh8 (id 11079): 4605MiB (3466309 parameters)
            nb2, nh4 (id 11053): 2557MiB (2939205 parameters)
            nb4, nh4 (id 11103): 2557MiB (3466309 parameters)

        but it seems like having a shallower transformer actually doesn't decrease the memory usage that much. 
        Next we should try simplifying the encoder and decoder and seeing if they hurt
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,5])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('dslate.curr_every', [20000])
    r.add_flag('critic_stop_grad', [False])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])
    r.add_flag('dslate.slot_model.d_model', [128])
    r.add_flag('dslate.slot_model.slot_size', [128])

    r.add_flag('dslate.mono_train', [False])
    r.add_flag('dslate.slot_model.einsum_dict', [True])
    r.add_flag('dslate.stop_gradient_input', [True])
    r.add_flag('dslate.stop_gradient_output', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.slot_model.distributional', [False])

    r.add_flag('data_parallel', [False])

    r.add_flag('dslate.slot_model.obs_transformer.num_blocks', [2, 4])
    r.add_flag('dslate.slot_model.obs_transformer.num_heads', [8, 4])

    r.add_flag('logdir', ['runs/shallower_transformer'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'dslate.curr_every',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.lr',

        'dslate.slot_model.obs_transformer.num_blocks',
        'dslate.slot_model.obs_transformer.num_heads',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            r.generate_commands(args.for_real)

def shallower_encoder_decoder_12_21_21():
    """
        I'm observing that shallower encoder and decoder don't actually affect much. But what does affect memory capacity is the hidden dimension.

        dim 128: 2557 (2378309 parameters)
        dim 96: 2557 (1505701 parameters)
        dim 64: 1533 (992904 parameters)
        dim 32: 1533 (442469 parameters)
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,2,3,5])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.slot_model.decay_steps', [30000])
    r.add_flag('dslate.curr', [True])
    r.add_flag('dslate.curr_every', [20000])
    r.add_flag('dslate.dvae.weak', [True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('dslate.slot_model.einsum_dict', [True])

    r.add_flag('dslate.dvae.sm_hard', [False])

    r.add_flag('dslate.dvae.shallow', [True])
    r.add_flag('dslate.slot_model.obs_transformer.num_blocks', [2, 4])
    r.add_flag('dslate.slot_model.dyn_transformer.num_blocks', [2, 4])

    r.add_flag('logdir', ['runs/shallower_encoder_decoder'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'dslate.curr_every',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.lr',
        'dslate.slot_model.d_model',

        'dslate.dvae.shallow',
        'dslate.slot_model.obs_transformer.num_blocks',
        'dslate.slot_model.dyn_transformer.num_blocks',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    dims = [128, 96, 64, 32]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])

            for dim in dims:
                r.add_flag('dslate.slot_model.d_model', [dim])
                r.add_flag('dslate.slot_model.slot_size', [dim])

                r.generate_commands(args.for_real)


def k1_does_dvae_decoder_break_anything_grace_12_22_21():
    """
        3844276 model parameters
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=['6,7'])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['grid_dvstrong'])
    r.add_flag('decoder_type', ['grid_dvstrong'])
    r.add_flag('pos_encode_type', ['slate'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_does_dvae_decoder_break_anything_2b8h'])
    to_watch = [
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
        'decoder_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def does_changing_warmup_peaktime_change_bump_12_23_21():
    """
        these all take 2557 MiB
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.curr', [True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('dslate.dvae.sm_hard', [False])

    r.add_flag('dslate.dvae.shallow', [True])
    r.add_flag('dslate.slot_model.obs_transformer.num_blocks', [4])
    r.add_flag('dslate.slot_model.dyn_transformer.num_blocks', [2])

    r.add_flag('dslate.slot_model.d_model', [64])
    r.add_flag('dslate.slot_model.slot_size', [64])

    r.add_flag('dslate.dvae.tau_steps', [15000, 30000, 45000])
    r.add_flag('dslate.slot_model.lr_warmup_steps', [15000, 30000, 45000])

    r.add_flag('logdir', ['runs/does_changing_warmup_peaktime_change_bump'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.weak',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.d_model',

        'dslate.dvae.shallow',
        'dslate.slot_model.obs_transformer.num_blocks',
        'dslate.slot_model.dyn_transformer.num_blocks',

        'dslate.dvae.tau_steps',
        'dslate.slot_model.lr_warmup_steps',

    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])
            r.generate_commands(args.for_real)


def does_slate_work_with_generic_encoder_decoder_12_23_21():
    """
        ids: 142261, 142272, 142379
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[1,2,3])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [2e-4, 3e-4, 4e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.curr', [True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.dvae.cnn_type', ['generic'])
    r.add_flag('dslate.slot_model.obs_transformer.num_blocks', [4])
    r.add_flag('dslate.slot_model.dyn_transformer.num_blocks', [2])

    r.add_flag('dslate.slot_model.d_model', [64])
    r.add_flag('dslate.slot_model.slot_size', [64])


    r.add_flag('logdir', ['runs/does_slate_work_with_generic_encoder_decoder'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.cnn_type',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.d_model',

        'dslate.slot_model.obs_transformer.num_blocks',
        'dslate.slot_model.dyn_transformer.num_blocks',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])
            r.generate_commands(args.for_real)

def k1_generic_encoder_decoder_coordconv_break_anything_12_23_21():
    """
        slate: 10921MiB (3531860 model parameters)
        coordconv: 10921MiB (3531860 model parameters)

        Hmm, wait actually in the original dreamer it actually had 13 million model parameters, whereas here you only have 3 million model parameters

        But why does this still take up so much memory though?
        Where is this memory usage coming from, if not from the model parameters?
        Is it mostly from the obs_transformer?

        I could still fit it with the encoder. So it must be due to the decoder.
        This version uses the two blocks eight heads
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=['4,5', '6,7'])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['grid_generic'])
    r.add_flag('decoder_type', ['grid_generic'])
    r.add_flag('pos_encode_type', ['slate', 'coordconv'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_generic_encoder_decoder_coordconv_break_anything'])
    to_watch = [
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
        'decoder_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def k1_generic_encoder_decoder_coordconv_break_anything_n6_12_23_21():
    """
        slate: 10921MiB (3563988 model parameters)
        coordconv: 10921MiB (3531860 model parameters)
        
        coordconv has slightly less model parameters

        Hmm yeah interesting; I can fit it in memory if I just use four heads instead of eight heads. Same number of parameters
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0,1])
    r.add_flag('configs', ['dmc_vision slot'])
    r.add_flag('task', ['dmc_cheetah_run'])
    r.add_flag('agent', ['causal'])
    r.add_flag('encoder_type', ['grid_generic'])
    r.add_flag('decoder_type', ['grid_generic'])
    r.add_flag('pos_encode_type', ['slate', 'coordconv'])
    r.add_flag('data_parallel', [True])

    r.add_flag('logdir', ['runs/k1_generic_encoder_decoder_coordconv_break_anything_2b4h'])
    to_watch = [
        'rssm.update_type',
        'rssm.dynamics_type',
        'rssm.initial_type',
        'behavior_type',
        'encoder_type',
        'pos_encode_type',
        'decoder_type',
        'data_parallel', 
    ]
    r.add_flag('watch', [' '.join(to_watch)])
    r.generate_commands(args.for_real)


def how_does_number_of_slotattn_iterations_affect_slate_12_23_21():
    """
        ids: 142261, 142272, 142379
    """
    r = RunnerWithIDs(command='python dreamerv2/train.py', gpus=[0])
    r.add_flag('configs', ['dmc_vision dslate'])
    r.add_flag('task', ['vmballs_simple_box4'])
    r.add_flag('agent', ['causal'])
    r.add_flag('prefill', [20000])
    r.add_flag('dataset.batch', [16])

    r.add_flag('wm_only', ['True'])

    r.add_flag('dslate.slot_model.slot_attn.num_slots', [5])
    r.add_flag('dslate.slot_model.consistency_loss', [True])
    r.add_flag('dslate.slot_model.slot_attn.temp', [1.0])
    r.add_flag('dslate.slot_model.lr', [3e-4])
    r.add_flag('dslate.slot_model.min_lr_factor', [0.1])
    r.add_flag('dslate.curr', [True])
    r.add_flag('delay_train_behavior_by', [0])

    r.add_flag('dslate.dvae.sm_hard', [False])
    r.add_flag('dslate.dvae.cnn_type', ['generic'])
    r.add_flag('dslate.slot_model.obs_transformer.num_blocks', [4])
    r.add_flag('dslate.slot_model.dyn_transformer.num_blocks', [2])

    r.add_flag('dslate.slot_model.d_model', [64])
    r.add_flag('dslate.slot_model.slot_size', [64])
    r.add_flag('dslate.slot_model.slot_attn.num_iterations', [1,2,4])


    r.add_flag('logdir', ['runs/how_does_number_of_slotattn_iterations_affect_slate'])
    to_watch = [
        'replay.maxlen',
        'dataset.batch',
        'dataset.length',
        'dslate.slot_model.slot_attn.num_slots',
        'dslate.curr',
        'wm_only',
        'dslate.dvae.cnn_type',
        'dslate.dvae.sm_hard',
        'dslate.slot_model.d_model',

        'dslate.slot_model.obs_transformer.num_blocks',
        'dslate.slot_model.dyn_transformer.num_blocks',

        'dslate.slot_model.slot_attn.num_iterations',
    ]
    r.add_flag('watch', [' '.join(to_watch)])

    lengths = [2]
    coeffs = [1]
    for t in lengths:
        for coeff in coeffs:
            r.add_flag('replay.minlen', [coeff*t])
            r.add_flag('replay.maxlen', [coeff*t])
            r.add_flag('dataset.length', [t])
            r.add_flag('eval_dataset.length', [coeff*t])
            r.add_flag('eval_dataset.seed_steps', [t])
            r.generate_commands(args.for_real)



if __name__ == '__main__':
    # perceiver_test_10_6_2021()
    # train_model_sanity()
    # train_model_single_step_sanity_10_22_21()
    # train_model_two_step_sanity_10_22_21()
    # train_model_balls_sanity_10_23_21()
    # train_model_balls_fwm_10_27_21()
    # comparison_for_train_on_dreamer_data_11_1_21()
    # batch_size_lr_11_1_21()
    # segregrate_manipulation_11_1_21()
    # push_learning_rate_batch_size_11_2_21()
    # does_prediction_horizon_affect_return_11_2_21()
    # push_learning_rate_batch_size_geb_11_3_21()
    # push_learning_rate_batch_size_gauss1_11_3_21()
    # does_sequence_length_affect_rollout_quality_11_2_21()
    # how_necessary_are_discrete_latents_11_4_21()
    # merge_train_and_train_model_11_4_21()
    # find_good_hyperparms_for_slim_train_model_11_4_21()
    # merge_train_and_train_model_manipulation_11_4_21()
    # merge_train_and_train_model_cheetah_default_11_5_21()
    # find_good_hyperparameters_train_mballs_11_5_21()
    # find_good_hyperparameters_train_cradle_11_5_21()
    # find_good_hyperparameters_train_cradle2_11_5_21()
    # find_good_hyperparameters_train_mballs2_11_5_21()
    # find_good_hyperparams_for_mballs_train_jit_compatible_11_6_21()
    # find_good_hyperparams_for_mballs_train_jit_compatible2_11_6_21()
    # find_good_hyperparams_for_finger_train_11_6_21()
    # min_lr_balls_11_7_21()
    # find_good_hyperparams_for_stacker_fish_11_7_21()
    # find_good_hyperparams_for_dmc_11_7_21()
    # find_good_hyperparams_for_dmc2_11_7_21()
    # find_good_hyperparams_for_dmc3_11_7_21()
    # find_good_hyperparams_for_dmc4_11_7_21()
    # make_sure_reset_state_when_isfirstTrue_did_not_break_anything_balls_11_8_21()
    # make_sure_reset_state_when_isfirstTrue_did_not_break_anything_fixed_bug_11_8_21()
    # can_it_model_the_balls_environment_that_have_reward_11_8_21()
    # make_sure_dreamer_can_solve_ball_environments_11_8_21()
    # can_dreamer_solve_manip_reach_site_11_8_21()
    # can_it_model_the_balls_environment_that_have_reward2_11_9_21()
    # find_good_hyperparams_for_dmc5_11_7_21()
    # find_good_hyperparams_for_finger_11_7_21()
    # can_dreamer_solve_finger_turn_easy_11_8_21()
    # does_slate_wrapper_work_11_19_21()
    # does_slate_wrapper_work_with_tffunction_in_causal_agent_11_19_21()
    # dynamic_slate_post_loss_only_11_21_21()
    # dynamic_slate_prior_and_post_loss_11_21_21()
    # does_raising_temp_enable_better_temporal_consistency_in_attn_11_23_21()
    # visrollout_and_consistency_loss_11_23_21()
    # visrollout_and_consistency_loss_t2_11_24_21()
    # check_that_autoregressive_works_for_static_11_24_21()
    # dslate_t2_replaymaxlen2_11_24_21()
    # can_it_reconstruct_for_only_two_balls_11_24_21()
    # can_it_reconstruct_for_only_two_balls_grace_11_24_21()
    # balls_stress_test_length_11_25_21()
    # do_the_hyperparameters_work_for_other_tasks_11_25_21()
    # balls_stress_test_length_try2_11_25_21()
    # lr_decay_other_tasks_11_26_21()
    # balls_stress_test_length_lr_decay_11_26_21()
    # balls_stress_test_length_lr_decay_t8_11_27_21()
    # balls_stress_test_length_lr_decay_t8_nomaskdyn_11_27_21()
    # balls_test_imagination_11_27_21()
    # balls_curriculum_t8_11_28_21()
    # balls_curriculum_t8_currevery_11_28_21()
    # dmc_curriculum_t8_11_29_21()
    # balls_test_is_first_11_29_21()
    # test_reward_head_11_30_21()
    # test_randomly_initialized_policy_11_30_21()
    # test_with_policy_learning_12_1_21()
    # test_with_policy_learning_gauss1_12_1_21()
    # test_with_policy_learning_delay_learning_behavior_gauss1_12_1_21()
    # debug_why_it_crashes_gauss1_12_1_21()
    # test_with_policy_learning_delay_learning_policy_12_4_21()
    # does_transformer_behavior_help_12_7_21()
    # does_training_immediately_imply_faster_behavior_efficiency_12_8_21()
    # test_larger_receptive_field_12_8_21()
    # does_larger_receptive_field_help_k1_behavior_12_9_21()
    # monolithic_reference_for_finger_12_9_21()
    # monolithic_reference2_12_9_21()
    # how_does_k1_perform_on_cheetah_12_9_21()
    # does_probabilistic_reward_head_improve_behavior_12_11_21()
    # is_slotdim128_sufficient_12_12_21()
    # train_on_longer_sequence_length_no_curr_for_cheetah_12_12_21()
    # train_on_longer_sequence_length_no_curr_for_cheetah2_12_12_21()
    # train_on_longer_sequence_length_no_curr_for_cheetah3_12_12_21()
    # train_on_longer_sequence_length_no_curr_for_cheetah4_12_12_21()
    # discrete_hard_test_stop_gradient_12_12_21()
    # discrete_soft_test_stop_gradient_12_13_21()
    # make_sure_nothing_has_broken_12_14_21()
    # how_necessary_are_discrete_latents_balls_12_14_21()
    # discrete_hard_test_stop_gradient_try_again_12_14_21()
    # discrete_hard_test_stop_gradient_try_again2_12_15_21()
    # discrete_soft_test_stop_gradient_try_again_12_15_21()
    # test_e2e_with_full_loss_12_16_21()
    # sanity_check_with_dummy_stoch_12_16_21()
    # sanity_check_without_dummy_stoch_12_17_21()
    # sanity_check_without_dummy_stoch_grace_12_17_21()
    # k1_with_cross_update_12_17_21()
    # monolithic_check_memory_12_17_21()
    # k1_with_cross_dynamics_and_update_no_action_bias_12_17_21()
    # k1_learnable_fixed_init_12_17_21()
    # k1_slotattn_update_12_17_21()
    # k1_selfattn_behavior_12_17_21()
    # k1_crossattn_behavior_12_17_21()
    # k1_how_much_reduction_do_we_get_with_rssmhidden_instead_of_head_units_12_17_21()
    # k1_test_data_parallel2_12_17_21()
    # k1_test_data_parallel3_12_17_21()
    # can_dslate_train_on_50_steps_12_20_21()
    # can_dslate_train_on_50_steps_12_21_21()
    # k1_does_dvae_encoder_break_anything_12_21_21()
    # k1_does_dvae_decoder_break_anything_12_21_21()
    # k1_does_dvae_decoder_break_anything_geb_12_21_21()
    # shallower_transformer_12_21_21()
    # shallower_encoder_decoder_12_21_21()
    # k1_does_dvae_decoder_break_anything_grace_12_22_21()
    # does_changing_warmup_peaktime_change_bump_12_23_21()
    # does_slate_work_with_generic_encoder_decoder_12_23_21()
    # k1_generic_encoder_decoder_coordconv_break_anything_12_23_21()
    # k1_generic_encoder_decoder_coordconv_break_anything_n6_12_23_21()
    how_does_number_of_slotattn_iterations_affect_slate_12_23_21()

# CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/data --configs debug --task dmc_manip_reach_site --agent causal --prefill 20000 --cpu=False --headless=True

"""
debug
12-13-21
python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125 --wm_only=False --delay_train_behavior_by 3 --slot_behavior.use_slot_heads True
"""


