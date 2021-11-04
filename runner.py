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
    r.add_flag('dataset.length', [50, 25, 10, 5])
    r.add_flag('logdir', ['runs/does_sequence_length_affect_rollout_quality'])
    r.add_flag('watch', ['dataset.length eval_dataset.length eval_dataset.seed_steps'])
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
    does_sequence_length_affect_rollout_quality_11_2_21()





