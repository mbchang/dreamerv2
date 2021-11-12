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
                # command += f' --{flag_name}={flag_value}'
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


def does_jit_really_make_the_difference_11_11_21():
    """
        seems like jit hurts the slot model's performance. Let's see if this is actually the case.

        conclusion: yes, it is only jit
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0, 1])
    r.add_flag('jit', [True, False])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/does_jit_really_make_the_difference'])
    r.generate_commands(args.for_real)

def what_if_we_just_jit_the_dvae_11_11_21():
    """
        works
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0, 1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_just_jit_the_dvae'])
    r.generate_commands(args.for_real)

def what_if_we_jit_dvae_and_slotattn_not_transformer_11_11_21():
    """
        works
    """
    r = RunnerWithIDs(command='python train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_dvae_and_slotattn_not_transformer'])
    r.generate_commands(args.for_real)

def what_if_we_jit_dvae_and_transformer_not_slot_attn_11_11_21():
    """
        works
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_dvae_and_transformer_not_slot_attn'])
    r.generate_commands(args.for_real)

def what_if_we_jit_dvae_slotattn_transformer_not_slotmodelcall_11_11_21():
    """
        expect this should work
        I think this works
        works
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_dvae_slotattn_transformer_not_slotmodelcall'])
    r.generate_commands(args.for_real)

def what_if_we_jit_entire_slotmodelcall_11_11_21():
    """
        this might fail
        this seems to work
    """
    r = RunnerWithIDs(command='python train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_entire_slotmodelcall'])
    r.generate_commands(args.for_real)

def what_if_we_jit_entire_slatecall_11_11_21():
    """
        this seems to work
    """
    r = RunnerWithIDs(command='python train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_entire_slatecall'])
    r.generate_commands(args.for_real)


if __name__ == '__main__':
    # does_jit_really_make_the_difference_11_11_21()
    # what_if_we_just_jit_the_dvae_11_11_21()
    # what_if_we_jit_dvae_and_slotattn_not_transformer_11_11_21()
    # what_if_we_jit_dvae_and_transformer_not_slot_attn_11_11_21()
    # what_if_we_jit_dvae_slotattn_transformer_not_slotmodelcall_11_11_21()
    # what_if_we_jit_entire_slotmodelcall_11_11_21()
    what_if_we_jit_entire_slatecall_11_11_21()


