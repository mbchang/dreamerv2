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

def what_if_we_jit_forward_and_backward_separately_11_11_21():
    """
        if I do this, then I get an error:

    ValueError: in user code:

        /home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/slate/tf_slate/train.py:273 backward  *  
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        /home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow_addons/optimizers/discriminative_layer_training.py:119 apply_gradients  *
            [
        /home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py:622 apply_gradients  **
            grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        /home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/keras/optimizer_v2/utils.py:72 filter_empty_gradients
            raise ValueError("No gradients provided for any variable: %s." %


    """
    r = RunnerWithIDs(command='python train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_forward_and_backward_separately'])
    r.generate_commands(args.for_real)

def check_again_that_jitting_train_step_does_not_work_11_11_21():
    """
        it does not work
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/check_again_that_jitting_train_step_does_not_work'])
    r.generate_commands(args.for_real)

def what_if_I_wrap_train_step_at_runtime_instead_of_decorating_11_11_21():
    """
        motivation from https://stackoverflow.com/questions/65988276/fail-to-train-a-model-using-tf-function

        then I get this error

        WARNING:tensorflow:5 out of the last 5 calls to <function train_step at 0x7fbf4c253a60> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
        WARNING:tensorflow:6 out of the last 6 calls to <function train_step at 0x7fbf4c253a60> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

        But it still seems to learn correctly though
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_I_wrap_train_step_at_runtime_instead_of_decorating'])
    r.generate_commands(args.for_real)

def what_if_we_jit_entire_slatecall_and_autoregressive_11_11_21():
    """
        this seems to work
    """
    r = RunnerWithIDs(command='python train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/what_if_we_jit_entire_slatecall_and_autoregressive_11_11_21'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_and_jit_each_11_11_21():
    """
        if this works, then we can work with this for dreamer

        --> it doesn't seem to work
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_and_jit_each'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_and_no_jit_11_11_21():
    """
        interesting - if I separate these two train steps, even without jit I only use 4GB of memory?
        wheresa if I put the entire train step together, without jit I use 8 GB of memory.
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_and_no_jit'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_jit_dvae_trainstep_only_11_11_21():
    """
        works
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_jit_dvae_trainstep_only'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_jit_slotmodel_trainstep_only_11_11_21():
    """
        does not work
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_jit_slotmodel_trainstep_only'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_jit_dvae_train_step_slot_model_forward_11_11_21():
    """
        works
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_jit_dvae_train_step_slot_model_forward'])
    r.generate_commands(args.for_real)

def split_into_separate_train_steps_jit_slotmodel_train_step_dvae_forward_11_11_21():
    """
        does not work
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/split_into_separate_train_steps_jit_slotmodel_train_step_dvae_forward'])
    r.generate_commands(args.for_real)

def jit_dvae_fb_slotmodel_f_numpy_inputs_into_slotmodel_11_12_21():
    """
        this should work
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/jit_dvae_fb_slotmodel_f_numpy_inputs_into_slotmodel_11_12_21'])
    r.generate_commands(args.for_real)


def jit_both_numpy_inputs_into_slotmodel_11_12_21():
    """
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/jit_both_numpy_inputs_into_slotmodel'])
    r.generate_commands(args.for_real)


def jit_dvae_do_not_jit_apply_gradient_for_slot_model_11_12_21():
    """

    @tf.function
    def slot_model_train_step(slot_model, main_optimizer, z_transformer_input, z_transformer_target):
        with tf.GradientTape() as tape:
            attns, cross_entropy = slot_model(z_transformer_input, z_transformer_target)
        gradients = tape.gradient(cross_entropy, slot_model.trainable_weights)
        return attns, cross_entropy, gradients

    It turns out that this works

    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/jit_dvae_do_not_jit_apply_gradient_for_slot_model'])
    r.generate_commands(args.for_real)

def do_not_jit_apply_gradients_at_all_11_12_21():
    """
    this should work -- works
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/do_not_jit_apply_gradients_at_all'])
    r.generate_commands(args.for_real)

def do_not_jit_apply_gradients_at_all_for_combined_11_12_21():
    """
    this should work --> works
    but this takes up twice as much memory as do_not_jit_apply_gradients_at_all_11_12_21 for some reason
    """
    r = RunnerWithIDs(command='python train.py', gpus=[1])
    r.add_flag('jit', [True])
    r.add_flag('headless', [True])
    r.add_flag('log_path', ['logs/do_not_jit_apply_gradients_at_all_for_combined'])
    r.generate_commands(args.for_real)

"""
Traceback (most recent call last):
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/slate/tf_slate/modular_train.py", line 196, in <module>
    model(train_loader.get_batch(), tau=tf.constant(1.0), hard=args.hard)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/keras/engine/base_layer.py", line 1037, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/mbchang/Documents/research/counterfactual_dyna_umbrella/baselines/slate/tf_slate/slate.py", line 161, in call
    attns, cross_entropy = self.slot_model(z_transformer_input, z_transformer_target)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/keras/engine/base_layer.py", line 1037, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 885, in __call__
    result = self._call(*args, **kwds)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 933, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 759, in _initialize
    self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 3066, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 3463, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 3298, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/framework/func_graph.py", line 1007, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 668, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 3990, in bound_method_wrapper
    return wrapped_fn(*args, **kwargs)
  File "/home/mbchang/.anaconda2/envs/dv2/lib/python3.9/site-packages/tensorflow/python/framework/func_graph.py", line 994, in wrapper
    raise e.ag_error_metadata.to_exception(e)
TypeError: in user code:


    TypeError: tf__autoregressive_decode() missing 1 required positional argument: 'gen_len'
"""

def try_on_balls_data_normalized_11_15_21():
    """
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('args.data_path', ['../ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('args.jit', [True])
    r.add_flag('args.headless', [True])
    r.add_flag('args.slot_attn.num_slots', [5])
    r.add_flag('args.log_path', ['logs/try_on_balls_data_normalized_11_15_21'])
    r.generate_commands(args.for_real)

def try_on_balls_data_unnormalized_11_15_21():
    """
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[0])
    r.add_flag('args.data_path', ['../ball_data/whiteballpush/U-Dk4s0n2000t10_ab'])
    r.add_flag('args.jit', [True])
    r.add_flag('args.headless', [True])
    r.add_flag('args.slot_attn.num_slots', [5])
    r.add_flag('args.log_path', ['logs/try_on_balls_data_unnormalized_11_15_21'])
    r.generate_commands(args.for_real)

def try_on_dmc_manip_normalized_11_15_21():
    """
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('args.data_path', ['../dmc_data/data/dmc_manip_place_cradle/wmoFalse_20211101152629'])
    r.add_flag('args.jit', [True])
    r.add_flag('args.headless', [True])
    r.add_flag('args.slot_attn.num_slots', [5])
    r.add_flag('args.log_path', ['logs/try_on_dmc_manip_normalized_11_15_21'])
    r.generate_commands(args.for_real)


def try_on_dmc_manip_bigmodel_11_15_21():
    """
        vocab size 4096
        decoder layers 8
        decoder heads 8
        slot attention iterations 7

        obs_transformer
    """
    r = RunnerWithIDs(command='python modular_train.py', gpus=[1])
    r.add_flag('args.data_path', ['../dmc_data/data/dmc_manip_place_cradle/wmoFalse_20211101152629'])
    r.add_flag('args.jit', [True])
    r.add_flag('args.headless', [True])
    r.add_flag('args.vocab_size', [4096])
    r.add_flag('args.obs_transformer.num_blocks', [8])
    r.add_flag('args.obs_transformer.num_heads', [8])
    r.add_flag('args.slot_attn.num_slots', [5])
    r.add_flag('args.slot_attn.num_iterations', [7])
    r.add_flag('args.log_path', ['logs/try_on_dmc_manip_bigmodel_11_15_21'])
    r.generate_commands(args.for_real)


if __name__ == '__main__':
    # does_jit_really_make_the_difference_11_11_21()
    # what_if_we_just_jit_the_dvae_11_11_21()
    # what_if_we_jit_dvae_and_slotattn_not_transformer_11_11_21()
    # what_if_we_jit_dvae_and_transformer_not_slot_attn_11_11_21()
    # what_if_we_jit_dvae_slotattn_transformer_not_slotmodelcall_11_11_21()
    # what_if_we_jit_entire_slotmodelcall_11_11_21()
    # what_if_we_jit_entire_slatecall_11_11_21()
    # what_if_we_jit_forward_and_backward_separately_11_11_21()
    # check_again_that_jitting_train_step_does_not_work_11_11_21()
    # what_if_I_wrap_train_step_at_runtime_instead_of_decorating_11_11_21()
    # what_if_we_jit_entire_slatecall_and_autoregressive_11_11_21()
    # split_into_separate_train_steps_and_jit_each_11_11_21()
    # split_into_separate_train_steps_and_no_jit_11_11_21()
    # split_into_separate_train_steps_jit_dvae_trainstep_only_11_11_21()
    # split_into_separate_train_steps_jit_slotmodel_trainstep_only_11_11_21()
    # split_into_separate_train_steps_jit_dvae_train_step_slot_model_forward_11_11_21()
    # split_into_separate_train_steps_jit_slotmodel_train_step_dvae_forward_11_11_21()
    # jit_dvae_fb_slotmodel_f_numpy_inputs_into_slotmodel_11_12_21()
    # jit_both_numpy_inputs_into_slotmodel_11_12_21()
    # jit_dvae_do_not_jit_apply_gradient_for_slot_model_11_12_21()
    # do_not_jit_apply_gradients_at_all_11_12_21()
    # do_not_jit_apply_gradients_at_all_for_combined_11_12_21()
    # try_on_balls_data_11_15_21()
    # try_on_balls_data_normalized_11_15_21()
    # try_on_balls_data_unnormalized_11_15_21()
    # try_on_dmc_manip_normalized_11_15_21()
    try_on_dmc_manip_bigmodel_11_15_21()
