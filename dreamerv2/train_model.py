import collections
import datetime
import functools
import logging
from loguru import logger as lgr
import os
import pathlib
import re
import sys
import warnings

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import common

# def create_expname(args):
#     import sandbox.logging_utils as lu
#     abbrvs = {
#         'task': '',
#         'precision': 'p',
#         'fwm.optim.learning_rate': 'flr'
#         'wm_only': 'wmo'
#     }
#     watcher = lu.watch(args.watch, abbrvs)
#     expname = pathlib.Path(args.task) / f'{watcher(args)}_{datetime.datetime.now():%Y%m%d%H%M%S}'
#     return expname

def parse_args():
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)
  return config

def parse_args_with_fwm():
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)

  if 'fwm' in parsed.configs:
    sys.path.append(os.path.join(str(pathlib.Path(__file__).parent), 'sandbox'))
    from sandbox.slot_attention_learners import FactorizedWorldModel
    configs['defaults']['fwm'] = FactorizedWorldModel.get_default_args().to_dict()

  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    if name != 'fwm':
      config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)
  return config

def main():
  config = parse_args_with_fwm()

  import sandbox.logging_utils as lu
  config = config.update({'logdir': pathlib.Path(config.logdir) / lu.create_expname(config)})

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)

  # initialize lgr
  lgr.remove()   # remove default handler
  lgr.add(os.path.join(logdir, 'debug.log'))
  if not config.headless:
    lgr.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

  config.save(logdir / 'config.yaml')
  lgr.info(f'{config}\n')
  lgr.info(f'Logdir: {logdir}')

  import tensorflow as tf

  np.random.seed(config.seed)
  tf.random.set_seed(config.seed)

  tf.config.experimental_run_functions_eagerly(not config.jit)
  if not config.cpu:
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', seed=config.seed, **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', seed=config.seed, **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until)

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.seed, config.dmc_camera, config.headless)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      raise NotImplementedError('did you set the seed?')
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      raise NotImplementedError('did you set the seed?')
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    elif suite == 'balls':
      from sandbox import debugging_envs
      env = debugging_envs.Balls(
        name=task, 
        action_repeat=config.action_repeat, 
        size=config.render_size, 
        seed=config.seed, 
        headless=config.headless)
      env = common.NormalizeAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    lgr.info(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  lgr.info('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    lgr.info(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  lgr.info('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  #############################################################
  # maybe use the mirrored strategy here? 
  if config.agent == 'dv2':
    import agent
    agnt = agent.Agent(config, obs_space, act_space, step)
  elif config.agent == 'causal':
    from sandbox import causal_agent
    if config.wm == 'default':
      agnt = causal_agent.WorldModel(config, obs_space, tf.Variable(int(step), tf.int64))
    elif config.wm == 'fwm':
      from sandbox import dreamer_wrapper
      agnt = dreamer_wrapper.FactorizedWorldModelWrapperForDreamer(config, obs_space, tf.Variable(int(step), tf.int64))
    else:
      raise NotImplementedError

  else:
    raise NotImplementedError
  #############################################################
  # train_agent = common.CarryOverState(agnt.train)
  train_agent = common.CarryOverStateMultipleOutputs(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    lgr.info('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  # train_policy = lambda *args: agnt.policy(
  #     *args, mode='explore' if should_expl(step) else 'train')
  # eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        # mets = train_agent(next(train_dataset))
        outputs, mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  lgr.info('Training...')
  random_agent = common.RandomAgent(act_space)
  while step < config.steps:

    outputs, mets = train_agent(next(train_dataset))
    [metrics[key].append(value) for key, value in mets.items()]

    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      lgr.info(f'Generating train report (step {step.value})...')
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)

    step.increment()

    if step.value % config.eval_every == 0:
      logger.write()
      lgr.info('Start evaluation.')
      logger.add(agnt.report(next(eval_dataset)), prefix='eval')
      eval_driver(random_agent, episodes=config.eval_eps)
      agnt.save(logdir / 'variables.pkl')

  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()


"""
What are the current differences from the pytorch version?
- I do not learn the initial slot
- I feed the stoch into the dynamics transformer rather than the deterministic latents
- My deter state was from the posterior rather than the prior -- it could be that the reason why dreamer learns to predict so well is just because it's trained ot predict from the predicted deter rather than the posterior


python dreamerv2/train_model.py --logdir runs/debug/model/default_cheetah --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5

python dreamerv2/train_model.py --logdir runs/debug/model/slot2 --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5 --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 2


10/25/21
python dreamerv2/train_model.py --logdir runs/debug/model/slot2 --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5 --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 2 --wm fwm


10/30/21
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_with_video --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm &

debug
python dreamerv2/train_model.py --logdir runs/debug/model/slot2 --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5 --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 2 --wm fwm --precision 32

debug default
python dreamerv2/train_model.py --logdir runs/debug/model/default_cheetah --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5


Gauss1: just trying which DM control environment will give segregation

CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/walker_walk_b32_t3_ph1 --configs dmc_vision fwm --task dmc_walker_walk --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm &

CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/hopper_hop_b32_t3_ph1 --configs dmc_vision fwm --task dmc_hopper_hop --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm &

CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/bic_catch_b32_t3_ph1 --configs dmc_vision fwm --task dmc_cup_catch --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm &



10/31/21

Question: is the slot temperature the main thing that is preventing dreamer version from learning as efficiently? 




geb:

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st6e-1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 0.6 &

CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st7e-1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 0.7 &

gauss1:

CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st8e-1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 0.8 &

CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st9e-1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 0.9 &

CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 1.0 &

gauss1 again (because previously had the wrong code)

CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b64_t3_ph1_st5e-1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 64 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 0.5 &

CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b64_t3_ph1_st1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 1.0 &

CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b32_t3_ph1_st1 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.model.temp 1.0 &

geb again
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm/dw_fwm_b64_t3_ph1_st5e-1_lr2e-4 --configs dmc_vision fwm --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 64 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm --fwm.optim.learning_rate 2e-4 &


"""