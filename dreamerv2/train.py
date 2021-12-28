import collections
import datetime
import functools
import logging
from loguru import logger as lgr
import os
import pathlib
import psutil
import re
import shutil
import sys
import time
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

def parse_args():
  """ original """
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  configs = add_programmatically_generated_configs(parsed, configs)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)
  return config

def add_programmatically_generated_configs(parsed, configs):
  if 'fwm' in parsed.configs:
    sys.path.append(os.path.join(str(pathlib.Path(__file__).parent), 'sandbox'))
    from sandbox.slot_attention_learners import FactorizedWorldModel
    configs['defaults']['fwm'] = FactorizedWorldModel.get_default_args().to_dict()
  elif 'slate' in parsed.configs:
    sys.path.append(str(pathlib.Path(__file__).parent / 'sandbox' / 'tf_slate'))
    from slate import SLATE
    if 'debug' in parsed.configs:
      configs['defaults']['slate'] = SLATE.defaults_debug().to_dict()
    else:
      configs['defaults']['slate'] = SLATE.defaults().to_dict()
  elif 'dslate' in parsed.configs:
    sys.path.append(str(pathlib.Path(__file__).parent / 'sandbox' / 'tf_slate'))
    from slate import DynamicSLATE
    from sandbox.slot_behavior import SlotActorCritic
    if 'debug' in parsed.configs:
      configs['defaults']['dslate'] = DynamicSLATE.defaults_debug().to_dict()
      configs['defaults']['slot_behavior'] = SlotActorCritic.defaults_debug().to_dict()
    else:
      configs['defaults']['dslate'] = DynamicSLATE.defaults().to_dict()
      configs['defaults']['slot_behavior'] = SlotActorCritic.defaults().to_dict()
  elif 'slot' in parsed.configs:
    from sandbox import slot_machine, causal_agent
    configs['defaults']['slot'] = dict(
      rssm=slot_machine.SlotEnsembleRSSM.defaults().to_dict(),
      encoder=slot_machine.GridEncoder.defaults().to_dict(),
      decoder=slot_machine.GridDecoder.defaults().to_dict(),
      behavior=causal_agent.ActorCritic.defaults().to_dict(),  # or maybe I should subclass?
      )
  configs['defaults']['expdir'] = f'{datetime.datetime.now():%Y%m%d%H%M%S}'
  return configs


def save_source_code(exproot):
  code_dir = pathlib.Path(os.path.join(exproot, 'code'))
  os.makedirs(code_dir, exist_ok=True)
  root = pathlib.Path('.')

  shutil.copy2(root / 'runner.py', code_dir)

  tail = 'dreamerv2'
  subroot = root / tail
  os.makedirs(code_dir / tail, exist_ok=True)
  for src_file in [x for x in os.listdir(subroot) if '.py' in x or '.yaml' in x]:
    shutil.copy2(subroot / src_file, code_dir / tail)

  tail = 'dreamerv2/common'
  shutil.copytree(root / tail, code_dir / tail)

  tail = 'dreamerv2/sandbox'
  os.makedirs(code_dir / tail, exist_ok=True)
  subroot = root / tail
  for src_file in [x for x in os.listdir(subroot) if '.py' in x]:
    shutil.copy2(subroot / src_file, code_dir / tail)

  tail = 'dreamerv2/sandbox/tf_slate'
  os.makedirs(code_dir / tail, exist_ok=True)
  subroot = root / tail
  for src_file in [x for x in os.listdir(subroot) if '.py' in x]:
    shutil.copy2(subroot / src_file, code_dir / tail)
  

def main():
  config = parse_args()

  import sandbox.logging_utils as lu
  config = config.update({'expdir': f't_{lu.create_expname(config)}'})

  logdir = (pathlib.Path(config.logdir) / pathlib.Path(config.expdir)).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)

  # initialize lgr
  lgr.remove()   # remove default handler
  lgr.add(os.path.join(logdir, 'debug.log'))
  if not config.headless:
    lgr.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

  # initialize wandb
  import wandb
  suffix = '_db' if config.debug else '' 
  wandb.init(
      config=config,  # will need to change this
      project='slot attention',
      dir=logdir,
      group=pathlib.Path(config.logdir).name,
      job_type='train',
      id=f'dv2_train_{config.task}_{pathlib.Path(config.expdir).name}'
      )

  config.save(logdir / 'config.yaml')
  lgr.info(f'{config}\n')
  lgr.info(f'Logdir: {logdir}')

  save_source_code(logdir)

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
      # raise NotImplementedError('did you set the seed?')
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
    elif suite == 'mballs':
      from sandbox import debugging_envs
      env = debugging_envs.MutedBalls(
        name=task, 
        action_repeat=config.action_repeat, 
        size=config.render_size, 
        seed=config.seed, 
        headless=config.headless)
      env = common.NormalizeAction(env)
    elif suite == 'vmballs':
      from sandbox import debugging_envs
      env = debugging_envs.VeryMutedBalls(
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
    wandb.log({
      f'{mode}_return': score,
      f'{mode}_length': length}, step=step.value)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
        wandb.log({f'sum_{mode}_{key}': ep[key].sum()}, step=step.value)
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
        wandb.log({f'mean_{mode}_{key}': ep[key].mean()}, step=step.value)
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        wandb.log({f'max_{mode}_{key}': ep[key].max(0).mean()}, step=step.value)
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    wandb.log({f'{mode}_{key}': value for key, value in replay.stats.items()}, step=step.value)
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

  if config.data_parallel:
    strategy = tf.distribute.MirroredStrategy()  # DISTRIBUTED
  else:
    strategy = tf.distribute.get_strategy()  # DISTRIBUTED

  # NOTE: you would create the distributed dataset before you call iter
  lgr.info('Create agent.')
  train_dataset = iter(strategy.experimental_distribute_dataset(train_replay.dataset(**config.dataset)))  # DISTRIBUTED
  report_dataset = iter(strategy.experimental_distribute_dataset(train_replay.dataset(
    batch=config.eval_dataset.batch,
    length=config.eval_dataset.length)))  # DISTRIBUTED
  eval_dataset = iter(strategy.experimental_distribute_dataset(train_replay.dataset(
    batch=config.eval_dataset.batch,
    length=config.eval_dataset.length)))  # DISTRIBUTED
  #############################################################
  # maybe use the mirrored strategy here? 
  with strategy.scope():  # DISTRIBUTED
    if config.agent == 'dv2':
      import agent
      agnt = agent.Agent(config, obs_space, act_space, step)
    elif config.agent == 'causal':
      from sandbox import causal_agent
      agnt = causal_agent.CausalAgent(config, obs_space, act_space, step)
    else:
      raise NotImplementedError
  #############################################################
  train_agent = common.CarryOverState(agnt.train)
  # train_agent(next(train_dataset))
  strategy.run(train_agent, args=(next(train_dataset),))  # DISTRIBUTED
  if config.load_from and (pathlib.Path(config.load_from) / 'variables.pkl').exists():
    agnt.load(pathlib.Path(config.load_from) / 'variables.pkl')
    lgr.info(f'Loaded pretrained agent from: {config.load_from}')
  else:
    lgr.info('Pretrain agent.')
    for _ in range(config.pretrain):
      # train_agent(next(train_dataset))
      strategy.run(train_agent, args=(next(train_dataset),))  # DISTRIBUTED
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        t0 = time.time()
        # mets = train_agent(next(train_dataset))
        mets = strategy.run(train_agent, args=(next(train_dataset),))  # DISTRIBUTED
        mets = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, mets[key], axis=None) for key in mets}  # DISTRIBUTED
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      lgr.debug(f'Env step {step.value}: a train step took {time.time()-t0} seconds.')
      lgr.debug(f'RAM memory {psutil.virtual_memory()[2]}% used. Swap: {psutil.swap_memory()} used.')
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        #############################################################
        wandb.log({name: np.array(values, np.float64).mean()}, step=step.value)
        #############################################################
        metrics[name].clear()
      #############################################################
      if config.wm == 'dslate':
        agnt.wm.log_weights(step)
      #############################################################
      # report = agnt.report(next(report_dataset))

      report = strategy.run(agnt.report, args=(next(report_dataset),))  # DISTRIBUTED
      report = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, report[key], axis=None) for key in report}  # DISTRIBUTED
      wandb.log({key: np.array(report[key], np.float64).item() for key in report if 'openl' not in key}, step=step.value)
      logger.add(report, prefix='train')
      
      logger.write(fps=True)
      wandb.log({'fps': logger._compute_fps()}, step=step.value)
  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    lgr.info('Start evaluation.')
    # report = agnt.report(next(eval_dataset))
    ########################################
    report = strategy.run(agnt.report, args=(next(eval_dataset),))
    report = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, report[key], axis=None) for key in report}
    ########################################
    wandb.log({key: np.array(report[key], np.float64).item() for key in report if 'openl' not in key}, step=step.value)
    logger.add(report, prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    lgr.info('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
    agnt.wm.save(logdir / 'wm_variables.pkl')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()

"""
python dreamerv2/train.py --logdir runs/data --configs debug --task dmc_manip_reach_site --agent causal
"""