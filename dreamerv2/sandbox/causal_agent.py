from einops import rearrange
from loguru import logger as lgr
import ml_collections
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl

from sandbox import machine
from sandbox import normalize as nmlz
import sandbox.tf_slate.utils as slate_utils
from sandbox.tf_slate import slot_model as sm


def encoder_interface(embed, config):
    ###########################################################
    # encoder to latent?
    # import ipdb;ipdb.set_trace(context=20)
    if 'grid' in config.encoder_type:
      # embed: ... H, W, C
      if config.rssm.update_type in ['slot', 'cross']:
        embed = rearrange(embed, '... h w c -> ... (h w) c')
      else:
        embed = rearrange(embed, '... h w c -> ... (h w c)')
    else:
      # embed: ... (H W C)
      if config.rssm.update_type in ['slot', 'cross']:
        embed = rearrange(embed, '... d -> ... 1 d')
      else:
        pass  # nothing
    ###########################################################
    return embed

class CausalAgent(common.Module):

  def __init__(self, config, obs_space, act_space, step):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    # self.wm = WorldModel(config, obs_space, self.tfstep)

    if config.wm == 'default':
      self.wm = WorldModel(config, obs_space, self.tfstep)
    elif config.wm == 'fwm':
      from sandbox import dreamer_wrapper
      self.wm = dreamer_wrapper.FactorizedWorldModelWrapperForDreamer(config, obs_space, self.tfstep)
    elif config.wm == 'slate':
      from sandbox import slate_wrapper
      self.wm = slate_wrapper.SlateWrapperForDreamer(config, obs_space, self.tfstep)
    elif config.wm == 'dslate':
      # from sandbox import dynamic_slate_wrapper
      from sandbox import slate_wrapper
      self.wm = slate_wrapper.DynamicSlateWrapperForDreamer(config, obs_space, self.tfstep)
    else:
      raise NotImplementedError

    ###########################################################
    if config.wm == 'default':
      self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    elif config.wm == 'dslate':
      from sandbox import slot_behavior
      self._task_behavior = slot_behavior.SlotActorCritic(config, self.act_space, self.tfstep)
    else:
      raise NotImplementedError

    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      # NOTE: 12-4-21 5:25pm: did not integrate slots into expl_behavior
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())
    # *********************************************************
    # self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    # if config.expl_behavior == 'greedy':
    #   self._expl_behavior = self._task_behavior
    # else:
    #   self._expl_behavior = getattr(expl, config.expl_behavior)(
    #       self.config, self.act_space, self.wm, self.tfstep,
    #       lambda seq: self.wm.heads['reward'](seq['feat']).mode())
    ###########################################################

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    # if self.config.wm in ['fwm', 'slate', 'dslate'] and self.config.wm_only:
    if self.config.wm in ['fwm', 'slate'] or self.config.wm_only:
      random_policy = common.RandomAgent({'action': self.act_space})
      return random_policy(obs, state)
    else:
      obs = tf.nest.map_structure(tf.tensor, obs)
      tf.py_function(lambda: self.tfstep.assign(
          int(self.step), read_value=False), [], [])
      if state is None:
        latent = self.wm.rssm.initial(len(obs['reward']))
        # latent = latent['deter']  # HACK FOR NOW
        action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
        state = latent, action
      latent, action = state
      embed = self.wm.encoder(self.wm.preprocess(obs))
      ###########################################################
      # encoder to latent?
      embed = encoder_interface(embed, self.config)
      ###########################################################
      sample = (mode == 'train') or not self.config.eval_state_mean

      if self.config.wm in ['dslate']:
        _, latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs['is_first'], sample)  # we'll use this until we refactor the obs_step
      else:
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs['is_first'], sample)

      feat = self.wm.rssm.get_feat(latent)
      ###########################################################
      # latent to decoder?
      if self.config.wm == 'dslate' and not self.config.slot_behavior.use_slot_heads:
        feat = rearrange(feat, '... k featdim -> ... (k featdim)')
      ###########################################################
      if mode == 'eval':
        actor = self._task_behavior.actor(feat)
        action = actor.mode()  # (1, A)
        noise = self.config.eval_noise
      elif mode == 'explore':
        actor = self._expl_behavior.actor(feat)
        action = actor.sample()  # (1, A)
        noise = self.config.expl_noise
      elif mode == 'train':
        actor = self._task_behavior.actor(feat)
        action = actor.sample()
        noise = self.config.expl_noise
      action = common.action_noise(action, noise, self.act_space)
      outputs = {'action': action}
      state = (latent, action)
      return outputs, state


  # # @tf.function
  # def train(self, data, state=None):
  #   metrics = {}
  #   state, outputs, mets = self.wm.train(data, state)
  #   metrics.update(mets)
  #   if not self.config.wm_only:
  #     start = outputs['post']
  #     reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
  #     metrics.update(self._task_behavior.train(
  #         self.wm, start, data['is_terminal'], reward))
  #     if self.config.expl_behavior != 'greedy':
  #       mets = self._expl_behavior.train(start, outputs, data)[-1]
  #       metrics.update({'expl_' + key: value for key, value in mets.items()})
  #   return state, metrics



  # @tf.function
  def train(self, data, state=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    if not self.config.wm_only:
      """
        still want to train on the first step to initialize the variables

        we are looking at step = 1 rather than step = 0 because the wm already took a train step

        dslate  |  step = 1  |  step < delay  |  train_behavior

        False         -              -             True
        True        True             -             True
        True        False          True            False
        True        False          False           True

        if delay_train_behavior_by in [0,1,2]: always train
        if delay_train_behavior_by in [3, ...]: 
          train for the first step
          do not train for [delay_train_behavior_by-2] steps
          train for the rest
      """
      if not(self.config.wm == 'dslate' and self.wm.model.step < self.config.delay_train_behavior_by and not self.wm.model.step == 1):
        start = outputs['post']
        if self.config.wm == 'dslate':
          if not self.config.slot_behavior.use_slot_heads:
            reward = lambda seq: rearrange(slate_utils.bottle(self.wm.model.slot_model.heads['reward'])(
                rearrange(seq['feat'], 'h bt (k featdim) -> h bt k featdim', k=self.wm.model.slot_model.slot_attn.num_slots)), 'h bt 1 -> h bt')  # seq['feat'](H, B*T, K*D) --> reward (H, B*T)
          else:
            # reward = lambda seq: rearrange(slate_utils.bottle(self.wm.model.slot_model.heads['reward'])(seq['feat']), 'h bt 1 -> h bt')  # seq['feat'](H, B*T, K*D) --> reward (H, B*T)
            reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
             # seq['feat'](H, B*T, K, D) --> reward (H, B*T)
        else:
          reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
        metrics.update(self._task_behavior.train(
            self.wm, start, data['is_terminal'], reward))
        if self.config.expl_behavior != 'greedy':
          mets = self._expl_behavior.train(start, outputs, data)[-1]
          metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics



  # @tf.function --> turn this back on when you 
  def report(self, data):
    return self.wm.report(data)
    # report = {}
    # data = self.wm.preprocess(data)
    # for key in self.wm.heads['decoder'].cnn_keys:
    #   name = key.replace('/', '_')
    #   if self.config.rssm.num_slots > 1:
    #     report[f'openl_{name}'] = self.wm.slot_video_pred(data, key)
    #   else:
    #     report[f'openl_{name}'] = self.wm.video_pred(data, key)
    # return report


class WorldModel(common.Module):

  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.tfstep = tfstep

    # if config.rssm.num_slots > 1:
    #   from sandbox import slots_machine as machine
    # else:
    #   from sandbox import machine
    from sandbox import machine

    # import ipdb; ipdb.set_trace(context=20)
    if 'slot' in self.config:
      from sandbox import slot_machine
      # self.rssm = slot_machine.SlotEnsembleRSSM(**config.rssm)
      self.rssm = slot_machine.SlotEnsembleRSSM(config.rssm, config.slot.rssm)
    else:
      # self.rssm = machine.EnsembleRSSM(**config.rssm)
      self.rssm = machine.EnsembleRSSM(config.rssm)

    # self.rssm.register_num_slots(self.config.num_slots)  # TODO later this may vary based on the episode

    if config.encoder_type == 'default':
      self.encoder = machine.Encoder(shapes, **config.encoder)
    elif config.encoder_type in ['slot', 'slimslot', 'slimmerslot']:
      self.encoder = machine.PreviousSlotEncoder(shapes, config.encoder_type, config.rssm.embed_dim, **config.encoder)
    elif 'grid' in config.encoder_type:
      assert self.config.encoder.mlp_keys == '$^', 'I did not implement the integration of cnn grid ouput with mlp output'
      """
      grid_default
      grid_dvweak
      grid_dvstrong
      grid_sa
      grid_saslim
      grid_sadebug
      """
      self.encoder = slot_machine.GridEncoder(
        shapes=shapes, 
        encoder_type=config.encoder_type, 
        pos_encode_type=config.pos_encode_type, 
        outdim=config.rssm.embed_dim, 
        resolution=config.rssm.resolution, 
        slot_config=config.slot.encoder, 
        **config.encoder)
    else:
      raise NotImplementedError

    self.heads = {}

    if config.decoder_type == 'default':
      self.heads['decoder'] = machine.Decoder(shapes, **config.decoder)
    elif config.decoder_type in ['slot', 'slimmerslot']:
      decoder_in_dim = config.rssm.deter//self.config.rssm.num_slots + config.rssm.stoch//self.config.rssm.num_slots * config.rssm.discrete
      self.heads['decoder'] = machine.PreviousSlotDecoder(shapes, decoder_in_dim, config.decoder_type, **config.decoder)
    elif 'grid' in config.decoder_type:
        assert self.config.decoder.mlp_keys == '$^', 'I did not implement the integration of cnn grid ouput with mlp output'
        self.heads['decoder'] = slot_machine.GridDecoder(
          shapes=shapes, 
          decoder_type=config.decoder_type, 
          pos_encode_type=config.pos_encode_type, 
          token_dim=config.rssm.embed_dim, 
          resolution=config.rssm.resolution, 
          slot_config=config.slot.decoder, 
          **config.decoder)
    else:
      raise NotImplementedError

    if self.config.behavior_type == 'default':
      self.heads['reward'] = common.MLP([], **config.reward_head)
      if config.pred_discount:
        self.heads['discount'] = common.MLP([], **config.discount_head)
    elif self.config.behavior_type in ['sa', 'ca']:
      head_type = sm.SelfAttnHead if self.config.behavior_type == 'sa' else sm.CrossAttnHead
      self.heads['reward'] = head_type(
        shape=[], 
        # slot_size=self.config.reward_head.units,
        slot_size=self.config.rssm.hidden,
        dist_cfg=dict(dist=self.config.reward_head.dist),
        cfg=head_type.defaults())
      if config.pred_discount:
        self.heads['discount'] = head_type(
          shape=[],
          # slot_size=self.config.discount_head.units,
          slot_size=self.config.rssm.hidden,
          dist_cfg=dict(dist=self.config.discount_head.dist),
          cfg=head_type.defaults())
    else:
      raise NotImplementedError
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    """
      reward (B, T)
      is_first (B, T)
      is_last (B, T)
      is_terminal (B, T)
      image (B, T, H, W, C)
      orientations (B, T, D)
      height (B, T)
      velocity (B, T, V)
      action (B, T, A)


      outputs:
        embed: (B, T, X)
        feat: (B, T, F)
        post: 
          stoch: (B, T, S, V)
          logit: (B, T, S, V)
          deter: (B, T, D)
        prior:
          stoch: (B, T, S, V)
          logit: (B, T, S, V)
          deter: (B, T, D)
        likes:
          image: (B, T)
          reward: (B, T)
          discount: (B, T)
        kl: (B, T)

      metrics: (scalars)
        kl_loss:
        image_loss:
        reward_loss:
        discount_loss:
        model_kl:
        prior_ent:
        post_ent:
    """
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    """
      data['image'] = (B, T, H, W, C)
      data['action'] = (B, T, A)
    """
    data = self.preprocess(data)
    embed = self.encoder(data)
    ###########################################################
    # encoder to latent?
    embed = encoder_interface(embed, self.config)
    ###########################################################
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    ###########################################################
    # latent to decoder?
    if self.config.wm == 'dslate' and not self.config.slot_behavior.use_slot_heads:
      feat = rearrange(feat, '... k featdim -> ... (k featdim)')
    ###########################################################
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())


    try:
      tf.debugging.check_numerics(model_loss, 'model_loss')
    except Exception as e:
      lgr.debug(e)
      import ipdb; ipdb.set_trace(context=20)


    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  @tf.function
  def imagine(self, policy, start, is_terminal, horizon):
    """
    flattened: TensorShape([3, 2, 3, 16]) to TensorShape([6, 3, 16]) = (B*T, K, D)

    by the end of this, feat will be flattened (for the actor and critic), which means we will unflatten it for the reward function

    check whether the discount shape is correct
      check what it should be based on monolithic
      then check if it is what it should be for slots
    """
    flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    ###########################################################
    # latent to decoder?
    if self.config.wm == 'dslate' and not self.config.slot_behavior.use_slot_heads:
      start['feat'] = rearrange(start['feat'], '... k featdim -> ... (k featdim)')
    ###########################################################
    start['action'] = tf.zeros_like(policy(start['feat']).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      ###########################################################
      # latent to decoder?
      if self.config.wm == 'dslate' and not self.config.slot_behavior.use_slot_heads:
        feat = rearrange(feat, '... k featdim -> ... (k featdim)')
      ###########################################################
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}  # {deter: [H, B*T, D], logit: [H, B*T, S, V]}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount  # (H)
        disc = tf.concat([true_first[None], disc[1:]], 0)  # (H, B*T)
    else:
      # disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])  # (H, B*T)
        disc = self.config.discount * tf.ones(seq['feat'].shape[:2])  # (H, B*T)
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)  # (H, B*T)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = nmlz.center(value.astype(dtype) / 255.0)
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key):
    n = self.config.eval_dataset.batch
    t = self.config.eval_dataset.seed_steps
    nll = lambda dist, x: -tf.cast(dist.log_prob(x), tf.float32).mean()
    recon_nll = lambda dist, x: nll(dist, x[:n, :t])
    imag_nll = lambda dist, x: nll(dist, x[:n, t:])

    decoder = self.heads['decoder']
    truth = nmlz.uncenter(data[key][:n])
    embed = self.encoder(data)
    ###########################################################
    # encoder to latent?
    embed = encoder_interface(embed, self.config)
    ###########################################################
    states, _ = self.rssm.observe(
        embed[:n, :t], data['action'][:n, :t], data['is_first'][:n, :t])
    post_feat = self.rssm.get_feat(states)
    recon_dist = decoder(post_feat)[key]
    recon = recon_dist.mode()[:n]
    recon_loss = recon_nll(recon_dist, data[key])
    recon_reward_loss = recon_nll(self.heads['reward'](post_feat), data['reward'])
    model = nmlz.uncenter(recon[:, :t])
    imag_loss = tf.constant(0, dtype=tf.float32)
    imag_reward_loss = tf.constant(0, dtype=tf.float32)
    if self.config.dataset.length > t:
      init = {k: v[:, -1] for k, v in states.items()}
      prior = self.rssm.imagine(data['action'][:n, t:], init)
      prior_feat = self.rssm.get_feat(prior)
      openl_dist = decoder(prior_feat)[key]
      openl = openl_dist.mode()
      model = tf.concat([model, nmlz.uncenter(openl)], 1)
      imag_loss += imag_nll(openl_dist, data[key])
      imag_reward_loss +=  imag_nll(self.heads['reward'](prior_feat), data['reward'])
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    output = dict(
      video=rearrange(video, 'b t h w c -> t h (b w) c'),
      recon_loss=recon_loss,
      imag_loss=imag_loss,
      recon_reward_loss=recon_reward_loss,
      imag_reward_loss=imag_reward_loss,
      )
    return output

  @tf.function
  def slot_video_pred(self, data, key):
    n = self.config.eval_dataset.batch
    t = self.config.eval_dataset.seed_steps
    decoder = self.heads['decoder']
    truth = nmlz.uncenter(data[key][:n])
    embed = self.encoder(data)
    ###########################################################
    # encoder to latent?
    embed = encoder_interface(embed, self.config)
    ###########################################################
    states, _ = self.rssm.observe(
        embed[:n, :t], data['action'][:n, :t], data['is_first'][:n, :t])
    post_feat = self.rssm.get_feat(states)
    if self.config.rssm.num_slots > 1:
      post_feat = rearrange(post_feat, '... k featdim -> ... (k featdim)')
      decoded = decoder(post_feat, return_components=True)
      recon = decoded[key].mode()[:n]
      recon_components = decoded['components'].mode()[:n]
    else:
      recon = decoder(post_feat)[key].mode()[:n]
    if self.config.dataset.length > t:
      init = {k: v[:, -1] for k, v in states.items()}
      prior = self.rssm.imagine(data['action'][:n, t:], init)
      prior_feat = self.rssm.get_feat(prior)
      if self.config.rssm.num_slots > 1:
        prior_feat = rearrange(prior_feat, '... k featdim -> ... (k featdim)')
        decoded = decoder(prior_feat, return_components=True)
        openl = decoded[key].mode()
        openl_components = decoded['components'].mode()[:n]
      else:
        openl = decoder(prior_feat)[key].mode()
      model = tf.concat([nmlz.uncenter(recon[:, :t]), nmlz.uncenter(openl)], 1)
      if self.config.rssm.num_slots > 1:
        model_components = tf.concat([nmlz.uncenter(recon_components[:, :t]), nmlz.uncenter(openl_components)], 1)
    else:
      model = nmlz.uncenter(recon[:, :t])
      if self.config.rssm.num_slots > 1:
        model_components = nmlz.uncenter(recon_components[:, :t])

    if self.config.rssm.num_slots > 1:
      model_components = rearrange(model_components, 'b t k h w c -> b t (k h) w c')
      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error, model_components], 2)
    else:
      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error], 2)
    output = dict(
      video=rearrange(video, 'b t h w c -> t h (b w) c'))
    return output

  @tf.function
  def report(self, data):
    report = {}
    data = self.preprocess(data)
    for key in self.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      generate_video = self.video_pred
      output = generate_video(data, key)
      report[f'openl_{name}'] = output['video']
      report[f'recon_loss_{name}'] = output['recon_loss']
      report[f'imag_loss_{name}'] = output['imag_loss']
      report[f'recon_reward_loss_{name}'] = output['recon_reward_loss']
      report[f'imag_reward_loss_{name}'] = output['imag_reward_loss']
    return report


class ActorCritic(common.Module):
  @staticmethod
  def defaults():
      default_args = ml_collections.ConfigDict(dict(
        behavior_type='ca',

        ca_cfg=sm.CrossAttnHead.defaults(),
        sa_cfg=sm.SelfAttnHead.defaults(),
          ))
      return default_args

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    if self.config.behavior_type == 'default':
      self.actor = common.MLP(act_space.shape[0], **self.config.actor)
      self.critic = common.MLP([], **self.config.critic)
      if self.config.slow_target:
        self._target_critic = common.MLP([], **self.config.critic)
        self._updates = tf.Variable(0, tf.int64)
      else:
        self._target_critic = self.critic
    elif self.config.behavior_type in ['sa', 'ca']:
      head_type = sm.SelfAttnHead if self.config.behavior_type == 'sa' else sm.CrossAttnHead
      defaults = dict(ca=config.slot.behavior.ca_cfg, sa=config.slot.behavior.sa_cfg)
      self.actor = head_type(
        shape=act_space.shape[0],
        # slot_size=self.config.actor.units,
        slot_size=self.config.rssm.hidden,
        dist_cfg=dict(dist=self.config.actor.dist, min_std=self.config.actor.min_std),
        # cfg=head_type.defaults()
        cfg=defaults[config.slot.behavior.behavior_type]
        )
      self.critic = head_type(
        shape=[],
        # slot_size=self.config.critic.units,
        slot_size=self.config.rssm.hidden,
        dist_cfg=dict(dist=self.config.critic.dist),
        # cfg=head_type.defaults()
        cfg=defaults[config.slot.behavior.behavior_type]
        )
      if self.config.slow_target:
        self._target_critic = head_type(
          shape=[],
          # slot_size=self.config.critic.units,
          slot_size=self.config.rssm.hidden,
          dist_cfg=dict(dist=self.config.critic.dist),
          # cfg=head_type.defaults()
          cfg=defaults[config.slot.behavior.behavior_type]
          )
        self._updates = tf.Variable(0, tf.int64)
      else:
        self._target_critic = self.critic
    else:
      raise NotImplementedError
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  # @tf.function
  def train(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      reward = reward_fn(seq)  # (H, B*T)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  @tf.function
  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))  # (H, B*T, D). This is a dist, with event_shape D
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  @tf.function
  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    if self.config.critic_stop_grad:
      dist = self.critic(tf.stop_gradient(seq['feat'][:-1]))  # it does not matter whether we stop_gradient into seq['feat']
    else:
      dist = self.critic(seq['feat'][:-1])  # it does not matter whether we stop_gradient into seq['feat']
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()  # (H, B*T)
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
