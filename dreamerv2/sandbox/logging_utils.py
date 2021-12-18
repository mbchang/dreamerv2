import datetime
import pathlib

def watch(args_to_watch, abbrvs):
    def _fn(args):
        exp_name = []
        for key in args_to_watch:
            val = getattr(args, key)
            if isinstance(val, bool):
                exp_name.append(f'{abbrvs[key]}T' if val else f'{abbrvs[key]}F')
            else:
                exp_name.append(f'{abbrvs[key]}{val}')
        exp_name = '_'.join(exp_name)
        return exp_name
    return _fn

def create_expname(args):
    abbrvs = {
        'task': '',
        'precision': 'p',

        'fwm.optim.learning_rate': 'flr',
        'dslate.slot_model.lr': 'flr',
        'slate.slot_model.lr': 'flr',

        'fwm.optim.warmup_steps': 'fws',
        'fwm.optim.decay_steps': 'fds',
        'fwm.optim.min_lr': 'fmlr',
        'fwm.model.posterior_loss': 'pl',
        'fwm.model.encoder_type': 'et',
        'wm_only': 'wmo',
        'dataset.batch': 'B',
        'dataset.length': 'T',

        'fwm.model.update_step.temp': 'tp',
        'dslate.slot_model.slot_attn.temp': 'tp',
        'slate.slot_model.slot_attn.temp': 'tp',

        'eval_dataset.length': 'eT',
        'eval_dataset.seed_steps': 'ss',
        'rssm.stoch': 'S',
        'rssm.discrete': 'V',
        'seed': 's',
        'fwm.model.dim': 'd',

        'slate.slot_model.slot_attn.num_slots': 'k',
        'dslate.slot_model.slot_attn.num_slots': 'k',

        'dslate.slot_model.consistency_loss': 'cl',

        'replay.minlen': 'rmnl',
        'replay.maxlen': 'rmxl',

        'dslate.slot_model.decay_steps': 'ds',
        'dslate.slot_model.min_lr_factor': 'mlf',

        'dslate.curr': 'cr',
        'dslate.curr_every': 'ce',

        'dslate.slot_model.hack_is_first': 'hkif',
        'dslate.slot_model.handle_is_first': 'hnif',

        'critic_stop_grad': 'csg',

        'delay_train_behavior_by': 'dly',

        'dslate.dvae.weak': 'dvwk',
        'dslate.dvae.sm_hard': 'dvsmh',

        'dslate.mono_train': 'mt',
        'dslate.slot_model.einsum_dict': 'esd',
        'dslate.stop_gradient_input': 'sgi',
        'dslate.stop_gradient_output': 'sgo',

        'dslate.slot_model.d_model': 'dim',
        'dslate.e2e': 'e2e',

        'dslate.slot_model.distributional': 'latdist',

        'rssm.update': 'ru',
        'rssm.dynamics': 'rd',
        'rssm.initial': 'ri',


    }
    watcher = watch(args.watch, abbrvs)
    expname = pathlib.Path(args.task) / f'{watcher(args)}_{datetime.datetime.now():%Y%m%d%H%M%S}'
    return expname