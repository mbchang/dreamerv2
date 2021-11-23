import datetime
import pathlib

def watch(args_to_watch, abbrvs):
    def _fn(args):
        exp_name = []
        for key in args_to_watch:
            val = getattr(args, key)
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

        'eval_dataset.length': 'eT',
        'eval_dataset.seed_steps': 'ss',
        'rssm.stoch': 'S',
        'rssm.discrete': 'V',
        'seed': 's',
        'fwm.model.dim': 'd',

        'slate.slot_model.slot_attn.num_slots': 'k',
        'dslate.slot_model.slot_attn.num_slots': 'k',

    }
    watcher = watch(args.watch, abbrvs)
    expname = pathlib.Path(args.task) / f'{watcher(args)}_{datetime.datetime.now():%Y%m%d%H%M%S}'
    return expname