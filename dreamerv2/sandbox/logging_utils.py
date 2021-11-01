def watch(args_to_watch, abbrvs):
    def _fn(args):
        exp_name = []
        for key in args_to_watch:
            val = getattr(args, key)
            exp_name.append(f'{abbrvs[key]}{val}')
        exp_name = '_'.join(exp_name)
        return exp_name
    return _fn

# def create_expname(args):
#     abbrvs = {
#         'task': 'task',
#         'opt.lr': 'lr',
#         'opt.lr_decay_gamma': 'ldg',
#         'num_slots': 'k',
#         'num_frames': 'T',
#         'batch_size': 'bs',
#         'experiment': 'ex',
#         'model.slot_dim': 'd',
#         'model.action_encoder': 'ae',
#         'curr.on': 'cr',
#         'opt.cap': 'cp',
#         'model.factorized': 'fct',
#         'model.dynamics_iters': 'di',
#         'model.encoder': 'e'
#     }
#     watcher = watch(args.watch, abbrvs)
#     expname = f'{os.path.basename(args.dataconfig)}_{args.task}'
#     expname += f'_{watcher(args)}'
#     return expname

"""
Example

CUDA_VISIBLE_DEVICES=6 DISPLAY=:0 python train.py --cfg.task as --cfg.watch "('experiment', 'opt.lr', 'num_frames', 'opt.lr_decay_gamma', 'model.factorized', 'curr.on', 'opt.cap', 'model.dynamics_iters', 'model.encoder')" --cfg.headless &

"""