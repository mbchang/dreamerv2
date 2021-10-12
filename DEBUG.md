10/8/21

For running on server:
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs dmc_vision --task dmc_walker_walk --cpu True --headless False

python dreamerv2/train.py --logdir runs/atari_pong/dreamerv2/1 --configs atari --task atari_pong --cpu True --headless False

debug:
rm -r runs
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs debug --task dmc_walker_walk

10/11/21
CUDA_VISIBLE_DEVICES=2 python dreamerv2/train.py --logdir runs/dynamics/default --configs dmc_vision --task dmc_cheetah_run --agent causal > runs/dynamics/default/debug.log &
CUDA_VISIBLE_DEVICES=1 python dreamerv2/train.py --logdir runs/dynamics/cross_attention --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.deter_model cross_attention > runs/dynamics/cross_attention/debug.log &