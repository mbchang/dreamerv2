10/8/21

For running on server:
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs dmc_vision --task dmc_walker_walk --cpu True --headless False

python dreamerv2/train.py --logdir runs/atari_pong/dreamerv2/1 --configs atari --task atari_pong --cpu True --headless False

debug:
rm -r runs
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs debug --task dmc_walker_walk