10/8/21

For running on server:
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs dmc_vision --task dmc_walker_walk --cpu True --headless False

python dreamerv2/train.py --logdir runs/atari_pong/dreamerv2/1 --configs atari --task atari_pong --cpu True --headless False

debug:
rm -r runs
python3 dreamerv2/train.py --logdir runs/dmc_walker_walk/dreamerv2/1 --configs debug --task dmc_walker_walk



10/11/21
launch:

CUDA_VISIBLE_DEVICES=2 python dreamerv2/train.py --logdir runs/dynamics/default --configs dmc_vision --task dmc_cheetah_run --agent causal > runs/dynamics/default/debug.log &
CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/dynamics/cross_attention --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.deter_model cross_attention > runs/dynamics/cross_attention/debug.log &

10/13/21
CUDA_VISIBLE_DEVICES=1 python dreamerv2/train.py --logdir runs/dynamics/slim_cross_attention --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention > runs/dynamics/slim_cross_attention/debug.log &

CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/dynamics/separate_embedding --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics separate_embedding > runs/dynamics/separate_embedding/debug.log &

CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/update/slim_attention --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention > runs/update/slim_attention/debug.log &

CUDA_VISIBLE_DEVICES=1 python dreamerv2/train.py --logdir runs/encoder/slot_res64_out1 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --encoder_type slot > runs/encoder/slot_res64_out1/debug.log &

CUDA_VISIBLE_DEVICES=2 python dreamerv2/train.py --logdir runs/decoder/slot_res64 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slot > runs/decoder/slot_res64/debug.log &

--> next will run an experiment with everything, except for the slot attention component

CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/encoder/slimslot --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --encoder_type slimslot > runs/encoder/slimslot/debug.log &

CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/decoder/batch8 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slot --dataset.batch 8 > runs/decoder/batch8/debug.log &

CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/encoderdecoder/batch8_slimslot --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slot --dataset.batch 8 --encoder_type slimslot > runs/encoderdecoder/batch8_slimslot/debug.log &

CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/encoderdecoder/batch8_slimslot_3dembed --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slot --dataset.batch 8 --encoder_type slimslot > runs/encoderdecoder/batch8_slimslot_3dembed/debug.log &

debug:

CUDA_VISIBLE_DEVICES=2 python dreamerv2/train.py --logdir runs/debug --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --encoder_type default

CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/debug/slot_res64_out1_ --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --encoder_type slot

10/14/21
debug:

rm -r runs/*
python dreamerv2/train.py --logdir runs/debug --configs debug --task dmc_walker_walk --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slimmerslot --encoder_type slimmerslot

do this as reference
python dreamerv2/train.py --logdir runs/debug --configs debug --task dmc_walker_walk --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention


10/16/21

CUDA_VISIBLE_DEVICES=3 python dreamerv2/train.py --logdir runs/slot_attention/ns1_constant --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --dataset.batch 8 --encoder_type slimslot > runs/slot_attention/ns1_constant/debug.log &


10/17/21
Debugging:

python dreamerv2/train.py --logdir runs/debug/slim_update --configs debug --task dmc_walker_walk --agent causal --rssm.dynamics slim_cross_attention --rssm.update slim_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 1

python dreamerv2/train.py --logdir runs/debug/slot --configs debug --task dmc_walker_walk --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot

python dreamerv2/train.py --logdir runs/debug/default --configs debug --task dmc_walker_walk --agent causal


Next:
Run and debug
python dreamerv2/train.py --logdir runs/debug/slot2 --configs debug --task dmc_walker_walk --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 2

Once you get the multi-slot version to work, then make sure that the above commands (10/17/21) for slot=1 still works as expected.

Then after that remove the //self.num_slots. It would be nice at that point to use ml_collections


10/19/21
CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py --logdir runs/slot_attention/ns2_d200_s32 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --encoder_type slimslot --dataset.batch 8 --rssm.num_slots 2 &

CUDA_VISIBLE_DEVICES=1 python dreamerv2/train.py --logdir runs/slot_attention/ns4_d200_s32 --configs dmc_vision --task dmc_cheetah_run --agent causal --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slot --encoder_type slimslot --dataset.batch 8 --rssm.num_slots 4 &

(but did not run because nfs seemed to be hanging)



10/30/21
debug:
python dreamerv2/train.py --logdir runs/debug/merge/default --configs debug --task dmc_walker_walk --agent causal

python dreamerv2/train.py --logdir runs/debug/merge/default --configs debug --task balls_whiteball_push --agent causal --wm_only=True

