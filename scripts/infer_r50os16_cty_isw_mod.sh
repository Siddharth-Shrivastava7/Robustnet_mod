#!/usr/bin/env bash
# stop the script running if args are not set
set -o nounset

echo "Running inference on" ${1}
echo "Saving Results :" ${2}

# python -m torch.distributed.launch --nproc_per_node=2 eval_mod.py \
#     --dataset cityscapes \
#     --arch network.deepv3.DeepR50V3PlusD \
#     --inference_mode sliding \
#     --scales 0.5,1.0,2.0 \
#     --split val \
#     --crop_size 1024 \
#     --cv_split 0 \
#     --ckpt_path ${2} \
#     --snapshot ${1} \
#     --wt_layer 0 0 2 2 2 0 0 \
#     --dump_images \

python -m torch.distributed.launch --nproc_per_node=2 eval_mod.py \
    --dataset darkzurich \
    --arch network.deepv3.DeepR50V3PlusD \
    --inference_mode sliding \
    --scales 0.5,1.0,2.0 \
    --split val \
    --crop_size 1024 \
    --cv_split 0 \
    --ckpt_path ${2} \
    --snapshot ${1} \
    --wt_layer 0 0 2 2 2 0 0 \
    --dump_images \
    --exp_name darkzurich


# python -m torch.distributed.launch --nproc_per_node=2 eval_mod.py \
#     --dataset bdd100k \
#     --arch network.deepv3.DeepR50V3PlusD \
#     --inference_mode sliding \
#     --scales 0.5,1.0,2.0 \
#     --split val \
#     --crop_size 1024 \
#     --cv_split 0 \
#     --ckpt_path ${2} \
#     --snapshot ${1} \
#     --wt_layer 0 0 2 2 2 0 0 \
#     --dump_images \

