#!/bin/bash

# data_dir="D:/UNI/Studentermedhjælper/Contrastive Learning/kinetics10_30fps_frames"
data_dir="D:/UNI/Studentermedhjaelper/Contrastive Learning/tactileSensing/DataCollection"
output_dir="./video-contrastive-learning/results_gs"
pretrained="video-contrastive-learning/pretrain/moco_v2_800ep_pretrain.pth.tar"

mkdir -p "${output_dir}"

# torchrun --nproc_per_node=1 train_vclr.py \
#     --data_dir="${data_dir}" \
#     --datasplit=train \
#     --pretrained_model="${pretrained}" \
#     --output_dir="${output_dir}" \
#     --model_mlp \
#     --dataset KineticsClipFolderDatasetOrderTSN \
#    --batch_size 32

python video-contrastive-learning/train_vclr_no_parallel.py \
    --data_dir="${data_dir}" \
    --datasplit=train \
    --pretrained_model="${pretrained}" \
    --output_dir="${output_dir}" \
    --model_mlp \
    --dataset KineticsClipFolderDatasetOrderTSN \
    --batch_size 16