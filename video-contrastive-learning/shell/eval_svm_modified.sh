#!/bin/bash

# data_dir="D:/UNI/Studentermedhjælper/Contrastive Learning/kinetics10_30fps_frames"
# output_dir="./video-contrastive-learning/results_eval"
# eval_dir="./video-contrastive-learning/results_eval/eval_svm"
# pretrained="./video-contrastive-learning/results/current.pth"

data_dir="D:/UNI/Studentermedhjælper/Contrastive Learning/kinetics10_30fps_frames"
output_dir="./video-contrastive-learning/results_1"
eval_dir="./video-contrastive-learning/results_1"
pretrained="./video-contrastive-learning/results/ckpt_epoch_1.pth"
# pretrained="./video-contrastive-learning/pretrain/moco_v2_800ep_pretrain.pth.tar"

num_replica=1

mkdir -p ${output_dir}
mkdir -p ${eval_dir}

# Using single GPU for feature extraction
python video-contrastive-learning/eval_svm_feature_extract.py \
    --data_dir="${data_dir}" \
    --datasplit=train \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}
python video-contrastive-learning/eval_svm_feature_extract.py \
    --data_dir="${data_dir}" \
    --datasplit=val \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}


python3 video-contrastive-learning/eval_svm_feature_scikit.py \
    --trainsplit=train \
    --valsplit=val \
    --output-dir=${eval_dir} \
    --num_replica=${num_replica}

