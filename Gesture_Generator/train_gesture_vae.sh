#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference_vqvae.py \
    --data_dir ./output/ \
    --name_file speaker01_clip01 \
    --checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231219_101453/Checkpoints/trained_model.pth \
    --checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231219_101453/config.json5 \
    --lxc_checkpoint_path /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231218_190058/Checkpoints/trained_model.pth\
    --lxc_checkpoint_config /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231218_190058/config.json5 \
    --device cuda:0 \
    --save_dir ./output/ 