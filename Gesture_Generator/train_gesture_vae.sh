#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference_vqvae.py \
    --data_dir ./test_audio_motion/ \
    --name_file speaker01_clip03 \
    --checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231211_152647/Checkpoints/trained_model.pth \
    --checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231211_152647/config.json5 \
    --lxc_checkpoint_path /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231206_115324/Checkpoints/trained_model.pth\
    --lxc_checkpoint_config /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231206_115324/config.json5 \
    --device cuda:0 \
    --save_dir ./test_audio_motion/ 