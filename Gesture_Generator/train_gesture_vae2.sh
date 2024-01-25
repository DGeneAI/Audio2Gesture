#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference_vqvae2.py \
    --data_dir ./test_trinity/ \
    --name_file Recording_001 \
    --checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/Trinity/_RNN_20240124_130542/Checkpoints/trained_model.pth \
    --checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/Trinity/_RNN_20240124_130542/config.json5  \
    --lxc_checkpoint_path /root/project/Audio2Gesture/Gesture_Lexicon/Training/Trinity/_vqvae1d_20240123_104647/Checkpoints/trained_model.pth \
    --lxc_checkpoint_config /root/project/Audio2Gesture/Gesture_Lexicon/Training/Trinity/_vqvae1d_20240123_104647/config.json5 \
    --device cuda:0 \
    --save_dir ./test_trinity/ 