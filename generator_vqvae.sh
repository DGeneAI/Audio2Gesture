#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate_vqvae.py \
    --data_dir ./test_vqvae/ \
    --name_file speaker01_clip03 \
    --gen_checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231213_180121/Checkpoints/trained_model.pth \
    --gen_checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231213_180121/config.json5 \
    --lxm_intp_checkpoint_path /root/project/Audio2Gesture/Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231218_171003/Checkpoints/checkpoint_0k200.pth\
    --lxm_intp_checkpoint_config /root/project/Audio2Gesture/Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231218_171003/config.json5 \
    --device cuda:0 \
    --save_dir ./test_vqvae/
