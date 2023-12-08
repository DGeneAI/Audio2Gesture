#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate_vqvae.py \
    --data_dir ./test_audio_motion_vqvae/ \
    --name_file cctv \
    --gen_checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231208_155557/Checkpoints/trained_model.pth \
    --gen_checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231208_155557/config.json5 \
    --lxm_intp_checkpoint_path /root/project/Audio2Gesture/Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231208_195522/Checkpoints/checkpoint_0k200.pth\
    --lxm_intp_checkpoint_config /root/project/Audio2Gesture/Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231208_195522/config.json5 \
    --device cuda:0 \
    --save_dir ./test_audio_motion_vqvae/
