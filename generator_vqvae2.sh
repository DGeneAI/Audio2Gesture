#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate_vqvae2.py \
    --data_dir ./test_vqvae_trinity/ \
    --name_file maggie_gl \
    --gen_checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/Trinity/_RNN_20240124_130542/Checkpoints/trained_model.pth \
    --gen_checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/Trinity/_RNN_20240124_130542/config.json5 \
    --lxm_intp_checkpoint_path /root/project/Audio2Gesture/Lexeme_Interpreter/Training/Trinity/_LxmInterpreter_20240125_113601/Checkpoints/checkpoint_0k200.pth\
    --lxm_intp_checkpoint_config /root/project/Audio2Gesture/Lexeme_Interpreter/Training/Trinity/_LxmInterpreter_20240125_113601/config.json5 \
    --device cuda:0 \
    --save_dir ./test_vqvae_trinity/
