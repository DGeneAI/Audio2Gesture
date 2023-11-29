#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --data_dir ./test_audio_motion/ \
    --name_file cctv \
    --gen_checkpoint_path ./Gesture_Generator/Training/MOCCA/_RNN_20231113_113751/Checkpoints/trained_model.pth \
    --gen_checkpoint_config ./Gesture_Generator/Training/MOCCA/_RNN_20231113_113751/config.json5 \
    --lxm_intp_checkpoint_path ./Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231113_141334/Checkpoints/trained_model.pth \
    --lxm_intp_checkpoint_config ./Lexeme_Interpreter/Training/MOCCA/_LxmInterpreter_20231113_141334/config.json5 \
    --device cuda:0 \
    --save_dir ./test_audio_motion/
