#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --data_dir ./output/ \
    --name_file speaker01_clip01 \
    --checkpoint_path ./Training/MOCCA/_RNN_20231113_113751/Checkpoints/trained_model.pth \
    --checkpoint_config ./Training/MOCCA/_RNN_20231113_113751/config.json5 \
    --lxc_checkpoint_path ../Gesture_Lexicon/Training/MOCCA/_Transformer_20231110_142117/Checkpoints/trained_model.pth \
    --lxc_checkpoint_config ../Gesture_Lexicon/Training/MOCCA/_Transformer_20231110_142117/config.json5 \
    --device cuda:0 \
    --save_dir ./output/ 