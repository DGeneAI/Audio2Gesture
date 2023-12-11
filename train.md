A intro for train your model
# Step 0
Preprocess data

# Step 1
```
cd Gesture_Lexicon
python c
```

# Step 2
adjust the config ./Config/MOCCA/config_vae.json5
Attention: lxm_dim: 192, make sure as same as step 1.
```
cd Gesture_Generator
python train.py ./Config/MOCCA/config_vae.json5

```
revise below config
```
--checkpoint_path /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231211_152647/Checkpoints/trained_model.pth \
--checkpoint_config /root/project/Audio2Gesture/Gesture_Generator/Training/MOCCA/_RNN_20231211_152647/config.json5 \
--lxc_checkpoint_path /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231206_115324/Checkpoints/trained_model.pth\
--lxc_checkpoint_config /root/project/Audio2Gesture/Gesture_Lexicon/Training/MOCCA/_vqvae1d_20231206_115324/config.json5 \
```
run ```./train_gesture_vae.sh```


# step 3

Note: lexicon_size: 2048 in ```./Data/MOCCA/Processed_4/Training_Data/config.json5```
```
python train.py ./Config/MOCCA/config.json5
```