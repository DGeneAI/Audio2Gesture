## Rhythmic Gesticulator: A Reproduced Version

This is a reproduced PyTorch/GPU implementation of the paper [Rhythmic Gesticulator: Rhythm-Aware Co-Speech Gesture Synthesis with Hierarchical Neural Embeddings](https://pku-mocca.github.io/Rhythmic-Gesticulator-Page/) by *[Ironieser](https://github.com/Ironeiser)*.  
### 🚩Feature 
- &#x2705;  Add VQ-VAE for gesture lexicon.  
- &#x2705;  Improve data preprocess
- &#x2705;  Add *L_perc* and *L_z*(for style control)
- &#x2705;  Add perceptual loss
- &#x2705; Use vq-wav2vec as audio encoder
- &#x231B;  leverage Gumbel-softmax to convert a latent code into a codebook vector
- More coming soon.
###  🐛 Existing bug 
- 💥  The official  open-source code for the **gesture generator model** collapsed.

- 💥 The official  open-source code for the **Motion Encoder** of  **gesture generator model**  is missing.
- 💥 There is only on loss of reconstructive loss when training the gesture generator. Three loss functions are not provided, which is *L_perc*, *L_lexeme* and *L_z*(for style control). 
### 🐛 Fixed Bug
- &#x2705; The official  open-source code for the **Audio Encoder** does not use pre-trained encoder which is vq-wav2vec has L=8 conv layers. And do not finu-tine audio encoder with text using contrastive learning strategy.
- &#x2705; There is only on loss of reconstructive loss when training the gesture generator. Three loss functions are not provided, which is *L_perc*, *L_lexeme* and *L_z*(for style control).   

This version is based on [Trinity Speech-Gesture Dataset (GENEA Challenge 2020)](https://trinityspeechgesture.scss.tcd.ie/) and  [MOCCA Gesture Dataset](https://github.com/Aubrey-ao/HumanBehaviorAnimation/tree/main/HumanBehaviorAnimation/RhythmicGesticulator/MOCCA_Gesture_Dataset).
If want to get better performances of motion quality and speech generalization, you can try to train the system with bigger datasets like [BEAT Dataset](https://github.com/PantoMatrix/BEAT).

### Install

Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html.

``` shell
cd HumanBehaviorAnimation/RhythmicGesticulator/Simplified_Version
conda env create -f environment.yaml
conda activate rhyGes_simple
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

#### vq-wav2vec

Install vq-wav2vec refer to https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md

install fairseq and download the ckpt, save to ```./checkpoint/wav2vec/vq-wav2vec.pt```

### Sample

Download the pretrained source from [Google Drive](https://drive.google.com/file/d/1oIbZygcHivxWcRkIki3zis6LhCklpm8L/view?usp=sharing) and put the .zip file into the root directory of this project. Then, run the script to automatically process the pretrained source:

``` shell
python process_pretrained_source.py
```

Put the test audio file {audio_file} (in .wav format) into a directory "path/{data_dir}" built by yourself, and run the script to generate gestures:

``` shell
# Only support single GPU.

CUDA_VISIBLE_DEVICES={gpu} python generate.py \
--data_dir path/{data_dir} \
--name_file {audio_file} \
--gen_checkpoint_path ./Gesture_Generator/Training/Trinity/RNN/Checkpoints/trained_model.pth \
--gen_checkpoint_config ./Gesture_Generator/Training/Trinity/RNN/config.json5 \
--lxm_intp_checkpoint_path ./Lexeme_Interpreter/Training/Trinity/Transformer/Checkpoints/trained_model.pth \
--lxm_intp_checkpoint_config ./Lexeme_Interpreter/Training/Trinity/Transformer/config.json5 \
--device cuda:0 \
--save_dir path/{data_dir}  # The directory of saving the generated motion, which usually equals to the directory of the test audio file.
```

### Training

#### Dataset

Download [Trinity Speech-Gesture Dataset (GENEA Challenge 2020)](https://trinityspeechgesture.scss.tcd.ie/), and put the dataset into folder {Data} like:

```
- Data
  - Trinity
    - Source
      - Training_Data
        - Audio
          - Recording_001.wav
            ...
        - Motion
          - Recording_001.bvh
            ...
      - Test_Data
        - Audio
          - TestSeq001.wav
            ...
        - Motion
          - TestSeq001.bvh
            ...
```

Then, pre-process data (it will take about 30-50 minutes):

``` shell
cd Data_Preprocessing
# python preprocess.py ./Config/Trinity/config.json5
python preprocess.py ./Config/MOCCA/config_vae.json5
```

#### Run

First, build the Gesture Lexicon:

``` shell
cd Gesture_Lexicon


## Train the motion autoencoder. 
# It will take about 3 hours (tested on a NVIDIA GeForce RTX 3090). 
# Only support single GPU training.
# Both net architectures are adequate and the Transformer excels in temporal perception.

CUDA_VISIBLE_DEVICES=0 python train.py ./Config/Trinity/config_transformer.json5  # Or "./Config/Trinity/config_conv1d.json5". 

# When training finished, the model will be saved in "./Training/Trinity/{log}". {log} is the directory of saving the trained model.


## Build the gesture lexicon.
# It will take about 5 minutes.
# Only support single GPU.

CUDA_VISIBLE_DEVICES=0 python lexicon.py \
--data_dir ../Data/MOCCA/Processed/Training_Data \
--checkpoint_path ./Training/MOCCA/Transformer/Checkpoints/trained_model.pth \
--checkpoint_config ./Training/MOCCA/Transformer/config.json5 \
--lexicon_size 2048 \
--save
```

Second, train the Gesture Generator:

``` shell
cd Gesture_Generator


## Train the gesture generator.
# It will take about 40 minutes (tested on a NVIDIA GeForce RTX 3090).
# Only support single GPU training.

CUDA_VISIBLE_DEVICES=0 python train.py ./Config/MOCCA/config.json5


## Inference for test.
# Prepare the test audio-motion pair (extract gesture lexemes from the provided motion) and put them into folder "path/{Test}" like:

- {Test}
  - {test_seq}.wav
  - {test_seq}.bvh

# Note that the names of the audio and motion files must be the same.
# Then, run the inference script:
# Only support single GPU.

CUDA_VISIBLE_DEVICES=0 python inference.py \
--data_dir ./test_audio_motion/ \  # The directory of the test audio-motion pair.
--name_file speaker01_clip03 \  # The name of the test file.
--checkpoint_path ./Training/MOCCA/RNN/Checkpoints/trained_model.pth \ 
--checkpoint_config ./Training/MOCCA/RNN/config.json5 \
--lxc_checkpoint_path ../Gesture_Lexicon/Training/MOCCA/_Transformer_20231110_142117/Checkpoints/trained_model.pth \
--lxc_checkpoint_config ../Gesture_Lexicon/Training/MOCCA/_Transformer_20231110_142117/config.json5 \
--device cuda:0 \
--save_dir ./test_audio_motion/  # The directory of saving the generated motion, which usually equals to the directory of the test audio-motion pair.
```
```sh


Third, train the Lexeme Interpreter:

``` shell
cd Lexeme_Interpreter


## Train the lexeme interpreter.
# It will take about 11 minutes (tested on a NVIDIA GeForce RTX 3090).
# Only support single GPU training.

CUDA_VISIBLE_DEVICES=0 python train.py ./Config/MOCCA/config.json5
```

### Citation

```
@article{Ao2022RhythmicGesticulator,
    author = {Ao, Tenglong and Gao, Qingzhe and Lou, Yuke and Chen, Baoquan and Liu, Libin},
    title = {Rhythmic Gesticulator: Rhythm-Aware Co-Speech Gesture Synthesis with Hierarchical Neural Embeddings},
    year = {2022},
    issue_date = {December 2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {41},
    number = {6},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3550454.3555435},
    doi = {10.1145/3550454.3555435},
    journal = {ACM Trans. Graph.},
    month = {nov},
    articleno = {209},
    numpages = {19}
}
```

### Limitations

* The current version only supports single GPU training and inference.
* (Added by Ironeiser.)~~The current version uses "autoencoder + k-means" to build the gesture lexicon, which is not the best way. Using "VQ-VAE" or "Gumbel-Softmax" to build the gesture lexicon can get better performance.~~ 
* The current version deletes the "Text Input" part.
* The current version deletes the "Slient Period Hint" part.
* The current version deletes the ”Gesture Style Code“ part.