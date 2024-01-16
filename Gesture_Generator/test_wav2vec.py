# import torch
# import fairseq
# cp_path = '/root/project/Audio2Gesture/checkpoint/wav2vec/vq-wav2vec.pt'
# model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
# model = model[0]
# model.eval()

# wav_input_16khz = torch.randn(1,16000)
# z = model.feature_extractor(wav_input_16khz)
# _, idxs = model.vector_quantizer.forward_idx(z)
# print(idxs.shape)


# import torch
# import fairseq

# cp_path = '/root/project/Audio2Gesture/checkpoint/wav2vec/wav2vec_large.pt'
# model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
# model = model[0]
# model.eval()

# wav_input_16khz = torch.randn(1,16000)
# z = model.feature_extractor(wav_input_16khz)
# c = model.feature_aggregator(z)



import torch
import torchaudio
import platform
import math
import numpy as np
import os, cv2, argparse, subprocess
import time
from tqdm import tqdm
from torch.nn import functional as F
from argparse import Namespace
from python_speech_features import logfbank
from fairseq import checkpoint_utils, utils, tasks
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
# from utils.data_avhubert import collater_audio, emb_roi2im
# from utils.inference_utils import inpatient

# from models.talklip import TalkLip
# import face_detection


def load_audio(dir_audio: str, names_file: List[str], dir_save: str,
               sr: int = 48000, normalize_loudness: bool = True, save: bool = False) -> List[np.ndarray]:
    res = []

    for n in names_file:
        path_audio = os.path.join(dir_audio, n + ".wav")
        audio, _ = librosa.load(path_audio, sr=sr)

        if normalize_loudness:
            meter = pyln.Meter(sr)  # create BS.1770 meter
            loudness = meter.integrated_loudness(audio)
            # loudness normalize audio to -20 dB LUFS
            audio = pyln.normalize.loudness(audio, loudness, -20.0)

        if save:
            np.save(os.path.join(dir_save, n + ".npy"), audio)

        res.append(audio)

        print(n + ':', res[-1].shape)

    return res

def fre_audio(wav_data, sample_rate):
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats
    if len(wav_data.shape)>1:
        audio_feats = logfbank(wav_data[:,0], samplerate=sample_rate).astype(np.float32)  # [T, F]
    else:
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, 4)  # [T/stack_order_audio, F*stack_order_audio]
    return audio_feats


wav_path = r'/root/project/Audio2Gesture/test_audio_motion_copy/speaker01_clip03.wav'
sr = 16000
path_audio = os.path.join(dir_audio, n + ".wav")
audio, _ = librosa.load(path_audio, sr=sr)

meter = pyln.Meter(sr)  # create BS.1770 meter
loudness = meter.integrated_loudness(audio)
# loudness normalize audio to -20 dB LUFS
audio = pyln.normalize.loudness(audio, loudness, -20.0)
    
    
sampRate, wav = wavfile.read(wav_path)
spectrogram = fre_audio(wav, sampRate)
spectrogram = torch.tensor(spectrogram)  # T'* F

with torch.no_grad():
    spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

audioBatch = spectrogram.unsqueeze(dim=0).transpose(1, 2)

