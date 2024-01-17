# region Import.

import os
import sys
import json5
import shutil

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from Utils.audio_features import *
from Utils.motion_features import *
from Utils.beat_features import *
from Utils.utils import *

# endregion

if __name__ == '__main__':
    dir_wav = "/root/project/Audio2Gesture/Data/MOCCA/Processed_4/Training_Data/Features/WAV_Audio"
    name_files = ["speaker01_clip01", 
                        "speaker01_clip02",
                        "speaker01_clip04",
                        "speaker02_clip01",
                        "speaker02_clip04",
                        "speaker02_clip06",
                        "speaker03_clip01",
                        "speaker03_clip03",
                        "speaker03_clip05",
                        "speaker04_clip01",
                        "speaker04_clip03",
                        "speaker04_clip04",
                        "speaker05_clip01",
                        "speaker05_clip03",
                        "speaker05_clip05"]

    fps = 20
    sr = 48000
    dir_onset = 'test'
    dir_mel_motion_aligned = "/root/project/Audio2Gesture/Data/MOCCA/Processed_4/Training_Data/Features/Mel_Motion_Aligned"
    dir_onset = "/root/project/Audio2Gesture/Data/MOCCA/Processed_4/Training_Data/Features/Onset"
    dir_data_len_uniform = "test"
    _, _, _ = uniform_data_fragment_length_tsm(dir_mel_motion_aligned=dir_mel_motion_aligned,
                                                       dir_onset=dir_onset, dir_wav=dir_wav, names_file=name_files,
                                                       dir_save=dir_data_len_uniform, sr=sr, fps=fps,
                                                       rotation_order="ZXY",
                                                       uniform_len=12, save=False)