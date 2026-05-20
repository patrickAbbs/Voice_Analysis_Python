import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Audio_File_Set
from Spectrogram_Generator import Generate_Audio_Spectrogram_Set

def Run_Analysis():
    audios_analysis_data = {}
    for audio_file in Audio_File_Set:
        audios_analysis_data[audio_file] = {}
    audios_analysis_data =  Generate_Audio_Spectrogram_Set(audios_analysis_data)


Run_Analysis()