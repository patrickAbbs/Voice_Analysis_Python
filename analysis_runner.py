import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Audio_File_Set
from Spectrogram_Generator import Generate_Audio_Spectrogram_Set
from Frequency_Distribution_Generator import Generate_Bucketed_Frequency_Distribution_Set
from Subdistribution_Extractor import Extract_Frequency_Subdistribution_Sets


class Audio_Analysis_Data:
    def __init__(self, audio_file_name):
        self.Audio_File_Name = audio_file_name
        self.Spectrogram_Data = None
        self.Typed_Bucketed_Frequency_Progressions = None
        self.Typed_Bucketed_Frequency_Distributions = None
        self.Frequency_Bucket_Centers = None
        self.Typed_Tiered_Subdistributions = None


def Run_Analysis():
    all_audios_analysis_data = []
    for audio_file in Audio_File_Set:
        all_audios_analysis_data.append(Audio_Analysis_Data(audio_file))
    all_audios_analysis_data = Generate_Audio_Spectrogram_Set(all_audios_analysis_data)
    all_audios_analysis_data = Generate_Bucketed_Frequency_Distribution_Set(all_audios_analysis_data)
    all_audios_analysis_data = Extract_Frequency_Subdistribution_Sets(all_audios_analysis_data)


Run_Analysis()