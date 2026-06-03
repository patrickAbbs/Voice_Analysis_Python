import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Audio_File_Set
from Spectrogram_Generator import Generate_Audio_Spectrogram_Set
from Frequency_Distribution_Generator import Generate_Bucketed_Frequency_Distribution_Set
from Subdistribution_Extractor import Extract_Frequency_Subdistribution_Sets
from Subdistribution_Difference_Analyzer import Analyze_Subdistribution_Differences
from Layered_Occurrence_Count_Populator import Run_Layered_Occurence_Count_Population
from Layered_Subdistribution_Generator import Run_Layered_Subdistribution_Generation  

Layered_Subdistribution_Audio_Set = {"FCJF0": ["SA1", "SI648"], "MEDR0": ["SA1", "SI1374"], "MCPM0": ["SA1", "SA2", "SI564", "SI1194"], "FDAW0": ["SA1", "SA2", "SI1271", "SI1406"]}

#Layered_Subdistribution_Audio_Set = {"FCJF0": ["SA1", "SI648"], "MEDR0": ["SA1", "SI1374"]}
Subdistribution_Layer = "universal"



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
    all_audios_analysis_data = Analyze_Subdistribution_Differences(all_audios_analysis_data)

def Run_Subdstributions():
    #Run_Layered_Occurence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer)
    #Run_Layered_Subdistribution_Generation(Subdistribution_Layer)
    #Run_Layered_Subdistribution_Generation("voice", voice_set=["FCJF0", "MEDR0"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=True)
    Run_Layered_Subdistribution_Generation("phoneme", phoneme_set=["ae", "iy"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=True)


#Run_Analysis()
Run_Subdstributions()