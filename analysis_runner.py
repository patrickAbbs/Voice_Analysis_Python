import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Audio_File_Set
from Spectrogram_Generator import Generate_Audio_Spectrogram_Set
from Frequency_Distribution_Generator import Generate_Bucketed_Frequency_Distribution_Set
from Subdistribution_Extractor import Extract_Frequency_Subdistribution_Sets
from Subdistribution_Difference_Analyzer import Analyze_Subdistribution_Differences
from Layered_Occurrence_Count_Populator import Run_Layered_Occurrence_Count_Population, Run_Subtractive_Layered_Occurrence_Count_Population
from Layered_Subdistribution_Generator import Run_Layered_Subdistribution_Generation, Run_Subtractive_Layered_Subdistribution_Generation
from Voice_Subdistribution_Deviation_Tracker import Run_Voice_Subdistribution_Deviation_Tracking

Layered_Subdistribution_Audio_Set = {
    "FCJF0": ["SA1", "SA2", "SI648", "SI1027", "SI1657", "SX37", "SX127", "SX217", "SX307", "SX397"], 
    "MEDR0": ["SA1", "SA2", "SI744", "SI1374", "SI2004", "SX24", "SX114", "SX204", "SX294", "SX384"], 
    "MCPM0": ["SA1", "SA2", "SI564", "SI1194", "SI1824", "SX24", "SX114", "SX204", "SX294", "SX384"], 
    "FDAW0": ["SA1", "SA2", "SI1271", "SI1406", "SI2036", "SX56", "SX146", "SX236", "SX326", "SX416"], 
    "FDML0": ["SA1", "SA2", "SI1149", "SI1779", "SI2075", "SX69", "SX159", "SX249", "SX339", "SX429"], 
    "MDAC0": ["SA1", "SA2", "SI631", "SI1261", "SI1837", "SX91", "SX181", "SX271", "SX361", "SX451"],
    "FECD0": ["SA1", "SA2", "SI788", "SI1418", "SI2048", "SX68", "SX158", "SX248", "SX338", "SX428"],
    "MDPK0": ["SA1", "SA2", "SI552", "SI1053", "SI1683", "SX63", "SX153", "SX243", "SX333", "SX423"],
    }

#Layered_Subdistribution_Audio_Set = {"FCJF0": ["SA1", "SI648"], "MEDR0": ["SA1", "SI1374"]}
Subdistribution_Layer = "phoneme"

Voice_Subdistribution_Deviation_Audio_Set = {
    "FCJF0": ["SA1", "SA2", "SI648", "SI1027", "SI1657", "SX37", "SX127", "SX217", "SX307", "SX397"], 
    "MEDR0": ["SA1", "SA2", "SI744", "SI1374", "SI2004", "SX24", "SX114", "SX204", "SX294", "SX384"], 
    "FDAW0": ["SA1", "SA2", "SI1271", "SI1406", "SI2036", "SX56", "SX146", "SX236", "SX326", "SX416"]
    }



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
    Run_Layered_Occurrence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer)
    #Run_Layered_Subdistribution_Generation(Subdistribution_Layer)
    #Run_Layered_Subdistribution_Generation("voice", voice_set=["FCJF0", "MEDR0", "MCPM0", "FDAW0", "FDML0", "MDAC0", "FECD0", "MDPK0"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=True)
    #Run_Layered_Subdistribution_Generation("phoneme", phoneme_set=["ae", "ay", "ix", "iy"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=True)

def Run_Subtractive_Subdstributions():
    Run_Subtractive_Layered_Occurrence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer, 0.6, subtract_voice_for_phoneme=True)
    #Run_Subtractive_Layered_Subdistribution_Generation(Subdistribution_Layer)
    #Run_Subtractive_Layered_Subdistribution_Generation("voice", voice_set=["FCJF0", "MEDR0", "MCPM0", "FDAW0", "FDML0", "MDAC0", "FECD0", "MDPK0"])
    #Run_Subtractive_Layered_Subdistribution_Generation("phoneme", phoneme_set=["ae", "ay", "ix", "iy"])

def Run_Voice_Subdistribution_Deviation_Analysis():
    Run_Voice_Subdistribution_Deviation_Tracking("FCJF0", Layered_Subdistribution_Audio_Set, 0.975, 0.3)


#Run_Analysis()
#Run_Subdstributions()
#Run_Subtractive_Subdstributions()
Run_Voice_Subdistribution_Deviation_Analysis()