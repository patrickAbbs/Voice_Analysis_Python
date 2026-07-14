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
from Element_Match_Contribution_Type_Explorer import Run_Element_Match_Contribution_Type_Exploration
from Occurrence_Ratio_Percentile_Shape_Visualizer import Visualize_Occurrence_Ratio_Percentile_Shapes

Layered_Subdistribution_Audio_Set = {
    "FCJF0": ["SA1", "SA2", "SI648", "SI1027", "SI1657", "SX37", "SX127", "SX217", "SX307", "SX397"], 
    "MEDR0": ["SA1", "SA2", "SI744", "SI1374", "SI2004", "SX24", "SX114", "SX204", "SX294", "SX384"], 
    "MCPM0": ["SA1", "SA2", "SI564", "SI1194", "SI1824", "SX24", "SX114", "SX204", "SX294", "SX384"], 
    "FDAW0": ["SA1", "SA2", "SI1271", "SI1406", "SI2036", "SX56", "SX146", "SX236", "SX326", "SX416"], 
    "FDML0": ["SA1", "SA2", "SI1149", "SI1779", "SI2075", "SX69", "SX159", "SX249", "SX339", "SX429"], 
    "MDAC0": ["SA1", "SA2", "SI631", "SI1261", "SI1837", "SX91", "SX181", "SX271", "SX361", "SX451"],
    "FECD0": ["SA1", "SA2", "SI788", "SI1418", "SI2048", "SX68", "SX158", "SX248", "SX338", "SX428"],
    "MDPK0": ["SA1", "SA2", "SI552", "SI1053", "SI1683", "SX63", "SX153", "SX243", "SX333", "SX423"],
    #"FJSP0": ["SA1", "SA2", "SI804", "SI1434", "SI1763", "SX84", "SX174", "SX264"],
    #"MGRL0": ["SA1", "SA2", "SI867", "SI1497", "SI2127", "SX57", "SX147", "SX237"]
    "FJSP0": ["SA1", "SA2", "SI804", "SI1434", "SI1763", "SX84", "SX174", "SX264", "SX354", "SX444"],
    "MGRL0": ["SA1", "SA2", "SI867", "SI1497", "SI2127", "SX57", "SX147", "SX237", "SX327", "SX417"]
    }

New_Partial_Speaker_Audio_Set = {
    "FJSP0": ["SA1", "SA2", "SI804", "SI1434", "SI1763", "SX84", "SX174", "SX264"],
    "MGRL0": ["SA1", "SA2", "SI867", "SI1497", "SI2127", "SX57", "SX147", "SX237"]
}


#Layered_Subdistribution_Audio_Set = {"FCJF0": ["SA1", "SI648"], "MEDR0": ["SA1", "SI1374"]}
Subdistribution_Layer = "voice"



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
    #Run_Layered_Occurrence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer)
    #Run_Layered_Occurrence_Count_Population(New_Partial_Speaker_Audio_Set, Subdistribution_Layer)
    Run_Layered_Occurrence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer, frequency_ratio_cumulation_half_life = 0.2)
    #Run_Layered_Subdistribution_Generation(Subdistribution_Layer)
    #Run_Layered_Subdistribution_Generation("voice", voice_set=["FCJF0", "MEDR0", "MCPM0", "FDAW0", "FDML0", "MDAC0", "FECD0", "MDPK0"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=False)
    #Run_Layered_Subdistribution_Generation("phoneme", phoneme_set=["ae", "ay", "ix", "iy"], allow_negative_subtractive_subdistributions=True, generate_original_subdistribution_charts=True, generate_subtractive_subdistribution_charts=True)

def Run_Subtractive_Subdstributions():
    #Run_Subtractive_Layered_Occurrence_Count_Population(Layered_Subdistribution_Audio_Set, Subdistribution_Layer, 0.6, subtract_voice_for_phoneme=True)
    #Run_Subtractive_Layered_Subdistribution_Generation(Subdistribution_Layer)
    Run_Subtractive_Layered_Subdistribution_Generation("voice", voice_set=["FCJF0", "MEDR0", "MCPM0", "FDAW0", "FDML0", "MDAC0", "FECD0", "MDPK0"])
    #Run_Subtractive_Layered_Subdistribution_Generation("phoneme", phoneme_set=["ae", "ay", "ix", "iy"])

def Run_Voice_Subdistribution_Deviation_Analysis():
    Run_Voice_Subdistribution_Deviation_Tracking("FECD0", Layered_Subdistribution_Audio_Set, 0.9875, 0.3)

def Run_Element_Match_Contribution_Type_Analysis():
    Run_Element_Match_Contribution_Type_Exploration("FCJF0", Layered_Subdistribution_Audio_Set,
        aggregate_match_types={
            "weighted_binary_match_contribution": {
                "include_variant": True,
                "hyperparameters": {
                    "positive_contribution_range": 0.5,
                    "positive_weight_power_curve": 0.5,
                    "negative_weight_proximity_half_distance_increment": 1.0
                }
            },
            "occurrence_percentile_deviation": {
                "include_variant": True,
                "hyperparameters": {}
            },
            "occurrence_percentile_inverse_deviation": {
                "include_variant": True,
                "hyperparameters": {
                    "deviation_power_curve": 1.0,
                    "inverse_deviation_minimum": -100.0
                }
            },
            "occurrence_percentile_half_distance": {
                "include_variant": True,
                "hyperparameters": {
                    "half_distance_minimum": -10.0
                }
            },
            "raw_distance": {
                "include_variant": True,
                "hyperparameters": {}
            }
        },
        cross_type_hyperparameters={
            "use_bell_curve_percentile_projection": True,
            "occurrence_ratio_cumulation_half_life": 0.2,
            "voice_profile_cumulation_half_life": 0.2
        }
    )

def Run_Visualize_Occurrence_Ratio_Percentile_Shapes():
    #Visualize_Occurrence_Ratio_Percentile_Shapes(["FCJF0", "MEDR0"], proximity_density_distance=0.001)
    Visualize_Occurrence_Ratio_Percentile_Shapes(["FCJF0", "MEDR0", "MCPM0", "FDAW0", "FDML0", "MDAC0", "FECD0", "MDPK0"], 
                                                 proximity_density_distance=0.0008, voice_profile_cumulation_half_life=0.2)

#Run_Analysis()
#Run_Subdstributions()
#Run_Subtractive_Subdstributions()
#Run_Voice_Subdistribution_Deviation_Analysis()
Run_Element_Match_Contribution_Type_Analysis()
#Run_Visualize_Occurrence_Ratio_Percentile_Shapes()