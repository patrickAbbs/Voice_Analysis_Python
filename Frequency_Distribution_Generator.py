import math
import numpy
import librosa
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import Chart_Image_Resolution, Analysis_Directory, Analysis_Run_Name, Spectrogram_Window_Size_In_Seconds, Spectrogram_Window_Jump_In_Seconds, Frequency_Distribution_Bucket_Increment, Frequency_Distribution_Bucket_Range, Frequency_Distribution_Frequency_Maximum, Frequency_Distribution_Displayed_Frequency_Maximum, Frequency_Distribution_Display_Type, Distribution_Types


#TODO [2026/05/25]: the "decibel" frequency distribution chart visualization looks incorrect | broken, debug why

class Typed_Bucketed_Frequency_Progressions:
    def __init__(self):
        self.Typed_Progressions = {}

class Typed_Bucketed_Frequency_Distributions:
    def __init__(self):
        self.Typed_Distributions = {}


def Generate_Frequency_Bucket_Centers(spectrogram_frequencies):
    frequency_bucket_centers = []
    current_frequency_center = Frequency_Distribution_Bucket_Increment
    while (current_frequency_center < Frequency_Distribution_Frequency_Maximum) and (current_frequency_center < spectrogram_frequencies[-1]):
        frequency_bucket_centers.append(current_frequency_center)
        current_frequency_center += Frequency_Distribution_Bucket_Increment
    return frequency_bucket_centers


#NOTE [2026/05/25]: below logic uses "windowed average" approach to bucket population, and does not currently have an option for "center-proximity-weighted average" approach as posited as a potential alternative in (1.3) of first 2026/05/25 pen-and-paper entry list. Maybe consider whether that would warrant adding in as a hyperparameter-selected alternative.
def Generate_Frequency_Center_Linear_Bucket_Progression(frequency_center, linear_spectrogram_data, spectrogram_frequencies):
    frequency_center_linear_bucket_progression = []
    frequency_bucket_minimum = frequency_center - (Frequency_Distribution_Bucket_Range / 2.0)
    frequency_bucket_maximum = frequency_center + (Frequency_Distribution_Bucket_Range / 2.0)
    minimum_spectrogram_frequency_index = next((index for index, frequency in enumerate(spectrogram_frequencies) if frequency >= frequency_bucket_minimum))
    maximum_spectrogram_frequency_index = minimum_spectrogram_frequency_index
    bucket_denominator = 1.0
    while (maximum_spectrogram_frequency_index < (len(spectrogram_frequencies) - 1)) and (spectrogram_frequencies[maximum_spectrogram_frequency_index] < frequency_bucket_maximum):
        maximum_spectrogram_frequency_index += 1
        bucket_denominator += 1.0
    for timepoint_index in range(len(linear_spectrogram_data[0])):
        timepoint_bucket_sum = 0.0
        for bucket_frequency_index in range(minimum_spectrogram_frequency_index, (maximum_spectrogram_frequency_index + 1)):
            timepoint_bucket_sum += linear_spectrogram_data[bucket_frequency_index][timepoint_index]
        timepoint_bucket_average = timepoint_bucket_sum / bucket_denominator
        frequency_center_linear_bucket_progression.append(timepoint_bucket_average)
    return frequency_center_linear_bucket_progression


def Generate_Typed_Bucketed_Frequency_Progressions(linear_spectrogram_data, spectrogram_frequencies, frequency_bucket_centers):
    typed_bucketed_frequency_progressions = Typed_Bucketed_Frequency_Progressions()

    typed_bucketed_frequency_progressions.Typed_Progressions["linear"] = []
    for frequency_bucket_center in frequency_bucket_centers:
        typed_bucketed_frequency_progressions.Typed_Progressions["linear"].append(Generate_Frequency_Center_Linear_Bucket_Progression(frequency_bucket_center, linear_spectrogram_data, spectrogram_frequencies))
    typed_bucketed_frequency_progressions.Typed_Progressions["linear"] = numpy.array(typed_bucketed_frequency_progressions.Typed_Progressions["linear"])

    typed_bucketed_frequency_progressions.Typed_Progressions["logarithmic"] = typed_bucketed_frequency_progressions.Typed_Progressions["linear"] ** 0.30102999566
    typed_bucketed_frequency_progressions.Typed_Progressions["decibel"] = librosa.amplitude_to_db(typed_bucketed_frequency_progressions.Typed_Progressions["linear"], ref=numpy.max)
    typed_bucketed_frequency_progressions.Typed_Progressions["decibel"] += 80.0

    return typed_bucketed_frequency_progressions


def Generate_Bucketed_Frequency_Distribution(bucketed_frequencies_progression):
    bucketed_progression_sums = numpy.sum(bucketed_frequencies_progression, axis=0)
    bucketed_frequency_distribution = []
    for frequency_center_progression in bucketed_frequencies_progression:
        frequency_center_distribution_ratio_progression = []
        for timepoint_index in range(len(frequency_center_progression)):
            frequency_center_distribution_ratio = (frequency_center_progression[timepoint_index] / bucketed_progression_sums[timepoint_index]) if (bucketed_progression_sums[timepoint_index] > 0.0000001) else 0.0
            frequency_center_distribution_ratio_progression.append(frequency_center_distribution_ratio)
        bucketed_frequency_distribution.append(frequency_center_distribution_ratio_progression)
    bucketed_frequency_distribution = numpy.array(bucketed_frequency_distribution)
    return bucketed_frequency_distribution


def Generate_Typed_Bucketed_Frequency_Distributions(typed_bucketed_frequency_progressions):
    typed_bucketed_frequency_distributions = Typed_Bucketed_Frequency_Distributions()
    for distribution_type in Distribution_Types:
        typed_bucketed_frequency_distributions.Typed_Distributions[distribution_type] = Generate_Bucketed_Frequency_Distribution(typed_bucketed_frequency_progressions.Typed_Progressions[distribution_type])
    return typed_bucketed_frequency_distributions


def Generate_Frequency_Distribution_Chart(all_audios_analysis_data):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(all_audios_analysis_data) * 100) + 11
    for audio_analysis_entry in all_audios_analysis_data:
        spectrogram_window_size_in_audio_samples = round(audio_analysis_entry.Spectrogram_Data.Sample_Rate * Spectrogram_Window_Size_In_Seconds)
        spectrogram_jump_length_in_audio_samples = round(audio_analysis_entry.Spectrogram_Data.Sample_Rate * Spectrogram_Window_Jump_In_Seconds)
        spectrogram_fft_data_points = 2 * spectrogram_window_size_in_audio_samples  # Number of fft points; Kept 2x to increase the frequency resolution

        distribution_to_spectrogram_frequency_bucket_multiplier = audio_analysis_entry.Spectrogram_Data.Frequencies[1] / Frequency_Distribution_Bucket_Increment
        bucketed_frequency_distribution = audio_analysis_entry.Typed_Bucketed_Frequency_Distributions.Typed_Distributions[Frequency_Distribution_Display_Type]
        visualizer_compatibility_mapped_bucketed_frequency_distribution = []
        for spectrogram_frequency_index in range(len(audio_analysis_entry.Spectrogram_Data.Frequencies)):
            spectrogram_frequency_progression = [0 for index in range(len(bucketed_frequency_distribution[0]))]
            if audio_analysis_entry.Spectrogram_Data.Frequencies[spectrogram_frequency_index] < Frequency_Distribution_Displayed_Frequency_Maximum:
                frequency_distribution_bucket_index = round(spectrogram_frequency_index * distribution_to_spectrogram_frequency_bucket_multiplier)
                spectrogram_frequency_progression = bucketed_frequency_distribution[frequency_distribution_bucket_index]
            visualizer_compatibility_mapped_bucketed_frequency_distribution.append(spectrogram_frequency_progression)
        visualizer_compatibility_mapped_bucketed_frequency_distribution = numpy.array(visualizer_compatibility_mapped_bucketed_frequency_distribution)

        pyplot.subplot(subplot_number)
        pyplot.title(audio_analysis_entry.Audio_File_Name)
        pyplot.ylim(0, Frequency_Distribution_Displayed_Frequency_Maximum)
        librosa.display.specshow(visualizer_compatibility_mapped_bucketed_frequency_distribution,
                                 win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples,
                                 x_axis='time', y_axis='hz', sr=audio_analysis_entry.Spectrogram_Data.Sample_Rate)
        subplot_number += 1
    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "_frequency_distribution.png", dpi=Chart_Image_Resolution)
    pyplot.close()


def Generate_Bucketed_Frequency_Distribution_Set(all_audios_analysis_data):
    for audio_analysis_entry in all_audios_analysis_data:
        audio_analysis_entry.Frequency_Bucket_Centers = Generate_Frequency_Bucket_Centers(audio_analysis_entry.Spectrogram_Data.Frequencies)

        audio_analysis_entry.Typed_Bucketed_Frequency_Progressions = Generate_Typed_Bucketed_Frequency_Progressions(audio_analysis_entry.Spectrogram_Data.Spectrogram, audio_analysis_entry.Spectrogram_Data.Frequencies, audio_analysis_entry.Frequency_Bucket_Centers)

        audio_analysis_entry.Typed_Bucketed_Frequency_Distributions = Generate_Typed_Bucketed_Frequency_Distributions(audio_analysis_entry.Typed_Bucketed_Frequency_Progressions)

    Generate_Frequency_Distribution_Chart(all_audios_analysis_data)

    return all_audios_analysis_data
