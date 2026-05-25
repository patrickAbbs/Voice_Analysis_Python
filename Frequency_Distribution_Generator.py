import numpy
import librosa
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import Chart_Image_Resolution, Audio_File_Set, Analysis_Directory, Analysis_Run_Name, Spectrogram_Window_Size_In_Seconds, Spectrogram_Window_Jump_In_Seconds, Frequency_Distribution_Bucket_Increment, Frequency_Distribution_Bucket_Range, Frequency_Distribution_Frequency_Maximum, Frequency_Distribution_Displayed_Frequency_Maximum, Frequency_Distribution_Display_Type


#TODO [2026/05/25]: the "decibel" frequency distribution chart visualization looks incorrect | broken, debug why

def Generate_Frequency_Centers(spectrogram_frequencies):
    frequency_centers = []
    current_frequency_center = Frequency_Distribution_Bucket_Increment
    while (current_frequency_center < Frequency_Distribution_Frequency_Maximum) and (current_frequency_center < spectrogram_frequencies[-1]):
        frequency_centers.append(current_frequency_center)
        current_frequency_center += Frequency_Distribution_Bucket_Increment
    return frequency_centers


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


def Generate_Typed_Bucketed_Frequency_Progressions(linear_spectrogram_data, spectrogram_frequencies, frequency_centers):
    typed_bucketed_frequency_progressions = {}

    typed_bucketed_frequency_progressions["linear"] = []
    for frequency_center in frequency_centers:
        typed_bucketed_frequency_progressions["linear"].append(Generate_Frequency_Center_Linear_Bucket_Progression(frequency_center, linear_spectrogram_data, spectrogram_frequencies))
    typed_bucketed_frequency_progressions["linear"] = numpy.array(typed_bucketed_frequency_progressions["linear"])

    typed_bucketed_frequency_progressions["logarithmic"] = typed_bucketed_frequency_progressions["linear"] ** 0.30102999566
    typed_bucketed_frequency_progressions["decibel"] = librosa.amplitude_to_db(typed_bucketed_frequency_progressions["linear"], ref=numpy.max)
    typed_bucketed_frequency_progressions["decibel"] += 80

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
    typed_frequency_distributions = {}
    typed_frequency_distributions["linear"] = Generate_Bucketed_Frequency_Distribution(typed_bucketed_frequency_progressions["linear"])
    typed_frequency_distributions["logarithmic"] = Generate_Bucketed_Frequency_Distribution(typed_bucketed_frequency_progressions["logarithmic"])
    typed_frequency_distributions["decibel"] = Generate_Bucketed_Frequency_Distribution(typed_bucketed_frequency_progressions["decibel"])
    return typed_frequency_distributions


def Generate_And_Save_Frequency_Distribution_Chart(audios_analysis_data):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(Audio_File_Set) * 100) + 11
    for audio_file_name in Audio_File_Set:
        spectrogram_window_size_in_audio_samples = round(audios_analysis_data[audio_file_name]["sample_rate"] * Spectrogram_Window_Size_In_Seconds)
        spectrogram_jump_length_in_audio_samples = round(audios_analysis_data[audio_file_name]["sample_rate"] * Spectrogram_Window_Jump_In_Seconds)

        pyplot.subplot(subplot_number)
        pyplot.title(audio_file_name)
        pyplot.ylim(0, Frequency_Distribution_Displayed_Frequency_Maximum)
        librosa.display.specshow(audios_analysis_data[audio_file_name]["typed_bucketed_frequency_distributions"][Frequency_Distribution_Display_Type],
                                 win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples,
                                 x_axis='time', y_axis='hz', sr=audios_analysis_data[audio_file_name]["sample_rate"])
        subplot_number += 1
    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "_frequency_distribution.png", dpi=Chart_Image_Resolution)
    pyplot.close()


def Generate_Bucketed_Frequency_Distribution_Set(audios_analysis_data):
    for audio_file_name in Audio_File_Set:
        frequency_centers = audios_analysis_data[audio_file_name]["spectrogram_frequencies"]
        audios_analysis_data[audio_file_name]["typed_bucketed_frequency_progressions"] = Generate_Typed_Bucketed_Frequency_Progressions(audios_analysis_data[audio_file_name]["typed_spectrogram_data"]["linear"], audios_analysis_data[audio_file_name]["spectrogram_frequencies"], frequency_centers)

        audios_analysis_data[audio_file_name]["typed_bucketed_frequency_distributions"] = Generate_Typed_Bucketed_Frequency_Distributions(audios_analysis_data[audio_file_name]["typed_bucketed_frequency_progressions"])

    Generate_And_Save_Frequency_Distribution_Chart(audios_analysis_data)

    return audios_analysis_data
