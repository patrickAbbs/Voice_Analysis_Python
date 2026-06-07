import numpy
import matplotlib.pyplot as pyplot


from Global_Hyperparameters import Analysis_Directory, Analysis_Run_Name, Subdistribution_Thresholds, Subdistribution_Voiced_Frequency_Limit, Subdistribution_Timepoint_Voiced_Ratio_Minimum, Subdistribution_Display_Type, Distribution_Types, Chart_Image_Resolution, Subdistribution_Display_Colors

class Typed_Tiered_Subdistributions:
    def __init__(self):
        self.Typed_Subdistributions = {}

class Subdistribution_Tier:
    def __init__(self, occurrence_ratio_threshold, subdistribution):
        self.Occurrence_Ratio_Threshold = occurrence_ratio_threshold
        self.Subdistribution = subdistribution
        self.Subdistribution_Sum = sum(subdistribution)


def Accumulate_Frequency_Occurrence_Counts(bucketed_frequency_distribution_progression, frequency_bucket_centers, existing_counts=None, existing_voiced_timepoints_count=0.0, timepoint_mask=None, frequency_ratio_offsets=None):
    voiced_frequency_limit_index = next((center_index for center_index, center_frequency in enumerate(frequency_bucket_centers) if center_frequency > Subdistribution_Voiced_Frequency_Limit), None)
    frequency_amount_occurrence_counts = existing_counts if existing_counts is not None else [{} for _ in range(voiced_frequency_limit_index)]
    voiced_frequency_timepoints_count = existing_voiced_timepoints_count
    for timepoint_index in range(len(bucketed_frequency_distribution_progression[0])):
        voiced_frequency_range_distribution_ratio = numpy.sum(bucketed_frequency_distribution_progression[:voiced_frequency_limit_index, timepoint_index])
        if voiced_frequency_range_distribution_ratio < Subdistribution_Timepoint_Voiced_Ratio_Minimum:
            continue
        if timepoint_mask is not None and not timepoint_mask[timepoint_index]:
            continue
        voiced_frequency_timepoints_count += 1.0
        for voiced_frequency_index in range(voiced_frequency_limit_index):
            timepoint_frequency_ratio = bucketed_frequency_distribution_progression[voiced_frequency_index][timepoint_index]
            if frequency_ratio_offsets is not None:
                timepoint_frequency_ratio -= frequency_ratio_offsets[voiced_frequency_index]
            if timepoint_frequency_ratio not in frequency_amount_occurrence_counts[voiced_frequency_index]:
                nearest_encompassing_frequency_ratio = min((other_frequency_ratio for other_frequency_ratio in frequency_amount_occurrence_counts[voiced_frequency_index] if other_frequency_ratio > timepoint_frequency_ratio), default=None)
                if nearest_encompassing_frequency_ratio is not None:
                    frequency_amount_occurrence_counts[voiced_frequency_index][timepoint_frequency_ratio] = frequency_amount_occurrence_counts[voiced_frequency_index][nearest_encompassing_frequency_ratio]
                else:
                    frequency_amount_occurrence_counts[voiced_frequency_index][timepoint_frequency_ratio] = 0
            for frequency_amount_occurrence_key in frequency_amount_occurrence_counts[voiced_frequency_index]:
                if frequency_amount_occurrence_key <= timepoint_frequency_ratio:
                    frequency_amount_occurrence_counts[voiced_frequency_index][frequency_amount_occurrence_key] += 1
    return frequency_amount_occurrence_counts, voiced_frequency_timepoints_count


def Convert_Occurrence_Counts_To_Ratios(frequency_amount_occurrence_counts, voiced_frequency_timepoints_count):
    frequency_amount_occurrence_ratios = [{} for _ in range(len(frequency_amount_occurrence_counts))]
    for frequency_bucket_index in range(len(frequency_amount_occurrence_counts)):
        for distribution_ratio, occurrence_count in frequency_amount_occurrence_counts[frequency_bucket_index].items():
            frequency_amount_occurrence_ratios[frequency_bucket_index][distribution_ratio] = occurrence_count / voiced_frequency_timepoints_count
    return frequency_amount_occurrence_ratios


def Extract_Frequency_Amount_Occurrence_Ratios(bucketed_frequency_distribution_progression, frequency_bucket_centers):
    frequency_amount_occurrence_counts, voiced_frequency_timepoints_count = Accumulate_Frequency_Occurrence_Counts(bucketed_frequency_distribution_progression, frequency_bucket_centers)
    return Convert_Occurrence_Counts_To_Ratios(frequency_amount_occurrence_counts, voiced_frequency_timepoints_count)


def Extract_Frequency_Subdistributions(frequency_amount_occurrence_ratios):
    frequency_subdistributions = []
    for subdistribution_threshold in Subdistribution_Thresholds:
        thresholded_subdistribution = []
        for frequency_amount_occurrence_ratio_entry in frequency_amount_occurrence_ratios:
            highest_above_threshold_frequency_ratio = max((distribution_ratio for distribution_ratio, occurrence_ratio in frequency_amount_occurrence_ratio_entry.items() if occurrence_ratio >= subdistribution_threshold), default=0.0)
            thresholded_subdistribution.append(highest_above_threshold_frequency_ratio)
        subdistribution_tier = Subdistribution_Tier(subdistribution_threshold, thresholded_subdistribution)
        frequency_subdistributions.append(subdistribution_tier)
    return frequency_subdistributions


def Generate_Tiered_Subdistribution_Charts(all_audios_analysis_data):
    y_axis_maximum = 0.0
    for audio_entry in all_audios_analysis_data:
        for type_subdistributions in audio_entry.Typed_Tiered_Subdistributions.Typed_Subdistributions.values():
            for subdistribution_tier in type_subdistributions:
                highest_subdistribution_value = max(subdistribution_tier.Subdistribution)
                if highest_subdistribution_value > y_axis_maximum:
                    y_axis_maximum = highest_subdistribution_value

    for audio_analysis_entry in all_audios_analysis_data:
        voiced_frequency_limit_index = next((center_index for center_index, center_frequency in enumerate(audio_analysis_entry.Frequency_Bucket_Centers) if center_frequency > Subdistribution_Voiced_Frequency_Limit), None)
        subdistribution_included_voiced_frequency_bucket_centers = audio_analysis_entry.Frequency_Bucket_Centers[:voiced_frequency_limit_index]

        pyplot.figure(figsize=(20, 24))
        audio_entry_color = Subdistribution_Display_Colors[all_audios_analysis_data.index(audio_analysis_entry)]
        subplot_number = (len(audio_analysis_entry.Typed_Tiered_Subdistributions.Typed_Subdistributions[Subdistribution_Display_Type]) * 100) + 11
        for subdistribution_tier in audio_analysis_entry.Typed_Tiered_Subdistributions.Typed_Subdistributions[Subdistribution_Display_Type]:
            pyplot.subplot(subplot_number)
            pyplot.title(f"Name {audio_analysis_entry.Audio_File_Name} | Threshold {subdistribution_tier.Occurrence_Ratio_Threshold} | Sum {subdistribution_tier.Subdistribution_Sum}")
            pyplot.ylim(0, y_axis_maximum)
            pyplot.bar(subdistribution_included_voiced_frequency_bucket_centers, subdistribution_tier.Subdistribution, color=audio_entry_color, width=audio_analysis_entry.Frequency_Bucket_Centers[0])
            subplot_number += 1
        pyplot.tight_layout()
        pyplot.savefig(Analysis_Directory + Analysis_Run_Name + f"self_subdistributions_{audio_analysis_entry.Audio_File_Name}.png", dpi=Chart_Image_Resolution)
        pyplot.close()

    for subdistribution_tier_index in range(len(Subdistribution_Thresholds)):
        pyplot.figure(figsize=(20, 24))
        subplot_number = (len(all_audios_analysis_data) * 100) + 11
        for audio_analysis_entry in all_audios_analysis_data:
            voiced_frequency_limit_index = next((center_index for center_index, center_frequency in enumerate(audio_analysis_entry.Frequency_Bucket_Centers) if center_frequency > Subdistribution_Voiced_Frequency_Limit), None)
            subdistribution_included_voiced_frequency_bucket_centers = audio_analysis_entry.Frequency_Bucket_Centers[:voiced_frequency_limit_index]

            audio_entry_color = Subdistribution_Display_Colors[all_audios_analysis_data.index(audio_analysis_entry)]
            audio_subdistribution_tier = audio_analysis_entry.Typed_Tiered_Subdistributions.Typed_Subdistributions[Subdistribution_Display_Type][subdistribution_tier_index]
            pyplot.subplot(subplot_number)
            pyplot.title(f"Name {audio_analysis_entry.Audio_File_Name} | Threshold {audio_subdistribution_tier.Occurrence_Ratio_Threshold} | Sum {audio_subdistribution_tier.Subdistribution_Sum}")
            pyplot.ylim(0, y_axis_maximum)
            pyplot.bar(subdistribution_included_voiced_frequency_bucket_centers, audio_subdistribution_tier.Subdistribution, color=audio_entry_color, width=audio_analysis_entry.Frequency_Bucket_Centers[0])
            subplot_number += 1
        pyplot.tight_layout()
        pyplot.savefig(Analysis_Directory + Analysis_Run_Name + f"cross_subdistributions_{Subdistribution_Thresholds[subdistribution_tier_index]}.png", dpi=Chart_Image_Resolution)
        pyplot.close()


def Extract_Frequency_Subdistribution_Sets(all_audios_analysis_data):
    for audio_analysis_entry in all_audios_analysis_data:
        audio_analysis_entry.Typed_Tiered_Subdistributions = Typed_Tiered_Subdistributions()
        for distribution_type in Distribution_Types:
            frequency_amount_occurrence_ratios = Extract_Frequency_Amount_Occurrence_Ratios(audio_analysis_entry.Typed_Bucketed_Frequency_Distributions.Typed_Distributions[distribution_type], audio_analysis_entry.Frequency_Bucket_Centers)
            audio_analysis_entry.Typed_Tiered_Subdistributions.Typed_Subdistributions[distribution_type] = Extract_Frequency_Subdistributions(frequency_amount_occurrence_ratios)
        print(f"Extract_Frequency_Subdistributions() complete for audio_file '{audio_analysis_entry.Audio_File_Name}'")
    Generate_Tiered_Subdistribution_Charts(all_audios_analysis_data)
    return all_audios_analysis_data
