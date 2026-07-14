import numpy
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import (
    Analysis_Directory, Analysis_Run_Name, Chart_Image_Resolution,
    Spectrogram_Window_Jump_In_Seconds, Subdistribution_Voiced_Frequency_Limit,
    Subdistribution_Timepoint_Voiced_Ratio_Minimum, Json_Directory
)
from Subdistribution_Extractor import Convert_Occurrence_Counts_To_Ratios, Subdistribution_Tier
from Layered_Occurrence_Count_Populator import Process_Audio
from Layered_Subdistribution_Generator import Load_Layered_State, Get_Voiced_Frequency_Bucket_Centers
from Global_Helper_Functions import Convert_Half_Life_To_Cumulation_Weight, Weighted_Average


# --- chart ---

def Generate_Deviation_Chart(voice_id, speaker_id, occurrence_ratio_threshold, voiced_frequency_bucket_centers, progressions):
    n_freqs = len(voiced_frequency_bucket_centers)
    sample_lists = next(iter(progressions.values()))
    n_points = len(sample_lists[0])
    x_values = [i * Spectrogram_Window_Jump_In_Seconds for i in range(n_points)]
    total_time = n_points * Spectrogram_Window_Jump_In_Seconds

    purple = numpy.array([0.502, 0.0, 0.502])
    orange = numpy.array([1.0, 0.647, 0.0])
    colors = [
        tuple(purple + (i / (n_freqs - 1) if n_freqs > 1 else 0.0) * (orange - purple))
        for i in range(n_freqs)
    ]

    fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(20, 12))

    for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
        first_list, second_list = progressions[freq_center]
        color = colors[freq_index]
        ax1.plot(x_values, first_list, color=color, linewidth=0.5)
        ax2.plot(x_values, second_list, color=color, linewidth=0.5)

    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, total_time)
    ax1.set_title(f"Cumulative Above-Baseline Ratio | {voice_id} baseline vs {speaker_id} | Threshold {occurrence_ratio_threshold}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Ratio")

    ax2.set_ylim(-1, 1)
    ax2.set_xlim(0, total_time)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title(f"Deviation from Threshold | {voice_id} baseline vs {speaker_id} | Threshold {occurrence_ratio_threshold}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Deviation")

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_voice_comparative_progression_{voice_id}_{speaker_id}_{occurrence_ratio_threshold}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Voice_Subdistribution_Deviation_Tracker: chart saved to '{output_path}'")


# --- entry point ---

def Run_Voice_Subdistribution_Deviation_Tracking(
    voice_id,
    comparative_voices_audio_set,
    occurrence_ratio_threshold,
    cumulation_half_life
):
    cumulation_weight = Convert_Half_Life_To_Cumulation_Weight(Spectrogram_Window_Jump_In_Seconds, cumulation_half_life)

    state_path = Json_Directory + f"Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts.json"
    state = Load_Layered_State(state_path)
    if state is None:
        print(f"Voice_Subdistribution_Deviation_Tracker: no data found for voice_id '{voice_id}', aborting")
        return

    frequency_amount_occurrence_ratios = Convert_Occurrence_Counts_To_Ratios(
        state["frequency_amount_occurrence_counts"],
        state["total_voiced_frequency_timepoints_count"]
    )
    thresholded_subdistribution = [
        max((ratio for ratio, occurrence in bucket.items() if occurrence >= occurrence_ratio_threshold), default=0.0)
        for bucket in frequency_amount_occurrence_ratios
    ]
    voice_subdistribution_tier = Subdistribution_Tier(occurrence_ratio_threshold, thresholded_subdistribution)

    voiced_frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
    voiced_frequency_limit_index = len(voiced_frequency_bucket_centers)

    for speaker_id, audio_list in comparative_voices_audio_set.items():
        cumulative_above_subdistribution_value_and_deviation_progressions = {
            freq: ([occurrence_ratio_threshold], [0.0])
            for freq in voiced_frequency_bucket_centers
        }

        for audio_name in audio_list:
            print(f"Voice_Subdistribution_Deviation_Tracker: processing '{speaker_id}/{audio_name}'...")
            distribution, audio_frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)

            for timepoint_index in range(len(distribution[0])):
                voiced_frequency_range_distribution_ratio = numpy.sum(distribution[:voiced_frequency_limit_index, timepoint_index])
                if voiced_frequency_range_distribution_ratio < Subdistribution_Timepoint_Voiced_Ratio_Minimum:
                    continue
                if timepoint_phonemes[timepoint_index] is None:
                    continue

                for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                    timepoint_frequency_ratio = distribution[freq_index][timepoint_index]
                    bucket_subdistribution_value = voice_subdistribution_tier.Subdistribution[freq_index]

                    first_list, second_list = cumulative_above_subdistribution_value_and_deviation_progressions[freq_center]

                    value_1 = first_list[-1]
                    weight_1 = cumulation_weight
                    value_2 = 1.0 if timepoint_frequency_ratio > bucket_subdistribution_value else 0.0
                    weight_2 = 1.0 - weight_1
                    new_first_value = Weighted_Average(value_1, weight_1, value_2, weight_2)
                    first_list.append(new_first_value)

                    if new_first_value > occurrence_ratio_threshold:
                        new_second_value = (new_first_value - occurrence_ratio_threshold) * (1.0 / (1.0 - occurrence_ratio_threshold))
                    else:
                        new_second_value = (occurrence_ratio_threshold - new_first_value) * (1.0 / occurrence_ratio_threshold) * -1.0
                    second_list.append(new_second_value)

        Generate_Deviation_Chart(
            voice_id, speaker_id, occurrence_ratio_threshold,
            voiced_frequency_bucket_centers,
            cumulative_above_subdistribution_value_and_deviation_progressions
        )

    print(f"Voice_Subdistribution_Deviation_Tracker: tracking complete for voice_id '{voice_id}'")
