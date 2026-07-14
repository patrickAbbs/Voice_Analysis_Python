import bisect
import math
import numpy
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import (
    Analysis_Directory, Analysis_Run_Name, Chart_Image_Resolution,
    Spectrogram_Window_Jump_In_Seconds, Subdistribution_Timepoint_Voiced_Ratio_Minimum,
    Json_Directory
)
from Subdistribution_Extractor import Convert_Occurrence_Counts_To_Ratios
from Layered_Occurrence_Count_Populator import Process_Audio
from Layered_Subdistribution_Generator import Load_Layered_State, Get_Voiced_Frequency_Bucket_Centers
from Color_Assignment_Manager import Get_Speaker_Color
from Global_Helper_Functions import Convert_Half_Life_To_Cumulation_Weight, Weighted_Average


# --- internal helpers ---

def _Build_Sorted_Keys(inverted_occurrence_ratios):
    return [sorted(bucket.keys()) for bucket in inverted_occurrence_ratios]


def _Lookup_Closest_Value(bucket, sorted_keys, target):
    pos = bisect.bisect_left(sorted_keys, target)
    candidates = []
    if pos > 0:
        candidates.append(sorted_keys[pos - 1])
    if pos < len(sorted_keys):
        candidates.append(sorted_keys[pos])
    if not candidates:
        return 0.5
    closest_key = min(candidates, key=lambda k: abs(k - target))
    return bucket[closest_key]


def _Extract_Bell_Curve_Projections(inverted_occurrence_ratios):
    # For each bucket, find the distribution_ratio key whose inverted_occurrence_ratio
    # is closest to each of the three target values:
    #   0.5     = median (projected_bell_curve_center)
    #   0.84135 = 16th-percentile key (below center) → left std = center - that key
    #   0.15865 = 84th-percentile key (above center) → right std = that key - center
    # Stored as (center, projected_lower_standard_deviation, projected_upper_standard_deviation).
    projections = []
    for bucket in inverted_occurrence_ratios:
        center = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.5))[0]
        lower_percentile_key = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.15865))[0]
        upper_percentile_key = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.84135))[0]
        projected_lower_standard_deviation = center - lower_percentile_key
        projected_upper_standard_deviation = upper_percentile_key - center
        projections.append((center, projected_lower_standard_deviation, projected_upper_standard_deviation))
    return projections


_SQRT2 = math.sqrt(2.0)

def _Bell_Curve_Value2(timepoint_frequency_ratio, center, projected_lower_standard_deviation, projected_upper_standard_deviation):
    std = projected_lower_standard_deviation if timepoint_frequency_ratio < center else projected_upper_standard_deviation
    if std <= 0.0:
        return 0.5
    z_score = (timepoint_frequency_ratio - center) / std
    return 0.5 * (1.0 + math.erf(z_score / _SQRT2))


def _Freq_Colors(voiced_frequency_bucket_centers):
    n = len(voiced_frequency_bucket_centers)
    purple = numpy.array([0.502, 0.0, 0.502])
    orange = numpy.array([1.0, 0.647, 0.0])
    return [
        tuple(purple + (i / (n - 1) if n > 1 else 0.0) * (orange - purple))
        for i in range(n)
    ]


# --- chart generation ---

def Generate_Match_Score_Chart(
    voice_id, speaker_id, voiced_frequency_bucket_centers,
    match_scores, cumulative_comparative_occurrence_ratios,
    match_contribution_weights, hyperparameters
):
    positive_contribution_range = hyperparameters["positive_contribution_range"]
    lower_bound = 0.5 - (positive_contribution_range * 0.5)
    upper_bound = 0.5 + (positive_contribution_range * 0.5)

    n_points = len(match_scores)
    x_values = [i * Spectrogram_Window_Jump_In_Seconds for i in range(n_points)]
    total_time = n_points * Spectrogram_Window_Jump_In_Seconds
    colors = _Freq_Colors(voiced_frequency_bucket_centers)

    max_weight = max(
        (w for freq in voiced_frequency_bucket_centers for w in match_contribution_weights[freq]),
        default=1.0
    )
    subplot3_y_max = 1.0 + max_weight

    fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1, figsize=(20, 18))

    # Subplot 1: match_scores
    ax1.plot(x_values, match_scores, color=Get_Speaker_Color(speaker_id), linewidth=0.75)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, total_time)
    ax1.set_title(f"Match Score | {voice_id} baseline vs {speaker_id}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Match Score")

    # Subplot 2: cumulative_comparative_occurrence_ratios
    for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
        ax2.plot(x_values, cumulative_comparative_occurrence_ratios[freq_center],
                 color=colors[freq_index], linewidth=0.5)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0, total_time)
    ax2.set_title(f"Cumulative Comparative Occurrence Ratios | {voice_id} baseline vs {speaker_id}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Occurrence Ratio")

    # Subplot 3: transformed match_contribution_weights
    for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
        transformed = []
        for tp_i in range(n_points):
            ratio = cumulative_comparative_occurrence_ratios[freq_center][tp_i]
            weight = match_contribution_weights[freq_center][tp_i]
            if ratio < lower_bound:
                y = (-1.0 * weight) - 1.0
            elif ratio < 0.5:
                y = (1.0 - weight) * -1.0
            elif ratio <= upper_bound:
                y = 1.0 - weight
            else:
                y = weight + 1.0
            transformed.append(y)
        ax3.plot(x_values, transformed, color=colors[freq_index], linewidth=0.5)
    ax3.set_ylim(-subplot3_y_max, subplot3_y_max)
    ax3.set_xlim(0, total_time)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title(f"Match Contribution Weights (transformed) | {voice_id} baseline vs {speaker_id}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Weight (transformed)")

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_voice_match_score_progression_{voice_id}_{speaker_id}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Occurrence_Ratio_Divergence_Match_Score_Tracker: chart saved to '{output_path}'")


def Generate_Combined_Match_Score_Chart(voice_id, all_speaker_match_scores):
    fig, ax = pyplot.subplots(1, 1, figsize=(20, 6))
    max_time = 0.0

    for speaker_id, match_scores in all_speaker_match_scores.items():
        n_points = len(match_scores)
        x_values = [i * Spectrogram_Window_Jump_In_Seconds for i in range(n_points)]
        total_time = n_points * Spectrogram_Window_Jump_In_Seconds
        if total_time > max_time:
            max_time = total_time
        ax.plot(x_values, match_scores, color=Get_Speaker_Color(speaker_id), linewidth=0.75, label=speaker_id)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, max_time)
    ax.set_title(f"Match Scores — All Speakers vs {voice_id} Baseline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Match Score")
    ax.legend()

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_voice_match_scores_{voice_id}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Occurrence_Ratio_Divergence_Match_Score_Tracker: combined chart saved to '{output_path}'")


# --- entry point ---

def Run_Occurrence_Ratio_Divergence_Match_Score_Tracking(
    voice_id,
    comparative_voices_audio_set,
    hyperparameters,
    use_bell_curve_percentile_projection=False
):
    hyperparameters["occurrence_ratio_cumulation_weight"] = Convert_Half_Life_To_Cumulation_Weight(
        Spectrogram_Window_Jump_In_Seconds, hyperparameters["occurrence_ratio_cumulation_half_life"]
    )
    occurrence_ratio_cumulation_weight = hyperparameters["occurrence_ratio_cumulation_weight"]
    positive_contribution_range = hyperparameters["positive_contribution_range"]
    positive_weight_power_curve = hyperparameters["positive_weight_power_curve"]
    negative_weight_proximity_half_distance_increment = hyperparameters["negative_weight_proximity_half_distance_increment"]
    lower_bound = 0.5 - (positive_contribution_range * 0.5)
    upper_bound = 0.5 + (positive_contribution_range * 0.5)

    # Load and invert voice_id occurrence_ratios
    state_path = Json_Directory + f"Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts.json"
    state = Load_Layered_State(state_path)
    if state is None:
        print(f"Occurrence_Ratio_Divergence_Match_Score_Tracker: no data found for voice_id '{voice_id}', aborting")
        return

    inverted_occurrence_ratios = Convert_Occurrence_Counts_To_Ratios(
        state["frequency_amount_occurrence_counts"],
        state["total_voiced_frequency_timepoints_count"],
        invert=True
    )

    if use_bell_curve_percentile_projection:
        bell_curve_projections = _Extract_Bell_Curve_Projections(inverted_occurrence_ratios)
        inverted_occurrence_ratios = None
        sorted_keys_per_bucket = None
    else:
        bell_curve_projections = None
        sorted_keys_per_bucket = _Build_Sorted_Keys(inverted_occurrence_ratios)

    voiced_frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
    voiced_frequency_limit_index = len(voiced_frequency_bucket_centers)

    all_speaker_match_scores = {}

    for speaker_id, audio_list in comparative_voices_audio_set.items():
        cumulative_comparative_occurrence_ratios = {freq: [0.5] for freq in voiced_frequency_bucket_centers}
        match_contribution_weights = {freq: [0.0] for freq in voiced_frequency_bucket_centers}
        match_scores = [0.5]

        for audio_name in audio_list:
            print(f"Occurrence_Ratio_Divergence_Match_Score_Tracker: processing '{speaker_id}/{audio_name}'...")
            distribution, audio_frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)

            for timepoint_index in range(len(distribution[0])):
                voiced_frequency_range_distribution_ratio = numpy.sum(distribution[:voiced_frequency_limit_index, timepoint_index])
                if voiced_frequency_range_distribution_ratio < Subdistribution_Timepoint_Voiced_Ratio_Minimum:
                    continue
                if timepoint_phonemes[timepoint_index] is None:
                    continue

                for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                    timepoint_frequency_ratio = float(distribution[freq_index][timepoint_index])

                    if use_bell_curve_percentile_projection:
                        center, lower_sd, upper_sd = bell_curve_projections[freq_index]
                        value_2 = _Bell_Curve_Value2(timepoint_frequency_ratio, center, lower_sd, upper_sd)
                    else:
                        value_2 = _Lookup_Closest_Value(
                            inverted_occurrence_ratios[freq_index],
                            sorted_keys_per_bucket[freq_index],
                            timepoint_frequency_ratio
                        )

                    value_1 = cumulative_comparative_occurrence_ratios[freq_center][-1]
                    new_ratio = Weighted_Average(value_1, occurrence_ratio_cumulation_weight, value_2, 1.0 - occurrence_ratio_cumulation_weight)
                    cumulative_comparative_occurrence_ratios[freq_center].append(new_ratio)

                    # Below match_contribution_weight logic for positive contributions is power_curved proximity to 0.5,
                    # and logic for negative contributions is multiplicatively scaled power of 0.5 to produce
                    # [positive<->negative contribution crossing point proximity]. Consider experimenting with alternative
                    # calculation logic approaches, particularly for negative contribution (e.g. maybe power_curved rather
                    # than multiplicatively scaled, or maybe a different 'base function' than power of 0.5 to produce
                    # [positive<->negative contribution crossing point proximity]).
                    if lower_bound <= new_ratio <= upper_bound:
                        new_weight = (1.0 - (abs(new_ratio - 0.5) / (0.5 * positive_contribution_range))) ** positive_weight_power_curve
                    else:
                        crossing_point_proximity = (1.0 - new_ratio) if new_ratio > 0.5 else new_ratio
                        crossing_point_proximity_ratio = crossing_point_proximity / (0.5 * (1.0 - positive_contribution_range))
                        crossing_point_proximity_half_distance = math.log(crossing_point_proximity_ratio, 0.5)
                        new_weight = crossing_point_proximity_half_distance * negative_weight_proximity_half_distance_increment
                    match_contribution_weights[freq_center].append(new_weight)

                # Calculate match_score for this timepoint
                total_weight = sum(match_contribution_weights[freq][-1] for freq in voiced_frequency_bucket_centers)
                if total_weight == 0.0:
                    new_match_score = 0.5
                else:
                    new_match_score = sum(
                        (1.0 if lower_bound <= cumulative_comparative_occurrence_ratios[freq][-1] <= upper_bound else 0.0)
                        * match_contribution_weights[freq][-1]
                        for freq in voiced_frequency_bucket_centers
                    ) / total_weight
                match_scores.append(new_match_score)

        Generate_Match_Score_Chart(
            voice_id, speaker_id, voiced_frequency_bucket_centers,
            match_scores, cumulative_comparative_occurrence_ratios,
            match_contribution_weights, hyperparameters
        )
        all_speaker_match_scores[speaker_id] = match_scores

    Generate_Combined_Match_Score_Chart(voice_id, all_speaker_match_scores)
    print(f"Occurrence_Ratio_Divergence_Match_Score_Tracker: tracking complete for voice_id '{voice_id}'")
