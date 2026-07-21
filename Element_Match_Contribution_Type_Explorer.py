import bisect
import math
import os
import numpy
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import (
    Analysis_Directory, Analysis_Run_Name, Chart_Image_Resolution,
    Spectrogram_Window_Jump_In_Seconds, Subdistribution_Timepoint_Voiced_Ratio_Minimum,
    Json_Directory
)
from Subdistribution_Extractor import Convert_Occurrence_Counts_To_Ratios
from Layered_Occurrence_Count_Populator import Process_Audio, Format_Half_Life_For_Filename
from Layered_Subdistribution_Generator import Load_Layered_State, Get_Voiced_Frequency_Bucket_Centers
from Color_Assignment_Manager import Get_Speaker_Color
from Global_Helper_Functions import Convert_Half_Life_To_Cumulation_Weight, Weighted_Average


_SQRT2 = math.sqrt(2.0)
_LOG_0_5 = math.log(0.5)

VARIANT_ORDER = [
    "weighted_binary_match_contribution",
    "occurrence_percentile_deviation",
    "occurrence_percentile_inverse_deviation",
    "occurrence_percentile_half_distance",
    "raw_distance",
    "accumulative_deviation",
]

_OVERALL_KEYS = {
    "weighted_binary_match_contribution": "weighted_binary_match_contributions",
    "occurrence_percentile_deviation": "average_occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "average_occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "average_occurrence_percentile_half_distances",
    "raw_distance": "average_raw_distances",
    "accumulative_deviation": "average_element_accumulative_deviations",
}

_PER_BUCKET_KEYS = {
    "weighted_binary_match_contribution": "match_contribution_weights",
    "occurrence_percentile_deviation": "occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "occurrence_percentile_half_distances",
    "raw_distance": "cumulative_raw_distances",
    "accumulative_deviation": "element_accumulative_deviations",
}


# --- internal helpers ---

def _Build_Sorted_Keys(inverted_occurrence_ratios):
    return [sorted(bucket.keys()) for bucket in inverted_occurrence_ratios]


def _Lookup_Closest_Value(bucket, sorted_keys, target):
    position = bisect.bisect_left(sorted_keys, target)
    candidates = []
    if position > 0:
        candidates.append(sorted_keys[position - 1])
    if position < len(sorted_keys):
        candidates.append(sorted_keys[position])
    if not candidates:
        return 0.5
    closest_key = min(candidates, key=lambda key: abs(key - target))
    return bucket[closest_key]


def _Extract_Bell_Curve_Projections(inverted_occurrence_ratios):
    projections = []
    for bucket in inverted_occurrence_ratios:
        center = min(bucket.items(), key=lambda key_value: abs(key_value[1] - 0.5))[0]
        lower_key = min(bucket.items(), key=lambda key_value: abs(key_value[1] - 0.15865))[0]
        upper_key = min(bucket.items(), key=lambda key_value: abs(key_value[1] - 0.84135))[0]
        projections.append((center, center - lower_key, upper_key - center))
    return projections


def _Bell_Curve_Value_2(ratio, center, lower_standard_deviation, upper_standard_deviation):
    standard_deviation = lower_standard_deviation if ratio < center else upper_standard_deviation
    if standard_deviation <= 0.0:
        return 0.5
    return 0.5 * (1.0 + math.erf((ratio - center) / (standard_deviation * _SQRT2)))


def _Extract_Medians(inverted_occurrence_ratios):
    return [min(bucket.items(), key=lambda key_value: abs(key_value[1] - 0.5))[0] for bucket in inverted_occurrence_ratios]


def _Frequency_Colors(voiced_frequency_bucket_centers):
    bucket_count = len(voiced_frequency_bucket_centers)
    purple = numpy.array([0.502, 0.0, 0.502])
    orange = numpy.array([1.0, 0.647, 0.0])
    return [
        tuple(purple + (index / (bucket_count - 1) if bucket_count > 1 else 0.0) * (orange - purple))
        for index in range(bucket_count)
    ]


def _Signed_Weighted_Binary_Match_Contribution_Weight(weight, ratio, lower_bound, upper_bound):
    return weight if lower_bound <= ratio <= upper_bound else -weight


# --- per-bucket variant computations ---

def _Weighted_Binary_Match_Contribution_Weight(new_ratio, lower_bound, upper_bound, positive_contribution_range, positive_weight_power_curve, negative_weight_proximity_half_distance_increment):
    if lower_bound <= new_ratio <= upper_bound:
        return (1.0 - (abs(new_ratio - 0.5) / (0.5 * positive_contribution_range))) ** positive_weight_power_curve
    crossing_point_proximity = (1.0 - new_ratio) if new_ratio > 0.5 else new_ratio
    crossing_point_proximity_ratio = crossing_point_proximity / (0.5 * (1.0 - positive_contribution_range))
    return math.log(crossing_point_proximity_ratio, 0.5) * negative_weight_proximity_half_distance_increment


def _Occurrence_Percentile_Deviation(new_ratio):
    return 1.0 - (abs(0.5 - new_ratio) * 2.0)


def _Occurrence_Percentile_Inverse_Deviation(deviation, power_curve, minimum):
    if deviation <= 0.0:
        return minimum
    return max((-1.0 / (deviation ** power_curve)) + 1.0, minimum)


def _Occurrence_Percentile_Half_Distance(deviation, minimum):
    if deviation <= 0.0:
        return minimum
    return max(-1.0 * (math.log(deviation) / _LOG_0_5), minimum)


# --- global bounds computation ---

def _Compute_Global_Ylims(included_variants, all_results, voiced_frequency_bucket_centers, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound, accumulative_deviation_hyperparameters=None):
    overall_ylims = {}
    per_bucket_ylims = {}

    for variant in included_variants:
        if variant in ("weighted_binary_match_contribution", "occurrence_percentile_deviation"):
            overall_ylims[variant] = (0.0, 1.0)
        else:
            key = _OVERALL_KEYS[variant]
            global_minimum = 0.0
            for data in all_results.values():
                for value in data[key]:
                    if value < global_minimum:
                        global_minimum = value
            overall_ylims[variant] = (global_minimum, 0.0)

    for variant in included_variants:
        if variant == "occurrence_percentile_deviation":
            per_bucket_ylims[variant] = (0.0, 1.0)
        elif variant == "weighted_binary_match_contribution":
            absolute_maximum = 0.0
            weights_key = _PER_BUCKET_KEYS["weighted_binary_match_contribution"]
            cumulative_comparative_occurrence_ratios_key = "cumulative_comparative_occurrence_ratios"
            for data in all_results.values():
                weights = data.get(weights_key, {})
                cumulative_comparative_occurrence_ratios = data.get(cumulative_comparative_occurrence_ratios_key, {})
                for freq in voiced_frequency_bucket_centers:
                    weight_list = weights.get(freq, [])
                    ratio_list = cumulative_comparative_occurrence_ratios.get(freq, [])
                    for index, weight in enumerate(weight_list):
                        ratio = ratio_list[index] if index < len(ratio_list) else 0.5
                        signed_weight = _Signed_Weighted_Binary_Match_Contribution_Weight(weight, ratio, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound)
                        if abs(signed_weight) > absolute_maximum:
                            absolute_maximum = abs(signed_weight)
            per_bucket_ylims[variant] = (-absolute_maximum, absolute_maximum)
        elif variant == "accumulative_deviation":
            key = _PER_BUCKET_KEYS[variant]
            if accumulative_deviation_hyperparameters.get("use_non_directional_element_deviations", False):
                global_minimum = 0.0
                for data in all_results.values():
                    per_bucket = data.get(key, {})
                    for freq in voiced_frequency_bucket_centers:
                        for value in per_bucket.get(freq, []):
                            if value < global_minimum:
                                global_minimum = value
                per_bucket_ylims[variant] = (global_minimum, 0.0)
            else:
                absolute_maximum = 0.0
                for data in all_results.values():
                    per_bucket = data.get(key, {})
                    for freq in voiced_frequency_bucket_centers:
                        for value in per_bucket.get(freq, []):
                            if abs(value) > absolute_maximum:
                                absolute_maximum = abs(value)
                per_bucket_ylims[variant] = (-absolute_maximum, absolute_maximum)
        else:
            key = _PER_BUCKET_KEYS[variant]
            global_minimum = 0.0
            for data in all_results.values():
                per_bucket = data.get(key, {})
                for freq in voiced_frequency_bucket_centers:
                    for value in per_bucket.get(freq, []):
                        if value < global_minimum:
                            global_minimum = value
            per_bucket_ylims[variant] = (global_minimum, 0.0)

    return overall_ylims, per_bucket_ylims


# --- chart generation ---

def _Make_X_Values(point_count):
    return [index * Spectrogram_Window_Jump_In_Seconds for index in range(point_count)]


def _Build_Sequence_Filename_Suffix(speaker_segments):
    return "_".join(speaker_id for speaker_id, _, _ in speaker_segments)


def _Build_Sequence_Display_Label(speaker_segments):
    return " → ".join(speaker_id for speaker_id, _, _ in speaker_segments)


def _Draw_Speaker_Segment_Annotations(axis, speaker_segments):
    for index in range(1, len(speaker_segments)):
        previous_speaker_id, _, _ = speaker_segments[index - 1]
        speaker_id, start_index, end_index = speaker_segments[index]
        if start_index >= end_index or speaker_id == previous_speaker_id:
            continue
        boundary_x = start_index * Spectrogram_Window_Jump_In_Seconds
        axis.axvline(boundary_x, color=Get_Speaker_Color(speaker_id), linestyle=":", linewidth=1.0)

    for speaker_id, start_index, end_index in speaker_segments:
        if start_index >= end_index:
            continue
        midpoint_x = ((start_index + end_index - 1) / 2.0) * Spectrogram_Window_Jump_In_Seconds
        axis.annotate(
            speaker_id, xy=(midpoint_x, 0.0), xycoords=("data", "axes fraction"),
            xytext=(0, -15), textcoords="offset points",
            ha="center", va="top", fontsize=8, annotation_clip=False
        )


def Generate_Per_Speaker_Overall_Chart(voice_id, sequence_index, included_variants, overall_ylims, data):
    variant_count = len(included_variants)
    figure, axes = pyplot.subplots(variant_count, 1, figsize=(20, 5 * variant_count))
    if variant_count == 1:
        axes = [axes]

    speaker_segments = data["speaker_segments"]
    sequence_label = _Build_Sequence_Display_Label(speaker_segments)

    for axis, variant in zip(axes, included_variants):
        values = data[_OVERALL_KEYS[variant]]
        bucket_x_values = _Make_X_Values(len(values))

        for speaker_id, start_index, end_index in speaker_segments:
            if start_index >= end_index:
                continue
            segment_start = max(start_index - 1, 0)
            axis.plot(
                bucket_x_values[segment_start:end_index], values[segment_start:end_index],
                color=Get_Speaker_Color(speaker_id), linewidth=0.75
            )

        _Draw_Speaker_Segment_Annotations(axis, speaker_segments)

        y_limits = overall_ylims[variant]
        axis.set_ylim(y_limits[0], y_limits[1])
        axis.set_xlim(0, bucket_x_values[-1] if bucket_x_values else 0)
        axis.set_title(f"{variant} | {voice_id} baseline vs {sequence_label}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(variant)
        if y_limits[0] < 0 < y_limits[1]:
            axis.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    filename_suffix = _Build_Sequence_Filename_Suffix(speaker_segments)
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_overall_{voice_id}_{sequence_index}_{filename_suffix}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution, bbox_inches="tight")
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: overall chart saved to '{output_path}'")


def Generate_Per_Speaker_Per_Bucket_Chart(voice_id, sequence_index, included_variants, per_bucket_ylims, data, voiced_frequency_bucket_centers, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound):
    variant_count = len(included_variants)
    figure, axes = pyplot.subplots(variant_count, 1, figsize=(20, 5 * variant_count))
    if variant_count == 1:
        axes = [axes]

    colors = _Frequency_Colors(voiced_frequency_bucket_centers)
    speaker_segments = data["speaker_segments"]
    sequence_label = _Build_Sequence_Display_Label(speaker_segments)

    for axis, variant in zip(axes, included_variants):
        per_bucket_key = _PER_BUCKET_KEYS[variant]
        per_bucket = data.get(per_bucket_key, {})

        if variant == "weighted_binary_match_contribution":
            cumulative_comparative_occurrence_ratios = data.get("cumulative_comparative_occurrence_ratios", {})
            for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                weight_list = per_bucket.get(freq_center, [])
                ratio_list = cumulative_comparative_occurrence_ratios.get(freq_center, [])
                signed_values = [
                    _Signed_Weighted_Binary_Match_Contribution_Weight(weight, ratio_list[index] if index < len(ratio_list) else 0.5, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound)
                    for index, weight in enumerate(weight_list)
                ]
                bucket_x_values = _Make_X_Values(len(signed_values))
                axis.plot(bucket_x_values, signed_values, color=colors[freq_index], linewidth=0.5)
        else:
            for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                values = per_bucket.get(freq_center, [])
                bucket_x_values = _Make_X_Values(len(values))
                axis.plot(bucket_x_values, values, color=colors[freq_index], linewidth=0.5)

        _Draw_Speaker_Segment_Annotations(axis, speaker_segments)

        y_limits = per_bucket_ylims.get(variant, (-1.0, 1.0))
        axis.set_ylim(y_limits[0], y_limits[1])
        maximum_x_value = max(
            (len(per_bucket.get(freq, [])) - 1) * Spectrogram_Window_Jump_In_Seconds
            for freq in voiced_frequency_bucket_centers
            if per_bucket.get(freq)
        ) if per_bucket else 0
        axis.set_xlim(0, maximum_x_value)
        axis.set_title(f"{variant} (per bucket) | {voice_id} baseline vs {sequence_label}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(variant)
        if y_limits[0] < 0:
            axis.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    filename_suffix = _Build_Sequence_Filename_Suffix(speaker_segments)
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_per_bucket_{voice_id}_{sequence_index}_{filename_suffix}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution, bbox_inches="tight")
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: per-bucket chart saved to '{output_path}'")


def Generate_Combined_Overall_Chart(voice_id, included_variants, overall_ylims, all_results):
    variant_count = len(included_variants)
    figure, axes = pyplot.subplots(variant_count, 1, figsize=(20, 5 * variant_count))
    if variant_count == 1:
        axes = [axes]

    for axis, variant in zip(axes, included_variants):
        overall_key = _OVERALL_KEYS[variant]
        labeled_speaker_ids = set()
        for data in all_results.values():
            values = data[overall_key]
            bucket_x_values = _Make_X_Values(len(values))
            for speaker_id, start_index, end_index in data["speaker_segments"]:
                if start_index >= end_index:
                    continue
                segment_start = max(start_index - 1, 0)
                linewidth = 1.5 if speaker_id == voice_id else 0.75
                label = speaker_id if speaker_id not in labeled_speaker_ids else None
                labeled_speaker_ids.add(speaker_id)
                axis.plot(
                    bucket_x_values[segment_start:end_index], values[segment_start:end_index],
                    color=Get_Speaker_Color(speaker_id), linewidth=linewidth, label=label
                )
        y_limits = overall_ylims[variant]
        axis.set_ylim(y_limits[0], y_limits[1])
        axis.set_title(f"{variant} — All Sequences vs {voice_id} Baseline")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(variant)
        axis.legend()
        if y_limits[0] < 0 < y_limits[1]:
            axis.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_combined_{voice_id}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: combined chart saved to '{output_path}'")


# --- entry point ---

def Run_Element_Match_Contribution_Type_Exploration(
    voice_id,
    comparative_voices_audio_set,
    aggregate_match_types,
    cross_type_hyperparameters
):
    occurrence_ratio_cumulation_weight = Convert_Half_Life_To_Cumulation_Weight(
        Spectrogram_Window_Jump_In_Seconds, cross_type_hyperparameters["occurrence_ratio_cumulation_half_life"]
    )
    use_bell_curve = cross_type_hyperparameters.get("use_bell_curve_percentile_projection", False)

    included_variants = [variant for variant in VARIANT_ORDER if aggregate_match_types.get(variant, {}).get("include_variant", False)]
    if not included_variants:
        print("Element_Match_Contribution_Type_Explorer: no variants included, aborting")
        return

    include_weighted_binary_match_contribution = "weighted_binary_match_contribution" in included_variants
    include_occurrence_percentile_deviation = "occurrence_percentile_deviation" in included_variants
    include_occurrence_percentile_inverse_deviation = "occurrence_percentile_inverse_deviation" in included_variants
    include_occurrence_percentile_half_distance = "occurrence_percentile_half_distance" in included_variants
    include_raw_distance = "raw_distance" in included_variants
    include_accumulative_deviation = "accumulative_deviation" in included_variants
    need_cumulative_comparative_occurrence_ratios = (
        include_weighted_binary_match_contribution or include_occurrence_percentile_deviation
        or include_occurrence_percentile_inverse_deviation or include_occurrence_percentile_half_distance
    )

    weighted_binary_match_contribution_hyperparameters = aggregate_match_types.get("weighted_binary_match_contribution", {}).get("hyperparameters", {})
    weighted_binary_match_contribution_lower_bound = weighted_binary_match_contribution_upper_bound = None
    if include_weighted_binary_match_contribution:
        positive_contribution_range = weighted_binary_match_contribution_hyperparameters["positive_contribution_range"]
        weighted_binary_match_contribution_lower_bound = 0.5 - (positive_contribution_range * 0.5)
        weighted_binary_match_contribution_upper_bound = 0.5 + (positive_contribution_range * 0.5)
        weighted_binary_match_contribution_positive_weight_power_curve = weighted_binary_match_contribution_hyperparameters["positive_weight_power_curve"]
        weighted_binary_match_contribution_negative_weight_proximity_half_distance_increment = weighted_binary_match_contribution_hyperparameters["negative_weight_proximity_half_distance_increment"]

    occurrence_percentile_inverse_deviation_hyperparameters = aggregate_match_types.get("occurrence_percentile_inverse_deviation", {}).get("hyperparameters", {})
    occurrence_percentile_half_distance_hyperparameters = aggregate_match_types.get("occurrence_percentile_half_distance", {}).get("hyperparameters", {})

    accumulative_deviation_hyperparameters = aggregate_match_types.get("accumulative_deviation", {}).get("hyperparameters", {})
    accumulative_deviation_decay_rate = None
    accumulative_deviation_deviation_type = None
    accumulative_deviation_use_non_directional_element_deviations = False
    accumulative_deviation_use_average_element_deviations = False
    if include_accumulative_deviation:
        accumulative_deviation_decay_rate = Convert_Half_Life_To_Cumulation_Weight(
            Spectrogram_Window_Jump_In_Seconds, accumulative_deviation_hyperparameters["decay_half_life"]
        )
        accumulative_deviation_deviation_type = accumulative_deviation_hyperparameters["deviation_type"]
        accumulative_deviation_use_non_directional_element_deviations = accumulative_deviation_hyperparameters["use_non_directional_element_deviations"]
        accumulative_deviation_use_average_element_deviations = accumulative_deviation_hyperparameters["use_average_element_deviations"]

    need_percentile_deviation_for_accumulative_deviation = include_accumulative_deviation and accumulative_deviation_deviation_type in (
        "occurrence_percentile_inverse_deviation", "occurrence_percentile_half_distance"
    )
    need_bucket_medians = include_raw_distance or (
        include_accumulative_deviation and (
            accumulative_deviation_deviation_type == "raw_distance" or not accumulative_deviation_use_non_directional_element_deviations
        )
    )

    voice_profile_half_life = cross_type_hyperparameters.get("voice_profile_cumulation_half_life")
    state_path = Json_Directory + f"Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts{Format_Half_Life_For_Filename(voice_profile_half_life)}.json"
    state = Load_Layered_State(state_path)
    if state is None:
        print(f"Element_Match_Contribution_Type_Explorer: no data found for voice_id '{voice_id}', aborting")
        return

    inverted_occurrence_ratios = Convert_Occurrence_Counts_To_Ratios(
        state["frequency_amount_occurrence_counts"],
        state["total_voiced_frequency_timepoints_count"],
        invert=True
    )

    if use_bell_curve:
        bell_curve_projections = _Extract_Bell_Curve_Projections(inverted_occurrence_ratios)
        sorted_keys_per_bucket = None
        bucket_medians = [projection[0] for projection in bell_curve_projections] if need_bucket_medians else None
    else:
        bell_curve_projections = None
        sorted_keys_per_bucket = _Build_Sorted_Keys(inverted_occurrence_ratios)
        bucket_medians = _Extract_Medians(inverted_occurrence_ratios) if need_bucket_medians else None

    voiced_frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
    voiced_frequency_limit_index = len(voiced_frequency_bucket_centers)

    all_results = {}

    for sequence_index, sub_sequences in enumerate(comparative_voices_audio_set):
        cumulative_comparative_occurrence_ratios = {freq: [0.5] for freq in voiced_frequency_bucket_centers} if need_cumulative_comparative_occurrence_ratios else {}
        cumulative_raw_distances = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_raw_distance else {}

        match_contribution_weights = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_weighted_binary_match_contribution else {}
        weighted_binary_match_contributions = [0.5] if include_weighted_binary_match_contribution else []

        need_occurrence_percentile_deviation_buckets = include_occurrence_percentile_deviation or include_occurrence_percentile_inverse_deviation or include_occurrence_percentile_half_distance
        occurrence_percentile_deviations = {freq: [1.0] for freq in voiced_frequency_bucket_centers} if need_occurrence_percentile_deviation_buckets else {}
        average_occurrence_percentile_deviations = [1.0] if include_occurrence_percentile_deviation else []

        occurrence_percentile_inverse_deviations = {freq: [-1.0] for freq in voiced_frequency_bucket_centers} if include_occurrence_percentile_inverse_deviation else {}
        average_occurrence_percentile_inverse_deviations = [-1.0] if include_occurrence_percentile_inverse_deviation else []

        occurrence_percentile_half_distances = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_occurrence_percentile_half_distance else {}
        average_occurrence_percentile_half_distances = [0.0] if include_occurrence_percentile_half_distance else []

        average_raw_distances = [0.0] if include_raw_distance else []

        element_accumulative_deviations = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_accumulative_deviation else {}
        average_element_accumulative_deviations = [0.0] if include_accumulative_deviation else []

        speaker_segments = []
        processed_timepoint_count = 0

        for speaker_id, audio_list in sub_sequences:
            segment_start_index = processed_timepoint_count + 1

            for audio_name in audio_list:
                print(f"Element_Match_Contribution_Type_Explorer: processing '{speaker_id}/{audio_name}'...")
                distribution, audio_frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)

                for timepoint_index in range(len(distribution[0])):
                    voiced_ratio = numpy.sum(distribution[:voiced_frequency_limit_index, timepoint_index])
                    if voiced_ratio < Subdistribution_Timepoint_Voiced_Ratio_Minimum:
                        continue
                    if timepoint_phonemes[timepoint_index] is None:
                        continue

                    timepoint_weighted_binary_match_contribution_weights = []
                    timepoint_weighted_binary_match_contribution_is_positive = []
                    timepoint_occurrence_percentile_deviations = []
                    timepoint_occurrence_percentile_inverse_deviations = []
                    timepoint_occurrence_percentile_half_distances = []
                    timepoint_raw_distances = []
                    timepoint_element_accumulative_deviations = []

                    for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                        timepoint_ratio = float(distribution[freq_index][timepoint_index])

                        if need_cumulative_comparative_occurrence_ratios or need_percentile_deviation_for_accumulative_deviation:
                            if use_bell_curve:
                                center, lower_standard_deviation, upper_standard_deviation = bell_curve_projections[freq_index]
                                value_2 = _Bell_Curve_Value_2(timepoint_ratio, center, lower_standard_deviation, upper_standard_deviation)
                            else:
                                value_2 = _Lookup_Closest_Value(
                                    inverted_occurrence_ratios[freq_index],
                                    sorted_keys_per_bucket[freq_index],
                                    timepoint_ratio
                                )

                            if need_cumulative_comparative_occurrence_ratios:
                                previous_ratio = cumulative_comparative_occurrence_ratios[freq_center][-1]
                                new_ratio = Weighted_Average(previous_ratio, occurrence_ratio_cumulation_weight, value_2, 1.0 - occurrence_ratio_cumulation_weight)
                                cumulative_comparative_occurrence_ratios[freq_center].append(new_ratio)

                        if include_weighted_binary_match_contribution:
                            weight = _Weighted_Binary_Match_Contribution_Weight(new_ratio, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound, positive_contribution_range, weighted_binary_match_contribution_positive_weight_power_curve, weighted_binary_match_contribution_negative_weight_proximity_half_distance_increment)
                            match_contribution_weights[freq_center].append(weight)
                            timepoint_weighted_binary_match_contribution_weights.append(weight)
                            timepoint_weighted_binary_match_contribution_is_positive.append(weighted_binary_match_contribution_lower_bound <= new_ratio <= weighted_binary_match_contribution_upper_bound)

                        if need_occurrence_percentile_deviation_buckets:
                            deviation = _Occurrence_Percentile_Deviation(new_ratio)
                            occurrence_percentile_deviations[freq_center].append(deviation)
                            timepoint_occurrence_percentile_deviations.append(deviation)

                            if include_occurrence_percentile_inverse_deviation:
                                inverse_deviation = _Occurrence_Percentile_Inverse_Deviation(deviation, occurrence_percentile_inverse_deviation_hyperparameters["deviation_power_curve"], occurrence_percentile_inverse_deviation_hyperparameters["inverse_deviation_minimum"])
                                occurrence_percentile_inverse_deviations[freq_center].append(inverse_deviation)
                                timepoint_occurrence_percentile_inverse_deviations.append(inverse_deviation)

                            if include_occurrence_percentile_half_distance:
                                half_distance = _Occurrence_Percentile_Half_Distance(deviation, occurrence_percentile_half_distance_hyperparameters["half_distance_minimum"])
                                occurrence_percentile_half_distances[freq_center].append(half_distance)
                                timepoint_occurrence_percentile_half_distances.append(half_distance)

                        if include_raw_distance:
                            raw_distance_value_2 = -1.0 * abs(bucket_medians[freq_index] - timepoint_ratio)
                            previous_raw_distance = cumulative_raw_distances[freq_center][-1]
                            new_raw_distance = Weighted_Average(previous_raw_distance, occurrence_ratio_cumulation_weight, raw_distance_value_2, 1.0 - occurrence_ratio_cumulation_weight)
                            cumulative_raw_distances[freq_center].append(new_raw_distance)
                            timepoint_raw_distances.append(new_raw_distance)

                        if include_accumulative_deviation:
                            if accumulative_deviation_deviation_type == "raw_distance":
                                current_timepoint_deviation = -1.0 * abs(bucket_medians[freq_index] - timepoint_ratio)
                            else:
                                current_timepoint_percentile_deviation = _Occurrence_Percentile_Deviation(value_2)
                                if accumulative_deviation_deviation_type == "occurrence_percentile_inverse_deviation":
                                    current_timepoint_deviation = _Occurrence_Percentile_Inverse_Deviation(
                                        current_timepoint_percentile_deviation,
                                        occurrence_percentile_inverse_deviation_hyperparameters["deviation_power_curve"],
                                        occurrence_percentile_inverse_deviation_hyperparameters["inverse_deviation_minimum"]
                                    )
                                else:
                                    current_timepoint_deviation = _Occurrence_Percentile_Half_Distance(
                                        current_timepoint_percentile_deviation,
                                        occurrence_percentile_half_distance_hyperparameters["half_distance_minimum"]
                                    )

                            if not accumulative_deviation_use_non_directional_element_deviations:
                                if timepoint_ratio < bucket_medians[freq_index]:
                                    current_timepoint_deviation *= -1.0

                            previous_element_accumulative_deviation = element_accumulative_deviations[freq_center][-1]
                            if accumulative_deviation_use_average_element_deviations:
                                new_element_accumulative_deviation = Weighted_Average(
                                    previous_element_accumulative_deviation, accumulative_deviation_decay_rate,
                                    current_timepoint_deviation, 1.0 - accumulative_deviation_decay_rate
                                )
                            else:
                                new_element_accumulative_deviation = (previous_element_accumulative_deviation + current_timepoint_deviation) * accumulative_deviation_decay_rate

                            element_accumulative_deviations[freq_center].append(new_element_accumulative_deviation)
                            timepoint_element_accumulative_deviations.append(new_element_accumulative_deviation)

                    if include_weighted_binary_match_contribution:
                        total_weight = sum(timepoint_weighted_binary_match_contribution_weights)
                        if total_weight == 0.0:
                            weighted_binary_match_contributions.append(0.5)
                        else:
                            weighted_binary_match_contributions.append(
                                sum(weight for weight, positive in zip(timepoint_weighted_binary_match_contribution_weights, timepoint_weighted_binary_match_contribution_is_positive) if positive) / total_weight
                            )

                    if include_occurrence_percentile_deviation:
                        average_occurrence_percentile_deviations.append(sum(timepoint_occurrence_percentile_deviations) / len(timepoint_occurrence_percentile_deviations) if timepoint_occurrence_percentile_deviations else 1.0)

                    if include_occurrence_percentile_inverse_deviation:
                        average_occurrence_percentile_inverse_deviations.append(sum(timepoint_occurrence_percentile_inverse_deviations) / len(timepoint_occurrence_percentile_inverse_deviations) if timepoint_occurrence_percentile_inverse_deviations else -1.0)

                    if include_occurrence_percentile_half_distance:
                        average_occurrence_percentile_half_distances.append(sum(timepoint_occurrence_percentile_half_distances) / len(timepoint_occurrence_percentile_half_distances) if timepoint_occurrence_percentile_half_distances else 0.0)

                    if include_raw_distance:
                        average_raw_distances.append(sum(timepoint_raw_distances) / len(timepoint_raw_distances) if timepoint_raw_distances else 0.0)

                    if include_accumulative_deviation:
                        average_element_accumulative_deviations.append(
                            -1.0 * (sum(abs(value) for value in timepoint_element_accumulative_deviations) / len(timepoint_element_accumulative_deviations))
                            if timepoint_element_accumulative_deviations else 0.0
                        )

                    processed_timepoint_count += 1

            segment_end_index = processed_timepoint_count + 1
            speaker_segments.append((speaker_id, segment_start_index, segment_end_index))

        speaker_data = {"speaker_segments": speaker_segments}
        if need_cumulative_comparative_occurrence_ratios:
            speaker_data["cumulative_comparative_occurrence_ratios"] = cumulative_comparative_occurrence_ratios
        if include_weighted_binary_match_contribution:
            speaker_data["match_contribution_weights"] = match_contribution_weights
            speaker_data["weighted_binary_match_contributions"] = weighted_binary_match_contributions
        if need_occurrence_percentile_deviation_buckets:
            speaker_data["occurrence_percentile_deviations"] = occurrence_percentile_deviations
        if include_occurrence_percentile_deviation:
            speaker_data["average_occurrence_percentile_deviations"] = average_occurrence_percentile_deviations
        if include_occurrence_percentile_inverse_deviation:
            speaker_data["occurrence_percentile_inverse_deviations"] = occurrence_percentile_inverse_deviations
            speaker_data["average_occurrence_percentile_inverse_deviations"] = average_occurrence_percentile_inverse_deviations
        if include_occurrence_percentile_half_distance:
            speaker_data["occurrence_percentile_half_distances"] = occurrence_percentile_half_distances
            speaker_data["average_occurrence_percentile_half_distances"] = average_occurrence_percentile_half_distances
        if include_raw_distance:
            speaker_data["cumulative_raw_distances"] = cumulative_raw_distances
            speaker_data["average_raw_distances"] = average_raw_distances
        if include_accumulative_deviation:
            speaker_data["element_accumulative_deviations"] = element_accumulative_deviations
            speaker_data["average_element_accumulative_deviations"] = average_element_accumulative_deviations
        all_results[sequence_index] = speaker_data

    overall_ylims, per_bucket_ylims = _Compute_Global_Ylims(
        included_variants, all_results, voiced_frequency_bucket_centers,
        weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound,
        accumulative_deviation_hyperparameters
    )

    os.makedirs(Analysis_Directory, exist_ok=True)

    for sequence_index in range(len(comparative_voices_audio_set)):
        Generate_Per_Speaker_Overall_Chart(voice_id, sequence_index, included_variants, overall_ylims, all_results[sequence_index])
        Generate_Per_Speaker_Per_Bucket_Chart(voice_id, sequence_index, included_variants, per_bucket_ylims, all_results[sequence_index], voiced_frequency_bucket_centers, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound)

    Generate_Combined_Overall_Chart(voice_id, included_variants, overall_ylims, all_results)
    print(f"Element_Match_Contribution_Type_Explorer: exploration complete for voice_id '{voice_id}'")
