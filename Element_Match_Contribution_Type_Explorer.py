import bisect
import json
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
from Simulated_Conversation_Generator import Conversation_Sequence_Json_Directory


_SQRT2 = math.sqrt(2.0)
_LOG_0_5 = math.log(0.5)

VARIANT_ORDER = [
    "weighted_binary_match_contribution",
    "occurrence_percentile_deviation",
    "occurrence_percentile_inverse_deviation",
    "occurrence_percentile_half_distance",
    "accumulative_deviation",
]

METRIC_ORDER = [
    "match_ratio",
    "transition_duration",
    "match_differentiation",
]

_OVERALL_KEYS = {
    "weighted_binary_match_contribution": "weighted_binary_match_contributions",
    "occurrence_percentile_deviation": "average_occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "average_occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "average_occurrence_percentile_half_distances",
    "accumulative_deviation": "average_element_accumulative_deviations",
}

_PER_BUCKET_KEYS = {
    "weighted_binary_match_contribution": "match_contribution_weights",
    "occurrence_percentile_deviation": "occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "occurrence_percentile_half_distances",
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
    # new_ratio can land exactly on 0.0/1.0 (e.g. erf saturation under signal-rate simulation's transient overshoot), making crossing_point_proximity_ratio 0 and math.log(0, 0.5) domain-error; floor it at a tiny epsilon so the result stays finite (very negative) instead of raising
    crossing_point_proximity_ratio = max(crossing_point_proximity_ratio, 1e-15)
    return math.log(crossing_point_proximity_ratio, 0.5) * negative_weight_proximity_half_distance_increment


def _Occurrence_Percentile_Deviation(new_ratio):
    return abs(0.5 - new_ratio) * 2.0


def _Occurrence_Percentile_Inverse_Deviation(deviation, power_curve, minimum):
    # deviation is 0 at a perfect match and 1 at the worst match; closeness restores the inverse (1 = perfect) scale these formulas are written against
    closeness = 1.0 - deviation
    if closeness <= 0.0:
        return minimum
    return max((-1.0 / (closeness ** power_curve)) + 1.0, minimum)


def _Occurrence_Percentile_Half_Distance(deviation, minimum):
    closeness = 1.0 - deviation
    if closeness <= 0.0:
        return minimum
    return max(-1.0 * (math.log(closeness) / _LOG_0_5), minimum)


# --- global bounds computation ---

def _Compute_Global_Ylims(included_variants, all_results, voiced_frequency_bucket_centers, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound, accumulative_deviation_hyperparameters=None, chart_y_minimums=None):
    overall_ylims = {}
    per_bucket_ylims = {}
    chart_y_minimums = chart_y_minimums or {}

    # every variant now shares the same "0 is best, negative is worst" overall scale, so the true observed minimum is computed identically for all of them (NaN entries — from leading non-voiced timepoints — are silently skipped, since any comparison against NaN is False) and then floored at that variant's chart_y_minimum, if one is set and the observed minimum would otherwise be lower
    for variant in included_variants:
        key = _OVERALL_KEYS[variant]
        global_minimum = 0.0
        for data in all_results.values():
            for value in data[key]:
                if value < global_minimum:
                    global_minimum = value
        chart_y_minimum = chart_y_minimums.get(variant, float("-inf"))
        overall_ylims[variant] = (max(global_minimum, chart_y_minimum), 0.0)

    for variant in included_variants:
        if variant == "weighted_binary_match_contribution":
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


# --- aggregate metrics ---

def _Compute_All_Speaker_Metrics(variant, voice_ids, per_voice_results, sequence_index, speaker_segments, metric_inclusions):
    want_match_ratio = metric_inclusions.get("match_ratio", False)
    want_transition_duration = metric_inclusions.get("transition_duration", False)
    want_match_differentiation = metric_inclusions.get("match_differentiation", False)
    if not (want_match_ratio or want_transition_duration or want_match_differentiation):
        return {}

    overall_key = _OVERALL_KEYS[variant]
    voice_value_lists = {}
    for voice_id in voice_ids:
        all_results = per_voice_results.get(voice_id)
        if all_results is None:
            continue
        voice_value_lists[voice_id] = all_results[sequence_index][overall_key]

    match_ratio_numerator = 0
    match_ratio_denominator = 0
    transition_durations = []
    match_differentiation_terms = []

    # audio periods for a speaker whose voice isn't in voice_value_lists (not among the compared voices) are excluded entirely, from both numerator and denominator of every metric
    for speaker_id, start_index, end_index in speaker_segments:
        if start_index >= end_index or speaker_id not in voice_value_lists:
            continue

        own_values = voice_value_lists[speaker_id]
        first_reached_index = None

        for timepoint_index in range(start_index, end_index):
            own_value = own_values[timepoint_index] if timepoint_index < len(own_values) else math.nan
            if math.isnan(own_value):
                # a NaN own-value (e.g. a leading non-voiced timepoint under include_non_voiced_timepoints) can't meaningfully be compared, so this timepoint is skipped entirely, same as a not-in-voice_ids period
                continue

            other_values = {}
            for other_voice_id, values in voice_value_lists.items():
                if other_voice_id == speaker_id:
                    continue
                other_value = values[timepoint_index] if timepoint_index < len(values) else math.nan
                if not math.isnan(other_value):
                    other_values[other_voice_id] = other_value

            if want_match_ratio or want_transition_duration:
                # a tie between the speaker's own voice and any other voice does not count as the speaker's voice having the highest value, so this is a strict inequality against every other candidate (vacuously True if there are no other candidate values at this timepoint)
                is_speaker_highest = all(own_value > other_value for other_value in other_values.values())
                if want_match_ratio:
                    match_ratio_denominator += 1
                    if is_speaker_highest:
                        match_ratio_numerator += 1
                if want_transition_duration and first_reached_index is None and is_speaker_highest:
                    first_reached_index = timepoint_index

            if want_match_differentiation:
                # a zero-valued other-voice would be a divide-by-zero; that single (own / other) comparison is skipped rather than the whole timepoint
                ratio_terms = [own_value / other_value for other_value in other_values.values() if other_value != 0.0]
                if ratio_terms:
                    match_differentiation_terms.append(sum(ratio_terms) / len(ratio_terms))

        if want_transition_duration:
            # if the speaker's voice never reaches the highest value within its own turn, the full turn duration is used as the elapsed time
            elapsed_timepoint_count = (first_reached_index - start_index) if first_reached_index is not None else (end_index - start_index)
            transition_durations.append(elapsed_timepoint_count * Spectrogram_Window_Jump_In_Seconds)

    metrics = {}
    if want_match_ratio:
        metrics["match_ratio"] = (match_ratio_numerator / match_ratio_denominator) if match_ratio_denominator > 0 else math.nan
    if want_transition_duration:
        metrics["transition_duration"] = (sum(transition_durations) / len(transition_durations)) if transition_durations else math.nan
    if want_match_differentiation:
        metrics["match_differentiation"] = (sum(match_differentiation_terms) / len(match_differentiation_terms)) if match_differentiation_terms else math.nan

    return metrics


def _Format_Metrics_Text(metrics):
    return ", ".join(f"{metric_name} {round(metrics[metric_name], 3)}" for metric_name in METRIC_ORDER if metric_name in metrics)


# --- chart generation ---

def _Make_X_Values(point_count):
    return [index * Spectrogram_Window_Jump_In_Seconds for index in range(point_count)]


def _Build_Sequence_Filename_Suffix(speaker_segments):
    return "_".join(speaker_id for speaker_id, _, _ in speaker_segments)


def _Build_Sequence_Display_Label(speaker_segments):
    return " → ".join(speaker_id for speaker_id, _, _ in speaker_segments)


def _Draw_Speaker_Segment_Annotations(axis, speaker_segments, include_leading_line=False):
    if include_leading_line:
        for speaker_id, start_index, end_index in speaker_segments:
            if start_index >= end_index:
                continue
            # a line drawn exactly at the first timepoint (x=0) would sit on top of the axis's left spine and be hidden, so nudge it forward by half a timepoint step
            leading_line_x = Spectrogram_Window_Jump_In_Seconds / 2.0
            axis.axvline(leading_line_x, color=Get_Speaker_Color(speaker_id), linestyle=":", linewidth=1.0)
            break

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

        axis.plot(bucket_x_values, values, color=Get_Speaker_Color(voice_id), linewidth=0.75)

        _Draw_Speaker_Segment_Annotations(axis, speaker_segments, include_leading_line=True)

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


def Generate_All_Speaker_Overall_Chart(voice_ids, sequence_index, included_variants, overall_ylims, per_voice_results, speaker_segments, metric_inclusions):
    variant_count = len(included_variants)
    figure, axes = pyplot.subplots(variant_count, 1, figsize=(20, 5 * variant_count))
    if variant_count == 1:
        axes = [axes]

    sequence_label = _Build_Sequence_Display_Label(speaker_segments)

    for axis, variant in zip(axes, included_variants):
        overall_key = _OVERALL_KEYS[variant]
        max_x_value = 0.0

        for voice_id in voice_ids:
            all_results = per_voice_results.get(voice_id)
            if all_results is None:
                continue
            values = all_results[sequence_index][overall_key]
            bucket_x_values = _Make_X_Values(len(values))
            if bucket_x_values:
                max_x_value = max(max_x_value, bucket_x_values[-1])
            axis.plot(bucket_x_values, values, color=Get_Speaker_Color(voice_id), linewidth=0.75, label=voice_id)

        _Draw_Speaker_Segment_Annotations(axis, speaker_segments, include_leading_line=True)

        y_limits = overall_ylims[variant]
        axis.set_ylim(y_limits[0], y_limits[1])
        axis.set_xlim(0, max_x_value)
        metrics = _Compute_All_Speaker_Metrics(variant, voice_ids, per_voice_results, sequence_index, speaker_segments, metric_inclusions)
        metrics_text = _Format_Metrics_Text(metrics)
        title = f"{variant} | {sequence_label}"
        if metrics_text:
            title += f" | {metrics_text}"
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(variant)
        axis.legend()
        if y_limits[0] < 0 < y_limits[1]:
            axis.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    filename_suffix = _Build_Sequence_Filename_Suffix(speaker_segments)
    voice_ids_suffix = "_".join(voice_ids)
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_all_speaker_overall_{voice_ids_suffix}_{sequence_index}_{filename_suffix}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution, bbox_inches="tight")
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: all-speaker overall chart saved to '{output_path}'")


# --- entry point ---

def Load_Comparative_Voices_Audio_Set(conversation_json_file_name):
    conversation_json_path = Conversation_Sequence_Json_Directory + conversation_json_file_name
    with open(conversation_json_path, "r") as f:
        raw_comparative_voices_audio_set = json.load(f)
    return [
        [(speaker_id, audio_list) for speaker_id, audio_list in sub_sequences]
        for sub_sequences in raw_comparative_voices_audio_set
    ]


def Run_Element_Match_Contribution_Type_Exploration(
    voice_ids,
    conversation_json_file_name,
    aggregate_match_types,
    cross_type_hyperparameters,
    chart_type_inclusions,
    metric_inclusions
):
    comparative_voices_audio_set = Load_Comparative_Voices_Audio_Set(conversation_json_file_name)

    occurrence_ratio_cumulation_weight = Convert_Half_Life_To_Cumulation_Weight(
        Spectrogram_Window_Jump_In_Seconds, cross_type_hyperparameters["occurrence_ratio_cumulation_half_life"]
    )
    use_bell_curve = cross_type_hyperparameters.get("use_bell_curve_percentile_projection", False)
    use_signal_rate_simulation = cross_type_hyperparameters.get("use_signal_rate_simulation", False)
    include_non_voiced_timepoints_hyperparameter = cross_type_hyperparameters.get("include_non_voiced_timepoints", False)
    if include_non_voiced_timepoints_hyperparameter and not use_signal_rate_simulation:
        print("Element_Match_Contribution_Type_Explorer: WARNING - cross_type_hyperparameters['include_non_voiced_timepoints'] is True but 'use_signal_rate_simulation' is False; include_non_voiced_timepoints requires use_signal_rate_simulation and will be ignored for this run")
    include_non_voiced_timepoints = include_non_voiced_timepoints_hyperparameter and use_signal_rate_simulation

    included_variants = [variant for variant in VARIANT_ORDER if aggregate_match_types.get(variant, {}).get("include_variant", False)]
    if not included_variants:
        print("Element_Match_Contribution_Type_Explorer: no variants included, aborting")
        return

    include_combined_overall_chart = chart_type_inclusions.get("combined_overall", False)
    include_all_speaker_overall_chart = chart_type_inclusions.get("all_speaker_overall", False)
    include_per_speaker_overall_chart = chart_type_inclusions.get("per_speaker_overall", False)
    include_per_speaker_per_bucket_chart = chart_type_inclusions.get("per_speaker_per_bucket", False)
    if not (include_combined_overall_chart or include_all_speaker_overall_chart or include_per_speaker_overall_chart or include_per_speaker_per_bucket_chart):
        print("Element_Match_Contribution_Type_Explorer: no chart types included, aborting")
        return

    need_per_voice_ylims = include_per_speaker_overall_chart or include_combined_overall_chart or include_per_speaker_per_bucket_chart

    # an optional per-variant floor on how deep the overall-value y-axis is allowed to go; values below it are still counted toward aggregate metrics, just not displayed
    chart_y_minimums = {
        variant: aggregate_match_types.get(variant, {}).get("hyperparameters", {}).get("chart_y_minimum", float("-inf"))
        for variant in included_variants
    }

    include_weighted_binary_match_contribution = "weighted_binary_match_contribution" in included_variants
    include_occurrence_percentile_deviation = "occurrence_percentile_deviation" in included_variants
    include_occurrence_percentile_inverse_deviation = "occurrence_percentile_inverse_deviation" in included_variants
    include_occurrence_percentile_half_distance = "occurrence_percentile_half_distance" in included_variants
    include_accumulative_deviation = "accumulative_deviation" in included_variants
    need_cumulative_comparative_occurrence_ratios = (
        include_weighted_binary_match_contribution or include_occurrence_percentile_deviation
        or include_occurrence_percentile_inverse_deviation or include_occurrence_percentile_half_distance
    )
    need_occurrence_percentile_deviation_buckets = include_occurrence_percentile_deviation or include_occurrence_percentile_inverse_deviation or include_occurrence_percentile_half_distance

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
    accumulative_deviation_use_self_tracking_reset = False
    if include_accumulative_deviation:
        accumulative_deviation_decay_rate = Convert_Half_Life_To_Cumulation_Weight(
            Spectrogram_Window_Jump_In_Seconds, accumulative_deviation_hyperparameters["decay_half_life"]
        )
        accumulative_deviation_deviation_type = accumulative_deviation_hyperparameters["deviation_type"]
        accumulative_deviation_use_non_directional_element_deviations = accumulative_deviation_hyperparameters["use_non_directional_element_deviations"]
        accumulative_deviation_use_average_element_deviations = accumulative_deviation_hyperparameters["use_average_element_deviations"]
        accumulative_deviation_use_self_tracking_reset = accumulative_deviation_hyperparameters["use_self_tracking_reset"]

    need_percentile_deviation_for_accumulative_deviation = include_accumulative_deviation
    need_bucket_medians = include_accumulative_deviation and not accumulative_deviation_use_non_directional_element_deviations

    voice_profile_half_life = cross_type_hyperparameters.get("voice_profile_cumulation_half_life")

    # audio -> bucketed frequency distribution is voice_id-independent, so it's computed once per (speaker_id, audio_name) and reused across every voice_id's baseline comparison, instead of being recomputed once per voice_id
    audio_cache = {}
    for sub_sequences in comparative_voices_audio_set:
        for speaker_id, audio_list in sub_sequences:
            for audio_name in audio_list:
                cache_key = (speaker_id, audio_name)
                if cache_key not in audio_cache:
                    print(f"Element_Match_Contribution_Type_Explorer: processing '{speaker_id}/{audio_name}'...")
                    distribution, audio_frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
                    audio_cache[cache_key] = (distribution, timepoint_phonemes)

    def _Process_Sequences_For_Voice(voiced_frequency_bucket_centers, voiced_frequency_limit_index, inverted_occurrence_ratios, sorted_keys_per_bucket, bell_curve_projections, bucket_medians):
        all_results = {}

        for sequence_index, sub_sequences in enumerate(comparative_voices_audio_set):
            cumulative_comparative_occurrence_ratios = {freq: [0.5] for freq in voiced_frequency_bucket_centers} if need_cumulative_comparative_occurrence_ratios else {}
            # both signal-rate running totals reset together at the start of every sequence, so a new conversation never inherits a stale denominator/numerator pairing from the previous one
            total_distribution_signal_rate = 0.0
            distribution_ratio_signal_rates = {freq: 0.0 for freq in voiced_frequency_bucket_centers}

            match_contribution_weights = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_weighted_binary_match_contribution else {}
            weighted_binary_match_contributions = [-0.5] if include_weighted_binary_match_contribution else []

            occurrence_percentile_deviations = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if need_occurrence_percentile_deviation_buckets else {}
            average_occurrence_percentile_deviations = [0.0] if include_occurrence_percentile_deviation else []

            occurrence_percentile_inverse_deviations = {freq: [-1.0] for freq in voiced_frequency_bucket_centers} if include_occurrence_percentile_inverse_deviation else {}
            average_occurrence_percentile_inverse_deviations = [-1.0] if include_occurrence_percentile_inverse_deviation else []

            occurrence_percentile_half_distances = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_occurrence_percentile_half_distance else {}
            average_occurrence_percentile_half_distances = [0.0] if include_occurrence_percentile_half_distance else []

            element_accumulative_deviations = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_accumulative_deviation else {}
            average_element_accumulative_deviations = [0.0] if include_accumulative_deviation else []

            speaker_segments = []
            processed_timepoint_count = 0

            for speaker_id, audio_list in sub_sequences:
                segment_start_index = processed_timepoint_count + 1
                # the very first valid timepoint of a turn where the speaker is the voice_id itself resets accumulative_deviation tracking to 0, rather than continuing from wherever it left off
                pending_accumulative_deviation_reset = include_accumulative_deviation and accumulative_deviation_use_self_tracking_reset and speaker_id == voice_id
                # same self-tracking idea, applied to the signal-rate running totals themselves: a turn where the speaker is the voice_id being tracked starts that voice_id's signal-rate tracking over from 0 instead of carrying over whatever the previous (different) speaker left behind
                pending_signal_rate_reset = use_signal_rate_simulation and speaker_id == voice_id

                for audio_name in audio_list:
                    distribution, timepoint_phonemes = audio_cache[(speaker_id, audio_name)]

                    for timepoint_index in range(len(distribution[0])):
                        voiced_ratio = numpy.sum(distribution[:voiced_frequency_limit_index, timepoint_index])
                        timepoint_is_voiced = voiced_ratio >= Subdistribution_Timepoint_Voiced_Ratio_Minimum and timepoint_phonemes[timepoint_index] is not None
                        if not timepoint_is_voiced and not include_non_voiced_timepoints:
                            continue

                        if use_signal_rate_simulation:
                            if pending_signal_rate_reset:
                                total_distribution_signal_rate = 0.0
                                distribution_ratio_signal_rates = {freq: 0.0 for freq in voiced_frequency_bucket_centers}
                                pending_signal_rate_reset = False

                            # non-voiced timepoints (only reachable here when include_non_voiced_timepoints is True) decay both running totals toward 0 instead of being pulled toward this timepoint's real values, which keeps their ratio — and everything computed from it — unchanged for the duration of a non-voiced stretch
                            signal_rate_second_value = 1.0 if timepoint_is_voiced else 0.0
                            # total_distribution_signal_rate and every bucket's distribution_ratio_signal_rates must be updated for this timepoint before any bucket below can compute its signal_rate_ratio
                            total_distribution_signal_rate = Weighted_Average(total_distribution_signal_rate, occurrence_ratio_cumulation_weight, signal_rate_second_value, 1.0 - occurrence_ratio_cumulation_weight)
                            for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                                signal_rate_timepoint_ratio = float(distribution[freq_index][timepoint_index]) if timepoint_is_voiced else 0.0
                                distribution_ratio_signal_rates[freq_center] = Weighted_Average(
                                    distribution_ratio_signal_rates[freq_center], occurrence_ratio_cumulation_weight,
                                    signal_rate_timepoint_ratio, 1.0 - occurrence_ratio_cumulation_weight
                                )

                        if include_non_voiced_timepoints and total_distribution_signal_rate == 0.0:
                            # no voiced timepoint has been seen yet since the run (or the last sequence/self-turn reset) started, so total_distribution_signal_rate is still exactly 0.0 — dividing by it below would be a divide-by-zero, so this timepoint is still included on the timeline but reported as no-value (NaN) across every tracked list instead of being computed
                            if need_cumulative_comparative_occurrence_ratios:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    cumulative_comparative_occurrence_ratios[null_freq_center].append(math.nan)
                            if include_weighted_binary_match_contribution:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    match_contribution_weights[null_freq_center].append(math.nan)
                                weighted_binary_match_contributions.append(math.nan)
                            if need_occurrence_percentile_deviation_buckets:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    occurrence_percentile_deviations[null_freq_center].append(math.nan)
                            if include_occurrence_percentile_deviation:
                                average_occurrence_percentile_deviations.append(math.nan)
                            if include_occurrence_percentile_inverse_deviation:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    occurrence_percentile_inverse_deviations[null_freq_center].append(math.nan)
                                average_occurrence_percentile_inverse_deviations.append(math.nan)
                            if include_occurrence_percentile_half_distance:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    occurrence_percentile_half_distances[null_freq_center].append(math.nan)
                                average_occurrence_percentile_half_distances.append(math.nan)
                            if include_accumulative_deviation:
                                for null_freq_center in voiced_frequency_bucket_centers:
                                    element_accumulative_deviations[null_freq_center].append(math.nan)
                                average_element_accumulative_deviations.append(math.nan)

                            processed_timepoint_count += 1
                            continue

                        timepoint_weighted_binary_match_contribution_weights = []
                        timepoint_weighted_binary_match_contribution_is_positive = []
                        timepoint_occurrence_percentile_deviations = []
                        timepoint_occurrence_percentile_inverse_deviations = []
                        timepoint_occurrence_percentile_half_distances = []
                        timepoint_element_accumulative_deviations = []

                        for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                            timepoint_ratio = float(distribution[freq_index][timepoint_index])

                            if need_cumulative_comparative_occurrence_ratios or need_percentile_deviation_for_accumulative_deviation:
                                if use_signal_rate_simulation:
                                    signal_rate_ratio = distribution_ratio_signal_rates[freq_center] / total_distribution_signal_rate
                                    if use_bell_curve:
                                        center, lower_standard_deviation, upper_standard_deviation = bell_curve_projections[freq_index]
                                        new_ratio = _Bell_Curve_Value_2(signal_rate_ratio, center, lower_standard_deviation, upper_standard_deviation)
                                    else:
                                        new_ratio = _Lookup_Closest_Value(
                                            inverted_occurrence_ratios[freq_index],
                                            sorted_keys_per_bucket[freq_index],
                                            signal_rate_ratio
                                        )

                                    if need_cumulative_comparative_occurrence_ratios:
                                        cumulative_comparative_occurrence_ratios[freq_center].append(new_ratio)
                                else:
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
                                # occurrence_percentile_deviation's own reported/charted value is negated (0 = best, -1 = worst) to match the "zero is best" y-axis convention of the other occurrence_percentile_* charts; the un-negated deviation is still fed to inverse_deviation/half_distance below, which expect the original 0-to-1 scale
                                negated_deviation = -1.0 * deviation
                                occurrence_percentile_deviations[freq_center].append(negated_deviation)
                                timepoint_occurrence_percentile_deviations.append(negated_deviation)

                                if include_occurrence_percentile_inverse_deviation:
                                    inverse_deviation = _Occurrence_Percentile_Inverse_Deviation(deviation, occurrence_percentile_inverse_deviation_hyperparameters["deviation_power_curve"], occurrence_percentile_inverse_deviation_hyperparameters["inverse_deviation_minimum"])
                                    occurrence_percentile_inverse_deviations[freq_center].append(inverse_deviation)
                                    timepoint_occurrence_percentile_inverse_deviations.append(inverse_deviation)

                                if include_occurrence_percentile_half_distance:
                                    half_distance = _Occurrence_Percentile_Half_Distance(deviation, occurrence_percentile_half_distance_hyperparameters["half_distance_minimum"])
                                    occurrence_percentile_half_distances[freq_center].append(half_distance)
                                    timepoint_occurrence_percentile_half_distances.append(half_distance)

                            if include_accumulative_deviation:
                                current_timepoint_percentile_deviation = _Occurrence_Percentile_Deviation(new_ratio if use_signal_rate_simulation else value_2)
                                if accumulative_deviation_deviation_type == "occurrence_percentile_inverse_deviation":
                                    current_timepoint_deviation = _Occurrence_Percentile_Inverse_Deviation(
                                        current_timepoint_percentile_deviation,
                                        occurrence_percentile_inverse_deviation_hyperparameters["deviation_power_curve"],
                                        occurrence_percentile_inverse_deviation_hyperparameters["inverse_deviation_minimum"]
                                    )
                                elif accumulative_deviation_deviation_type == "occurrence_percentile_half_distance":
                                    current_timepoint_deviation = _Occurrence_Percentile_Half_Distance(
                                        current_timepoint_percentile_deviation,
                                        occurrence_percentile_half_distance_hyperparameters["half_distance_minimum"]
                                    )
                                else:
                                    # occurrence_percentile_deviation is 0 at a perfect match and grows toward 1 at the worst match; negate so it matches every other deviation_type's 0-is-best, negative-is-worse convention
                                    current_timepoint_deviation = -1.0 * current_timepoint_percentile_deviation

                                if not accumulative_deviation_use_non_directional_element_deviations:
                                    if (new_ratio < 0.5) if use_signal_rate_simulation else (timepoint_ratio < bucket_medians[freq_index]):
                                        current_timepoint_deviation *= -1.0

                                previous_element_accumulative_deviation_raw = element_accumulative_deviations[freq_center][-1]
                                # a NaN here can only come from a leading (pre-first-voiced-timepoint) null timepoint under include_non_voiced_timepoints — treat it the same as "no accumulation has happened yet" rather than letting NaN propagate forward forever
                                previous_element_accumulative_deviation = 0.0 if (pending_accumulative_deviation_reset or math.isnan(previous_element_accumulative_deviation_raw)) else previous_element_accumulative_deviation_raw
                                if accumulative_deviation_use_average_element_deviations:
                                    new_element_accumulative_deviation = Weighted_Average(
                                        previous_element_accumulative_deviation, accumulative_deviation_decay_rate,
                                        current_timepoint_deviation, 1.0 - accumulative_deviation_decay_rate
                                    )
                                else:
                                    new_element_accumulative_deviation = (previous_element_accumulative_deviation + current_timepoint_deviation) * accumulative_deviation_decay_rate

                                element_accumulative_deviations[freq_center].append(new_element_accumulative_deviation)
                                timepoint_element_accumulative_deviations.append(new_element_accumulative_deviation)

                        pending_accumulative_deviation_reset = False

                        if include_weighted_binary_match_contribution:
                            # shifted by -1.0 so 0 is the best value and -1 is the worst, matching the "0 is best" convention shared by every other aggregate_match_type
                            total_weight = sum(timepoint_weighted_binary_match_contribution_weights)
                            if total_weight == 0.0:
                                weighted_binary_match_contributions.append(-0.5)
                            else:
                                weighted_binary_match_contributions.append(
                                    (sum(weight for weight, positive in zip(timepoint_weighted_binary_match_contribution_weights, timepoint_weighted_binary_match_contribution_is_positive) if positive) / total_weight) - 1.0
                                )

                        if include_occurrence_percentile_deviation:
                            average_occurrence_percentile_deviations.append(sum(timepoint_occurrence_percentile_deviations) / len(timepoint_occurrence_percentile_deviations) if timepoint_occurrence_percentile_deviations else 0.0)

                        if include_occurrence_percentile_inverse_deviation:
                            average_occurrence_percentile_inverse_deviations.append(sum(timepoint_occurrence_percentile_inverse_deviations) / len(timepoint_occurrence_percentile_inverse_deviations) if timepoint_occurrence_percentile_inverse_deviations else -1.0)

                        if include_occurrence_percentile_half_distance:
                            average_occurrence_percentile_half_distances.append(sum(timepoint_occurrence_percentile_half_distances) / len(timepoint_occurrence_percentile_half_distances) if timepoint_occurrence_percentile_half_distances else 0.0)

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
            if include_accumulative_deviation:
                speaker_data["element_accumulative_deviations"] = element_accumulative_deviations
                speaker_data["average_element_accumulative_deviations"] = average_element_accumulative_deviations
            all_results[sequence_index] = speaker_data

        return all_results

    per_voice_results = {}
    per_voice_ylims = {}
    per_voice_voiced_frequency_bucket_centers = {}

    for voice_id in voice_ids:
        state_path = Json_Directory + f"Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts{Format_Half_Life_For_Filename(voice_profile_half_life)}.json"
        state = Load_Layered_State(state_path)
        if state is None:
            print(f"Element_Match_Contribution_Type_Explorer: no data found for voice_id '{voice_id}', skipping")
            continue

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

        all_results = _Process_Sequences_For_Voice(
            voiced_frequency_bucket_centers, voiced_frequency_limit_index,
            inverted_occurrence_ratios, sorted_keys_per_bucket, bell_curve_projections, bucket_medians
        )

        if need_per_voice_ylims:
            per_voice_ylims[voice_id] = _Compute_Global_Ylims(
                included_variants, all_results, voiced_frequency_bucket_centers,
                weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound,
                accumulative_deviation_hyperparameters, chart_y_minimums
            )

        per_voice_results[voice_id] = all_results
        per_voice_voiced_frequency_bucket_centers[voice_id] = voiced_frequency_bucket_centers

    if not per_voice_results:
        print("Element_Match_Contribution_Type_Explorer: no data found for any voice_id, aborting")
        return

    os.makedirs(Analysis_Directory, exist_ok=True)

    successful_voice_ids = list(per_voice_results.keys())

    if include_per_speaker_overall_chart or include_per_speaker_per_bucket_chart or include_combined_overall_chart:
        for voice_id, all_results in per_voice_results.items():
            overall_ylims, per_bucket_ylims = per_voice_ylims[voice_id]
            voiced_frequency_bucket_centers = per_voice_voiced_frequency_bucket_centers[voice_id]
            if include_per_speaker_overall_chart or include_per_speaker_per_bucket_chart:
                for sequence_index in range(len(comparative_voices_audio_set)):
                    if include_per_speaker_overall_chart:
                        Generate_Per_Speaker_Overall_Chart(voice_id, sequence_index, included_variants, overall_ylims, all_results[sequence_index])
                    if include_per_speaker_per_bucket_chart:
                        Generate_Per_Speaker_Per_Bucket_Chart(voice_id, sequence_index, included_variants, per_bucket_ylims, all_results[sequence_index], voiced_frequency_bucket_centers, weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound)
            if include_combined_overall_chart:
                Generate_Combined_Overall_Chart(voice_id, included_variants, overall_ylims, all_results)

    if include_all_speaker_overall_chart:
        # a shared y-axis scale across every included voice_id keeps all overlaid lines on the all-speaker chart visible and comparable, rather than reusing any single voice_id's own (potentially narrower) scale
        combined_all_voices_results = {}
        combined_index = 0
        for all_results in per_voice_results.values():
            for data in all_results.values():
                combined_all_voices_results[combined_index] = data
                combined_index += 1
        all_speaker_overall_ylims, _ = _Compute_Global_Ylims(
            included_variants, combined_all_voices_results, next(iter(per_voice_voiced_frequency_bucket_centers.values())),
            weighted_binary_match_contribution_lower_bound, weighted_binary_match_contribution_upper_bound,
            accumulative_deviation_hyperparameters, chart_y_minimums
        )

        reference_all_results = per_voice_results[successful_voice_ids[0]]
        for sequence_index in range(len(comparative_voices_audio_set)):
            speaker_segments = reference_all_results[sequence_index]["speaker_segments"]
            Generate_All_Speaker_Overall_Chart(successful_voice_ids, sequence_index, included_variants, all_speaker_overall_ylims, per_voice_results, speaker_segments, metric_inclusions)

    print(f"Element_Match_Contribution_Type_Explorer: exploration complete for voice_ids {successful_voice_ids}")
