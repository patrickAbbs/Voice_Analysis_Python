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
]

_OVERALL_KEYS = {
    "weighted_binary_match_contribution": "weighted_binary_match_contributions",
    "occurrence_percentile_deviation": "average_occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "average_occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "average_occurrence_percentile_half_distances",
    "raw_distance": "average_raw_distances",
}

_PER_BUCKET_KEYS = {
    "weighted_binary_match_contribution": "match_contribution_weights",
    "occurrence_percentile_deviation": "occurrence_percentile_deviations",
    "occurrence_percentile_inverse_deviation": "occurrence_percentile_inverse_deviations",
    "occurrence_percentile_half_distance": "occurrence_percentile_half_distances",
    "raw_distance": "cumulative_raw_distances",
}


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
    projections = []
    for bucket in inverted_occurrence_ratios:
        center = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.5))[0]
        lower_key = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.15865))[0]
        upper_key = min(bucket.items(), key=lambda kv: abs(kv[1] - 0.84135))[0]
        projections.append((center, center - lower_key, upper_key - center))
    return projections


def _Bell_Curve_Value2(ratio, center, lower_sd, upper_sd):
    std = lower_sd if ratio < center else upper_sd
    if std <= 0.0:
        return 0.5
    return 0.5 * (1.0 + math.erf((ratio - center) / (std * _SQRT2)))


def _Extract_Medians(inverted_occurrence_ratios):
    return [min(bucket.items(), key=lambda kv: abs(kv[1] - 0.5))[0] for bucket in inverted_occurrence_ratios]


def _Freq_Colors(voiced_frequency_bucket_centers):
    n = len(voiced_frequency_bucket_centers)
    purple = numpy.array([0.502, 0.0, 0.502])
    orange = numpy.array([1.0, 0.647, 0.0])
    return [
        tuple(purple + (i / (n - 1) if n > 1 else 0.0) * (orange - purple))
        for i in range(n)
    ]


def _Signed_Wbmc_Weight(weight, ratio, lower_bound, upper_bound):
    return weight if lower_bound <= ratio <= upper_bound else -weight


# --- per-bucket variant computations ---

def _Wbmc_Weight(new_ratio, lower_bound, upper_bound, positive_contribution_range, positive_weight_power_curve, negative_weight_proximity_half_distance_increment):
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

def _Compute_Global_Ylims(included_variants, all_results, voiced_frequency_bucket_centers, wbmc_lower_bound, wbmc_upper_bound):
    overall_ylims = {}
    per_bucket_ylims = {}

    for variant in included_variants:
        if variant in ("weighted_binary_match_contribution", "occurrence_percentile_deviation"):
            overall_ylims[variant] = (0.0, 1.0)
        else:
            key = _OVERALL_KEYS[variant]
            global_min = 0.0
            for data in all_results.values():
                for v in data[key]:
                    if v < global_min:
                        global_min = v
            overall_ylims[variant] = (global_min, 0.0)

    for variant in included_variants:
        if variant == "occurrence_percentile_deviation":
            per_bucket_ylims[variant] = (0.0, 1.0)
        elif variant == "weighted_binary_match_contribution":
            abs_max = 0.0
            weights_key = _PER_BUCKET_KEYS["weighted_binary_match_contribution"]
            ccr_key = "cumulative_comparative_occurrence_ratios"
            for data in all_results.values():
                weights = data.get(weights_key, {})
                ccr = data.get(ccr_key, {})
                for freq in voiced_frequency_bucket_centers:
                    w_list = weights.get(freq, [])
                    r_list = ccr.get(freq, [])
                    for i, w in enumerate(w_list):
                        ratio = r_list[i] if i < len(r_list) else 0.5
                        signed = _Signed_Wbmc_Weight(w, ratio, wbmc_lower_bound, wbmc_upper_bound)
                        if abs(signed) > abs_max:
                            abs_max = abs(signed)
            per_bucket_ylims[variant] = (-abs_max, abs_max)
        else:
            key = _PER_BUCKET_KEYS[variant]
            global_min = 0.0
            for data in all_results.values():
                per_bucket = data.get(key, {})
                for freq in voiced_frequency_bucket_centers:
                    for v in per_bucket.get(freq, []):
                        if v < global_min:
                            global_min = v
            per_bucket_ylims[variant] = (global_min, 0.0)

    return overall_ylims, per_bucket_ylims


# --- chart generation ---

def _Make_X_Values(n_points):
    return [i * Spectrogram_Window_Jump_In_Seconds for i in range(n_points)]


def Generate_Per_Speaker_Overall_Chart(voice_id, speaker_id, included_variants, overall_ylims, data):
    n = len(included_variants)
    fig, axes = pyplot.subplots(n, 1, figsize=(20, 5 * n))
    if n == 1:
        axes = [axes]

    color = Get_Speaker_Color(speaker_id)
    first_key = _OVERALL_KEYS[included_variants[0]]
    n_points = len(data[first_key])
    x_values = _Make_X_Values(n_points)

    for ax, variant in zip(axes, included_variants):
        vals = data[_OVERALL_KEYS[variant]]
        bx = _Make_X_Values(len(vals))
        ax.plot(bx, vals, color=color, linewidth=0.75)
        ylim = overall_ylims[variant]
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(0, bx[-1] if bx else 0)
        ax.set_title(f"{variant} | {voice_id} baseline vs {speaker_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(variant)
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_overall_{voice_id}_{speaker_id}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: overall chart saved to '{output_path}'")


def Generate_Per_Speaker_Per_Bucket_Chart(voice_id, speaker_id, included_variants, per_bucket_ylims, data, voiced_frequency_bucket_centers, wbmc_lower_bound, wbmc_upper_bound):
    n = len(included_variants)
    fig, axes = pyplot.subplots(n, 1, figsize=(20, 5 * n))
    if n == 1:
        axes = [axes]

    colors = _Freq_Colors(voiced_frequency_bucket_centers)

    for ax, variant in zip(axes, included_variants):
        per_bucket_key = _PER_BUCKET_KEYS[variant]
        per_bucket = data.get(per_bucket_key, {})

        if variant == "weighted_binary_match_contribution":
            ccr = data.get("cumulative_comparative_occurrence_ratios", {})
            for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                w_list = per_bucket.get(freq_center, [])
                r_list = ccr.get(freq_center, [])
                signed_vals = [
                    _Signed_Wbmc_Weight(w, r_list[i] if i < len(r_list) else 0.5, wbmc_lower_bound, wbmc_upper_bound)
                    for i, w in enumerate(w_list)
                ]
                bx = _Make_X_Values(len(signed_vals))
                ax.plot(bx, signed_vals, color=colors[freq_index], linewidth=0.5)
        else:
            for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                vals = per_bucket.get(freq_center, [])
                bx = _Make_X_Values(len(vals))
                ax.plot(bx, vals, color=colors[freq_index], linewidth=0.5)

        ylim = per_bucket_ylims.get(variant, (-1.0, 1.0))
        ax.set_ylim(ylim[0], ylim[1])
        max_x = max(
            (len(per_bucket.get(freq, [])) - 1) * Spectrogram_Window_Jump_In_Seconds
            for freq in voiced_frequency_bucket_centers
            if per_bucket.get(freq)
        ) if per_bucket else 0
        ax.set_xlim(0, max_x)
        ax.set_title(f"{variant} (per bucket) | {voice_id} baseline vs {speaker_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(variant)
        if ylim[0] < 0:
            ax.axhline(0, color="black", linewidth=0.5)

    pyplot.tight_layout()
    output_path = Analysis_Directory + Analysis_Run_Name + f"_element_match_per_bucket_{voice_id}_{speaker_id}.png"
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()
    print(f"Element_Match_Contribution_Type_Explorer: per-bucket chart saved to '{output_path}'")


def Generate_Combined_Overall_Chart(voice_id, included_variants, overall_ylims, all_results):
    n = len(included_variants)
    fig, axes = pyplot.subplots(n, 1, figsize=(20, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, variant in zip(axes, included_variants):
        key = _OVERALL_KEYS[variant]
        for speaker_id, data in all_results.items():
            vals = data[key]
            bx = _Make_X_Values(len(vals))
            linewidth = 1.5 if speaker_id == voice_id else 0.75
            ax.plot(bx, vals, color=Get_Speaker_Color(speaker_id), linewidth=linewidth, label=speaker_id)
        ylim = overall_ylims[variant]
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_title(f"{variant} — All Speakers vs {voice_id} Baseline")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(variant)
        ax.legend()
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="black", linewidth=0.5)

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

    included_variants = [v for v in VARIANT_ORDER if aggregate_match_types.get(v, {}).get("include_variant", False)]
    if not included_variants:
        print("Element_Match_Contribution_Type_Explorer: no variants included, aborting")
        return

    include_wbmc = "weighted_binary_match_contribution" in included_variants
    include_opd = "occurrence_percentile_deviation" in included_variants
    include_opid = "occurrence_percentile_inverse_deviation" in included_variants
    include_ophd = "occurrence_percentile_half_distance" in included_variants
    include_rd = "raw_distance" in included_variants
    need_ccr = include_wbmc or include_opd or include_opid or include_ophd

    wbmc_hp = aggregate_match_types.get("weighted_binary_match_contribution", {}).get("hyperparameters", {})
    wbmc_lower_bound = wbmc_upper_bound = None
    if include_wbmc:
        pcr = wbmc_hp["positive_contribution_range"]
        wbmc_lower_bound = 0.5 - (pcr * 0.5)
        wbmc_upper_bound = 0.5 + (pcr * 0.5)
        wbmc_pw = wbmc_hp["positive_weight_power_curve"]
        wbmc_nwhdi = wbmc_hp["negative_weight_proximity_half_distance_increment"]

    opid_hp = aggregate_match_types.get("occurrence_percentile_inverse_deviation", {}).get("hyperparameters", {})
    ophd_hp = aggregate_match_types.get("occurrence_percentile_half_distance", {}).get("hyperparameters", {})

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
        bucket_medians = [proj[0] for proj in bell_curve_projections] if include_rd else None
    else:
        bell_curve_projections = None
        sorted_keys_per_bucket = _Build_Sorted_Keys(inverted_occurrence_ratios)
        bucket_medians = _Extract_Medians(inverted_occurrence_ratios) if include_rd else None

    voiced_frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
    voiced_frequency_limit_index = len(voiced_frequency_bucket_centers)

    all_results = {}

    for speaker_id, audio_list in comparative_voices_audio_set.items():
        cumulative_comparative_occurrence_ratios = {freq: [0.5] for freq in voiced_frequency_bucket_centers} if need_ccr else {}
        cumulative_raw_distances = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_rd else {}

        match_contribution_weights = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_wbmc else {}
        weighted_binary_match_contributions = [0.5] if include_wbmc else []

        need_opd_buckets = include_opd or include_opid or include_ophd
        occurrence_percentile_deviations = {freq: [1.0] for freq in voiced_frequency_bucket_centers} if need_opd_buckets else {}
        average_occurrence_percentile_deviations = [1.0] if include_opd else []

        occurrence_percentile_inverse_deviations = {freq: [-1.0] for freq in voiced_frequency_bucket_centers} if include_opid else {}
        average_occurrence_percentile_inverse_deviations = [-1.0] if include_opid else []

        occurrence_percentile_half_distances = {freq: [0.0] for freq in voiced_frequency_bucket_centers} if include_ophd else {}
        average_occurrence_percentile_half_distances = [0.0] if include_ophd else []

        average_raw_distances = [0.0] if include_rd else []

        for audio_name in audio_list:
            print(f"Element_Match_Contribution_Type_Explorer: processing '{speaker_id}/{audio_name}'...")
            distribution, audio_frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)

            for timepoint_index in range(len(distribution[0])):
                voiced_ratio = numpy.sum(distribution[:voiced_frequency_limit_index, timepoint_index])
                if voiced_ratio < Subdistribution_Timepoint_Voiced_Ratio_Minimum:
                    continue
                if timepoint_phonemes[timepoint_index] is None:
                    continue

                tp_wbmc_weights = []
                tp_wbmc_is_positive = []
                tp_opd = []
                tp_opid = []
                tp_ophd = []
                tp_rd = []

                for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
                    timepoint_ratio = float(distribution[freq_index][timepoint_index])

                    if need_ccr:
                        if use_bell_curve:
                            center, lower_sd, upper_sd = bell_curve_projections[freq_index]
                            value_2 = _Bell_Curve_Value2(timepoint_ratio, center, lower_sd, upper_sd)
                        else:
                            value_2 = _Lookup_Closest_Value(
                                inverted_occurrence_ratios[freq_index],
                                sorted_keys_per_bucket[freq_index],
                                timepoint_ratio
                            )
                        prev = cumulative_comparative_occurrence_ratios[freq_center][-1]
                        new_ratio = Weighted_Average(prev, occurrence_ratio_cumulation_weight, value_2, 1.0 - occurrence_ratio_cumulation_weight)
                        cumulative_comparative_occurrence_ratios[freq_center].append(new_ratio)

                    if include_wbmc:
                        w = _Wbmc_Weight(new_ratio, wbmc_lower_bound, wbmc_upper_bound, pcr, wbmc_pw, wbmc_nwhdi)
                        match_contribution_weights[freq_center].append(w)
                        tp_wbmc_weights.append(w)
                        tp_wbmc_is_positive.append(wbmc_lower_bound <= new_ratio <= wbmc_upper_bound)

                    if need_opd_buckets:
                        deviation = _Occurrence_Percentile_Deviation(new_ratio)
                        occurrence_percentile_deviations[freq_center].append(deviation)
                        tp_opd.append(deviation)

                        if include_opid:
                            inv_dev = _Occurrence_Percentile_Inverse_Deviation(deviation, opid_hp["deviation_power_curve"], opid_hp["inverse_deviation_minimum"])
                            occurrence_percentile_inverse_deviations[freq_center].append(inv_dev)
                            tp_opid.append(inv_dev)

                        if include_ophd:
                            hd = _Occurrence_Percentile_Half_Distance(deviation, ophd_hp["half_distance_minimum"])
                            occurrence_percentile_half_distances[freq_center].append(hd)
                            tp_ophd.append(hd)

                    if include_rd:
                        rd_value_2 = -1.0 * abs(bucket_medians[freq_index] - timepoint_ratio)
                        prev_rd = cumulative_raw_distances[freq_center][-1]
                        new_rd = Weighted_Average(prev_rd, occurrence_ratio_cumulation_weight, rd_value_2, 1.0 - occurrence_ratio_cumulation_weight)
                        cumulative_raw_distances[freq_center].append(new_rd)
                        tp_rd.append(new_rd)

                if include_wbmc:
                    total_w = sum(tp_wbmc_weights)
                    if total_w == 0.0:
                        weighted_binary_match_contributions.append(0.5)
                    else:
                        weighted_binary_match_contributions.append(
                            sum(w for w, pos in zip(tp_wbmc_weights, tp_wbmc_is_positive) if pos) / total_w
                        )

                if include_opd:
                    average_occurrence_percentile_deviations.append(sum(tp_opd) / len(tp_opd) if tp_opd else 1.0)

                if include_opid:
                    average_occurrence_percentile_inverse_deviations.append(sum(tp_opid) / len(tp_opid) if tp_opid else -1.0)

                if include_ophd:
                    average_occurrence_percentile_half_distances.append(sum(tp_ophd) / len(tp_ophd) if tp_ophd else 0.0)

                if include_rd:
                    average_raw_distances.append(sum(tp_rd) / len(tp_rd) if tp_rd else 0.0)

        speaker_data = {}
        if need_ccr:
            speaker_data["cumulative_comparative_occurrence_ratios"] = cumulative_comparative_occurrence_ratios
        if include_wbmc:
            speaker_data["match_contribution_weights"] = match_contribution_weights
            speaker_data["weighted_binary_match_contributions"] = weighted_binary_match_contributions
        if need_opd_buckets:
            speaker_data["occurrence_percentile_deviations"] = occurrence_percentile_deviations
        if include_opd:
            speaker_data["average_occurrence_percentile_deviations"] = average_occurrence_percentile_deviations
        if include_opid:
            speaker_data["occurrence_percentile_inverse_deviations"] = occurrence_percentile_inverse_deviations
            speaker_data["average_occurrence_percentile_inverse_deviations"] = average_occurrence_percentile_inverse_deviations
        if include_ophd:
            speaker_data["occurrence_percentile_half_distances"] = occurrence_percentile_half_distances
            speaker_data["average_occurrence_percentile_half_distances"] = average_occurrence_percentile_half_distances
        if include_rd:
            speaker_data["cumulative_raw_distances"] = cumulative_raw_distances
            speaker_data["average_raw_distances"] = average_raw_distances
        all_results[speaker_id] = speaker_data

    overall_ylims, per_bucket_ylims = _Compute_Global_Ylims(
        included_variants, all_results, voiced_frequency_bucket_centers,
        wbmc_lower_bound, wbmc_upper_bound
    )

    for speaker_id in comparative_voices_audio_set:
        Generate_Per_Speaker_Overall_Chart(voice_id, speaker_id, included_variants, overall_ylims, all_results[speaker_id])
        Generate_Per_Speaker_Per_Bucket_Chart(voice_id, speaker_id, included_variants, per_bucket_ylims, all_results[speaker_id], voiced_frequency_bucket_centers, wbmc_lower_bound, wbmc_upper_bound)

    Generate_Combined_Overall_Chart(voice_id, included_variants, overall_ylims, all_results)
    print(f"Element_Match_Contribution_Type_Explorer: exploration complete for voice_id '{voice_id}'")
