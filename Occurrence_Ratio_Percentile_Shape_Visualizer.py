import numpy
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import Analysis_Directory, Analysis_Run_Name, Chart_Image_Resolution, Json_Directory
from Layered_Subdistribution_Generator import Load_Layered_State, Get_Voiced_Frequency_Bucket_Centers
from Layered_Occurrence_Count_Populator import Format_Half_Life_For_Filename


def _Reconstruct_Sorted_Frequency_Ratios(bucket_counts):
    # Sort descending by frequency_ratio. The occurrence counts are cumulative (each key's count
    # includes all timepoints with ratio <= that key), so the gap between adjacent entries is the
    # number of timepoints where exactly that ratio was observed.
    sorted_pairs = sorted(bucket_counts.items(), key=lambda kv: kv[0], reverse=True)
    result = []
    prev_count = 0
    for frequency_ratio, occurrence_count in sorted_pairs:
        gap = int(occurrence_count) - prev_count
        if gap > 0:
            result.extend([frequency_ratio] * gap)
        prev_count = int(occurrence_count)
    return result


def _Freq_Colors(n):
    purple = numpy.array([0.502, 0.0, 0.502])
    orange = numpy.array([1.0, 0.647, 0.0])
    return [
        tuple(purple + (i / (n - 1) if n > 1 else 0.0) * (orange - purple))
        for i in range(n)
    ]


def Visualize_Occurrence_Ratio_Percentile_Shapes(voice_ids, proximity_density_distance=0.001, voice_profile_cumulation_half_life=None):
    half_life_suffix = Format_Half_Life_For_Filename(voice_profile_cumulation_half_life)
    for voice_id in voice_ids:
        state_path = Json_Directory + f"Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts{half_life_suffix}.json"
        state = Load_Layered_State(state_path)
        if state is None:
            print(f"Occurrence_Ratio_Percentile_Shape_Visualizer: no data found for '{voice_id}', skipping")
            continue

        voiced_frequency_timepoints_count = state["total_voiced_frequency_timepoints_count"]
        voiced_frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
        n_freqs = len(voiced_frequency_bucket_centers)
        colors = _Freq_Colors(n_freqs)

        per_bucket_sorted_ratios = []
        for bucket_counts in state["frequency_amount_occurrence_counts"]:
            sorted_ratios = _Reconstruct_Sorted_Frequency_Ratios(bucket_counts)
            if len(sorted_ratios) != int(voiced_frequency_timepoints_count):
                print(f"Occurrence_Ratio_Percentile_Shape_Visualizer: WARNING — bucket length {len(sorted_ratios)} != voiced_timepoints_count {int(voiced_frequency_timepoints_count)} for '{voice_id}'")
            per_bucket_sorted_ratios.append(sorted_ratios)

        all_values = [v for bucket in per_bucket_sorted_ratios for v in bucket]
        y_min = min(all_values) if all_values else 0.0
        y_max = max(all_values) if all_values else 1.0

        x_values = [i / voiced_frequency_timepoints_count for i in range(int(voiced_frequency_timepoints_count))]

        # Compute min-max normalized version of each bucket's sorted ratios (independent per bucket)
        per_bucket_normalized = []
        for sorted_ratios in per_bucket_sorted_ratios:
            bucket_min = min(sorted_ratios) if sorted_ratios else 0.0
            bucket_max = max(sorted_ratios) if sorted_ratios else 1.0
            span = bucket_max - bucket_min
            if span == 0.0:
                per_bucket_normalized.append([0.0] * len(sorted_ratios))
            else:
                per_bucket_normalized.append([(v - bucket_min) / span for v in sorted_ratios])

        # Compute proximity density for each bucket.
        # For each datapoint, count how many OTHER datapoints fall within proximity_density_distance,
        # then normalize so all counts for that bucket sum to 1.0.
        # The sorted ascending array allows O(log n) neighbor counting via searchsorted.
        per_bucket_proximity_density = []
        per_bucket_proximity_x = []
        for sorted_ratios in per_bucket_sorted_ratios:
            ratios_asc = numpy.array(sorted(sorted_ratios))
            lo_indices = numpy.searchsorted(ratios_asc, ratios_asc - proximity_density_distance, side="left")
            hi_indices = numpy.searchsorted(ratios_asc, ratios_asc + proximity_density_distance, side="right")
            counts = (hi_indices - lo_indices - 1).astype(float)  # -1 to exclude self
            total = counts.sum()
            normalized_counts = counts / total if total > 0.0 else counts
            per_bucket_proximity_density.append(normalized_counts)
            per_bucket_proximity_x.append(ratios_asc)

        prox_y_max = max((v for bucket in per_bucket_proximity_density for v in bucket), default=1.0)

        fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1, figsize=(20, 24))

        for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
            sorted_ratios = per_bucket_sorted_ratios[freq_index]
            ax1.plot(x_values[:len(sorted_ratios)], sorted_ratios, color=colors[freq_index], linewidth=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"Frequency Ratio Percentile Shapes — Raw | {voice_id}")
        ax1.set_xlabel("Percentile (0 = highest observed ratio, 1 = lowest)")
        ax1.set_ylabel("Frequency Ratio")

        for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
            normalized = per_bucket_normalized[freq_index]
            ax2.plot(x_values[:len(normalized)], normalized, color=colors[freq_index], linewidth=0.5)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title(f"Frequency Ratio Percentile Shapes — Min-Max Normalized per Bucket | {voice_id}")
        ax2.set_xlabel("Percentile (0 = highest observed ratio, 1 = lowest)")
        ax2.set_ylabel("Normalized Ratio")

        for freq_index, freq_center in enumerate(voiced_frequency_bucket_centers):
            ax3.plot(per_bucket_proximity_x[freq_index], per_bucket_proximity_density[freq_index], color=colors[freq_index], linewidth=0.5)
        ax3.set_xlim(y_min, y_max)
        ax3.set_ylim(0, prox_y_max)
        ax3.set_title(f"Frequency Ratio Proximity Density (distance={proximity_density_distance}, sum-normalized per bucket) | {voice_id}")
        ax3.set_xlabel("Frequency Ratio")
        ax3.set_ylabel("Normalized Proximity Count")

        pyplot.tight_layout()
        output_path = Analysis_Directory + Analysis_Run_Name + f"_occurrence_ratio_percentile_shapes_{voice_id}.png"
        pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
        pyplot.close()
        print(f"Occurrence_Ratio_Percentile_Shape_Visualizer: chart saved to '{output_path}'")
