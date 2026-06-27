import itertools
import numpy
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import Analysis_Directory, Analysis_Run_Name, Subdistribution_Voiced_Frequency_Limit, Subdistribution_Thresholds, Subdistribution_Display_Type, Chart_Image_Resolution
from Color_Assignment_Manager import Get_Speaker_Color


def Compute_Tier_Differences(tiers_a, tiers_b):
    # returns list of signed per-bucket diff arrays (A - B), one entry per threshold tier
    return [numpy.array(tier_a.Subdistribution) - numpy.array(tier_b.Subdistribution) for tier_a, tier_b in zip(tiers_a, tiers_b)]


def Generate_Pair_Difference_Chart(audio_a, audio_b, tier_differences, frequency_bucket_centers, color_a, color_b):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(Subdistribution_Thresholds) * 100) + 11
    bucket_width = frequency_bucket_centers[0]

    for diff, threshold in zip(tier_differences, Subdistribution_Thresholds):
        pyplot.subplot(subplot_number)
        pyplot.title(f"{audio_a.Audio_File_Name} vs {audio_b.Audio_File_Name} | Threshold {threshold} | L1 {numpy.sum(numpy.abs(diff)):.4f}")
        bar_colors = [color_a if v >= 0 else color_b for v in diff]
        pyplot.bar(frequency_bucket_centers, diff, color=bar_colors, width=bucket_width)
        pyplot.axhline(0, color="black", linewidth=0.8)
        subplot_number += 1

    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + f"subdistribution_diff_{audio_a.Audio_File_Name}_vs_{audio_b.Audio_File_Name}.png", dpi=Chart_Image_Resolution)
    pyplot.close()


def Generate_L1_Summary_Chart(pair_labels, pair_l1_distances, pair_colors):
    pyplot.figure(figsize=(20, 12))
    subplot_number = (len(Subdistribution_Thresholds) * 100) + 11
    x_positions = range(len(pair_labels))

    for tier_index, threshold in enumerate(Subdistribution_Thresholds):
        pyplot.subplot(subplot_number)
        pyplot.title(f"L1 Distance Between Pairs | Threshold {threshold}")
        distances = [pair_l1_distances[pair_index][tier_index] for pair_index in range(len(pair_labels))]
        pyplot.bar(x_positions, distances, color=pair_colors)
        pyplot.xticks(x_positions, pair_labels)
        subplot_number += 1

    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "subdistribution_diff_summary.png", dpi=Chart_Image_Resolution)
    pyplot.close()


def Analyze_Subdistribution_Differences(all_audios_analysis_data):
    if len(all_audios_analysis_data) < 2:
        print("Analyze_Subdistribution_Differences(): fewer than 2 audio files, skipping")
        return all_audios_analysis_data

    pair_labels = []
    pair_l1_distances = []
    pair_colors = []

    for audio_a, audio_b in itertools.combinations(all_audios_analysis_data, 2):
        color_a = Get_Speaker_Color(audio_a.Audio_File_Name)
        color_b = Get_Speaker_Color(audio_b.Audio_File_Name)

        voiced_frequency_limit_index = next((i for i, f in enumerate(audio_a.Frequency_Bucket_Centers) if f > Subdistribution_Voiced_Frequency_Limit), None)
        frequency_bucket_centers = audio_a.Frequency_Bucket_Centers[:voiced_frequency_limit_index]

        tiers_a = audio_a.Typed_Tiered_Subdistributions.Typed_Subdistributions[Subdistribution_Display_Type]
        tiers_b = audio_b.Typed_Tiered_Subdistributions.Typed_Subdistributions[Subdistribution_Display_Type]
        tier_differences = Compute_Tier_Differences(tiers_a, tiers_b)

        Generate_Pair_Difference_Chart(audio_a, audio_b, tier_differences, frequency_bucket_centers, color_a, color_b)

        pair_labels.append(f"{audio_a.Audio_File_Name} vs {audio_b.Audio_File_Name}")
        pair_l1_distances.append([float(numpy.sum(numpy.abs(diff))) for diff in tier_differences])
        pair_colors.append(color_a)
        print(f"Analyze_Subdistribution_Differences() complete for '{audio_a.Audio_File_Name}' vs '{audio_b.Audio_File_Name}'")

    Generate_L1_Summary_Chart(pair_labels, pair_l1_distances, pair_colors)
    return all_audios_analysis_data
