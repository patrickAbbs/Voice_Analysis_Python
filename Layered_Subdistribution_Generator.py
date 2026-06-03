import json
import os
import matplotlib.pyplot as pyplot

from Global_Hyperparameters import Analysis_Directory, Analysis_Run_Name, Chart_Image_Resolution, Subdistribution_Display_Colors
from Subdistribution_Extractor import Convert_Occurrence_Counts_To_Ratios, Extract_Frequency_Subdistributions, Subdistribution_Tier

OUTPUT_DIRECTORY = "tmp/media/output/"


# --- state loading ---

def Load_Layered_State(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    raw_counts = raw.get("frequency_amount_occurrence_counts")
    if raw_counts is None or raw["total_voiced_frequency_timepoints_count"] == 0:
        return None
    return {
        "total_voiced_frequency_timepoints_count": float(raw["total_voiced_frequency_timepoints_count"]),
        "frequency_amount_occurrence_counts": [{float(k): v for k, v in bucket.items()} for bucket in raw_counts],
        "frequency_bucket_centers": raw.get("frequency_bucket_centers")
    }


def Get_Tiered_Subdistributions(state):
    frequency_amount_occurrence_ratios = Convert_Occurrence_Counts_To_Ratios(
        state["frequency_amount_occurrence_counts"],
        state["total_voiced_frequency_timepoints_count"]
    )
    return Extract_Frequency_Subdistributions(frequency_amount_occurrence_ratios)


def Get_Voiced_Frequency_Bucket_Centers(state):
    return state["frequency_bucket_centers"][:len(state["frequency_amount_occurrence_counts"])]


# --- subdistribution computation ---

def Compute_Subtractive_Tiers(subject_tiers, universal_tiers, allow_negative):
    if len(subject_tiers) != len(universal_tiers):
        print(f"WARNING: Layered_Subdistribution_Generator: subject and universal tier counts differ ({len(subject_tiers)} vs {len(universal_tiers)}), skipping subtraction")
        return subject_tiers
    subtractive_tiers = []
    for subject_tier, universal_tier in zip(subject_tiers, universal_tiers):
        if len(subject_tier.Subdistribution) != len(universal_tier.Subdistribution):
            print(f"WARNING: Layered_Subdistribution_Generator: bucket count mismatch at threshold {subject_tier.Occurrence_Ratio_Threshold}, skipping subtraction for this tier")
            subtractive_tiers.append(subject_tier)
            continue
        values = [subject_val - universal_val for subject_val, universal_val in zip(subject_tier.Subdistribution, universal_tier.Subdistribution)]
        if not allow_negative:
            values = [max(0.0, v) for v in values]
        subtractive_tiers.append(Subdistribution_Tier(subject_tier.Occurrence_Ratio_Threshold, values))
    return subtractive_tiers


# --- chart generation ---

def Generate_Layered_Subdistribution_Chart(label, subdistribution_tiers, frequency_bucket_centers, output_path, color):
    all_values = [v for tier in subdistribution_tiers for v in tier.Subdistribution]
    y_max = max(all_values) if all_values else 0.0
    y_min = min(min(all_values) if all_values else 0.0, 0.0)
    bucket_width = frequency_bucket_centers[0] if frequency_bucket_centers else 1.0

    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(subdistribution_tiers) * 100) + 11
    for tier in subdistribution_tiers:
        pyplot.subplot(subplot_number)
        pyplot.title(f"{label} | Threshold {tier.Occurrence_Ratio_Threshold} | Sum {tier.Subdistribution_Sum:.4f}")
        pyplot.ylim(y_min, y_max)
        pyplot.bar(frequency_bucket_centers, tier.Subdistribution, color=color, width=bucket_width)
        if y_min < 0:
            pyplot.axhline(0, color="black", linewidth=0.5)
        subplot_number += 1
    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=Chart_Image_Resolution)
    pyplot.close()


# --- analysis runners ---

def Run_Universal_Generation():
    path = OUTPUT_DIRECTORY + "Universal_Frequency_Amount_Occurrence_Counts.json"
    state = Load_Layered_State(path)
    if state is None:
        print(f"Layered_Subdistribution_Generator: no data at '{path}', skipping universal generation")
        return

    subdistribution_tiers = Get_Tiered_Subdistributions(state)
    frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
    output_path = Analysis_Directory + Analysis_Run_Name + "_universal_subdistributions.png"
    Generate_Layered_Subdistribution_Chart("Universal", subdistribution_tiers, frequency_bucket_centers, output_path, Subdistribution_Display_Colors[0])
    print(f"Layered_Subdistribution_Generator: universal chart saved to '{output_path}'")


def Run_Voice_Generation(voice_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts):
    universal_state = Load_Layered_State(OUTPUT_DIRECTORY + "Universal_Frequency_Amount_Occurrence_Counts.json")
    if universal_state is None:
        print("Layered_Subdistribution_Generator: no universal data found, cannot run voice generation")
        return
    universal_tiers = Get_Tiered_Subdistributions(universal_state)

    for voice_index, speaker_id in enumerate(voice_set):
        path = OUTPUT_DIRECTORY + f"Speaker_{speaker_id}_Frequency_Amount_Occurrence_Counts.json"
        state = Load_Layered_State(path)
        if state is None:
            print(f"Layered_Subdistribution_Generator: no data for speaker '{speaker_id}', skipping")
            continue

        voice_tiers = Get_Tiered_Subdistributions(state)
        frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
        color = Subdistribution_Display_Colors[voice_index % len(Subdistribution_Display_Colors)]

        if generate_original_subdistribution_charts:
            output_path = Analysis_Directory + Analysis_Run_Name + f"_voice_original_subdistributions_{speaker_id}.png"
            Generate_Layered_Subdistribution_Chart(f"Voice {speaker_id} (original)", voice_tiers, frequency_bucket_centers, output_path, color)
            print(f"Layered_Subdistribution_Generator: voice original chart saved for '{speaker_id}'")

        if generate_subtractive_subdistribution_charts:
            subtractive_tiers = Compute_Subtractive_Tiers(voice_tiers, universal_tiers, allow_negative_subtractive_subdistributions)
            output_path = Analysis_Directory + Analysis_Run_Name + f"_voice_subtractive_subdistributions_{speaker_id}.png"
            Generate_Layered_Subdistribution_Chart(f"Voice {speaker_id} (subtractive)", subtractive_tiers, frequency_bucket_centers, output_path, color)
            print(f"Layered_Subdistribution_Generator: voice subtractive chart saved for '{speaker_id}'")

    print("Layered_Subdistribution_Generator: voice generation complete")


def Run_Phoneme_Generation(phoneme_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts):
    universal_state = Load_Layered_State(OUTPUT_DIRECTORY + "Universal_Frequency_Amount_Occurrence_Counts.json")
    if universal_state is None:
        print("Layered_Subdistribution_Generator: no universal data found, cannot run phoneme generation")
        return
    universal_tiers = Get_Tiered_Subdistributions(universal_state)

    for phoneme_index, phoneme in enumerate(phoneme_set):
        path = OUTPUT_DIRECTORY + f"Phoneme_{phoneme}_Frequency_Amount_Occurrence_Counts.json"
        state = Load_Layered_State(path)
        if state is None:
            print(f"Layered_Subdistribution_Generator: no data for phoneme '{phoneme}', skipping")
            continue

        phoneme_tiers = Get_Tiered_Subdistributions(state)
        frequency_bucket_centers = Get_Voiced_Frequency_Bucket_Centers(state)
        color = Subdistribution_Display_Colors[phoneme_index % len(Subdistribution_Display_Colors)]

        if generate_original_subdistribution_charts:
            output_path = Analysis_Directory + Analysis_Run_Name + f"_phoneme_original_subdistributions_{phoneme}.png"
            Generate_Layered_Subdistribution_Chart(f"Phoneme {phoneme} (original)", phoneme_tiers, frequency_bucket_centers, output_path, color)
            print(f"Layered_Subdistribution_Generator: phoneme original chart saved for '{phoneme}'")

        if generate_subtractive_subdistribution_charts:
            subtractive_tiers = Compute_Subtractive_Tiers(phoneme_tiers, universal_tiers, allow_negative_subtractive_subdistributions)
            output_path = Analysis_Directory + Analysis_Run_Name + f"_phoneme_subtractive_subdistributions_{phoneme}.png"
            Generate_Layered_Subdistribution_Chart(f"Phoneme {phoneme} (subtractive)", subtractive_tiers, frequency_bucket_centers, output_path, color)
            print(f"Layered_Subdistribution_Generator: phoneme subtractive chart saved for '{phoneme}'")

    print("Layered_Subdistribution_Generator: phoneme generation complete")


# --- entry point ---

def Run_Layered_Subdistribution_Generation(
    subdistribution_layer,
    voice_set=None,
    phoneme_set=None,
    allow_negative_subtractive_subdistributions=False,
    generate_original_subdistribution_charts=True,
    generate_subtractive_subdistribution_charts=True
):
    if subdistribution_layer == "universal":
        Run_Universal_Generation()
    elif subdistribution_layer == "voice":
        Run_Voice_Generation(voice_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts)
    elif subdistribution_layer == "phoneme":
        Run_Phoneme_Generation(phoneme_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts)
    else:
        print(f"WARNING: Layered_Subdistribution_Generator: unrecognized subdistribution_layer '{subdistribution_layer}'")
