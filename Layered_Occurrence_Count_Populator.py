import json
import os

from Global_Hyperparameters import Distribution_Types, Spectrogram_Window_Jump_In_Seconds, Json_Directory
from Spectrogram_Generator import Generate_Audio_Spectrogram
from Frequency_Distribution_Generator import Generate_Frequency_Bucket_Centers, Generate_Typed_Bucketed_Frequency_Progressions, Generate_Typed_Bucketed_Frequency_Distributions
from Subdistribution_Extractor import Accumulate_Frequency_Occurrence_Counts, Convert_Occurrence_Counts_To_Ratios, Extract_Frequency_Subdistributions

PHONEME_CORPUS_DIRECTORY = "../Phoneme_Corpus/data/TRAIN/DR1/"
CORPUS_AUDIO_EXTENSION = ".WAV.wav"
DISTRIBUTION_TYPE = Distribution_Types[0]

VOICED_PHONEMES = [
    "l", "r", "w", "y", "el", "m", "n", "ng", "em", "en", "eng", "nx",
    "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
    "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"
]
VOICED_PHONEMES_SET = set(VOICED_PHONEMES)


# --- path helpers ---

def Get_Universal_State_Path():
    return Json_Directory + "Universal_Frequency_Amount_Occurrence_Counts.json"

def Get_Speaker_State_Path(speaker_id):
    return Json_Directory + f"Speaker_{speaker_id}_Frequency_Amount_Occurrence_Counts.json"

def Get_Phoneme_State_Path(phoneme):
    return Json_Directory + f"Phoneme_{phoneme}_Frequency_Amount_Occurrence_Counts.json"

def Format_Tier_For_Filename(tier):
    return str(tier).replace(".", "")

def Get_Subtractive_Speaker_State_Path(speaker_id, tier):
    return Json_Directory + f"Speaker_{speaker_id}_Subtractive_Frequency_Amount_Occurrence_Counts_{Format_Tier_For_Filename(tier)}.json"

def Get_Subtractive_Phoneme_State_Path(phoneme, tier):
    return Json_Directory + f"Phoneme_{phoneme}_Subtractive_Frequency_Amount_Occurrence_Counts_{Format_Tier_For_Filename(tier)}.json"


# --- state persistence ---

def Load_State(path):
    if not os.path.exists(path):
        return {"processed_audios": {}, "total_voiced_frequency_timepoints_count": 0.0, "frequency_amount_occurrence_counts": None, "frequency_bucket_centers": None}
    with open(path, "r") as f:
        raw = json.load(f)
    raw_counts = raw.get("frequency_amount_occurrence_counts")
    frequency_amount_occurrence_counts = None if raw_counts is None else [{float(k): v for k, v in bucket.items()} for bucket in raw_counts]
    return {
        "processed_audios": raw["processed_audios"],
        "total_voiced_frequency_timepoints_count": float(raw["total_voiced_frequency_timepoints_count"]),
        "frequency_amount_occurrence_counts": frequency_amount_occurrence_counts,
        "frequency_bucket_centers": raw.get("frequency_bucket_centers")
    }


def Save_State(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    counts = state["frequency_amount_occurrence_counts"]
    serializable_counts = None if counts is None else [{str(k): v for k, v in bucket.items()} for bucket in counts]
    with open(path, "w") as f:
        json.dump({
            "processed_audios": state["processed_audios"],
            "total_voiced_frequency_timepoints_count": state["total_voiced_frequency_timepoints_count"],
            "frequency_amount_occurrence_counts": serializable_counts,
            "frequency_bucket_centers": state.get("frequency_bucket_centers")
        }, f, indent=2)


def Is_Already_Processed(state, speaker_id, audio_name):
    return speaker_id in state["processed_audios"] and audio_name in state["processed_audios"][speaker_id]


def Mark_Processed(state, speaker_id, audio_name):
    if speaker_id not in state["processed_audios"]:
        state["processed_audios"][speaker_id] = []
    state["processed_audios"][speaker_id].append(audio_name)


# --- phoneme annotation helpers ---

def Load_Phoneme_Annotations(speaker_directory, audio_name):
    phn_path = speaker_directory + audio_name + ".PHN"
    annotations = []
    with open(phn_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                annotations.append((int(parts[0]), int(parts[1]), parts[2]))
    return annotations


def Map_Timepoints_To_Phonemes(phoneme_annotations, timepoint_count, sample_rate):
    hop_length_samples = round(sample_rate * Spectrogram_Window_Jump_In_Seconds)
    timepoint_phonemes = []
    annotation_index = 0
    for t in range(timepoint_count):
        center_sample = t * hop_length_samples
        while annotation_index < len(phoneme_annotations) and phoneme_annotations[annotation_index][1] <= center_sample:
            annotation_index += 1
        phoneme = None
        if annotation_index < len(phoneme_annotations):
            start, end, label = phoneme_annotations[annotation_index]
            if start <= center_sample < end and label in VOICED_PHONEMES_SET:
                phoneme = label
        timepoint_phonemes.append(phoneme)
    return timepoint_phonemes


# --- shared audio processing ---

def Process_Audio(speaker_id, audio_name):
    speaker_directory = PHONEME_CORPUS_DIRECTORY + speaker_id + "/"
    spectrogram_data = Generate_Audio_Spectrogram(speaker_directory, audio_name, CORPUS_AUDIO_EXTENSION)
    frequency_bucket_centers = Generate_Frequency_Bucket_Centers(spectrogram_data.Frequencies)
    typed_progressions = Generate_Typed_Bucketed_Frequency_Progressions(spectrogram_data.Spectrogram, spectrogram_data.Frequencies, frequency_bucket_centers)
    typed_distributions = Generate_Typed_Bucketed_Frequency_Distributions(typed_progressions)
    distribution = typed_distributions.Typed_Distributions[DISTRIBUTION_TYPE]

    phoneme_annotations = Load_Phoneme_Annotations(speaker_directory, audio_name)
    timepoint_phonemes = Map_Timepoints_To_Phonemes(phoneme_annotations, len(distribution[0]), spectrogram_data.Sample_Rate)

    return distribution, frequency_bucket_centers, timepoint_phonemes


# --- analysis runners ---

def Run_Universal_Analysis(speaker_audio_dict):
    state = Load_State(Get_Universal_State_Path())

    for speaker_id, audio_names in speaker_audio_dict.items():
        for audio_name in audio_names:
            if Is_Already_Processed(state, speaker_id, audio_name):
                print(f"Layered_Occurrence_Count_Populator: skipping '{speaker_id}/{audio_name}' — already in universal state")
                continue

            print(f"Layered_Occurrence_Count_Populator: processing '{speaker_id}/{audio_name}'...")
            distribution, frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
            timepoint_mask = [p is not None for p in timepoint_phonemes]
            if state["frequency_bucket_centers"] is None:
                state["frequency_bucket_centers"] = frequency_bucket_centers

            updated_counts, updated_timepoints_count = Accumulate_Frequency_Occurrence_Counts(
                distribution, frequency_bucket_centers,
                state["frequency_amount_occurrence_counts"],
                state["total_voiced_frequency_timepoints_count"],
                timepoint_mask
            )
            state["frequency_amount_occurrence_counts"] = updated_counts
            state["total_voiced_frequency_timepoints_count"] = updated_timepoints_count
            Mark_Processed(state, speaker_id, audio_name)
            Save_State(Get_Universal_State_Path(), state)
            print(f"Layered_Occurrence_Count_Populator: '{speaker_id}/{audio_name}' complete — total voiced timepoints: {int(state['total_voiced_frequency_timepoints_count'])}")

    print("Layered_Occurrence_Count_Populator: universal run complete")


def Run_Voice_Analysis(speaker_audio_dict):
    speaker_states = {speaker_id: Load_State(Get_Speaker_State_Path(speaker_id)) for speaker_id in speaker_audio_dict}

    for speaker_id, audio_names in speaker_audio_dict.items():
        state = speaker_states[speaker_id]
        for audio_name in audio_names:
            if Is_Already_Processed(state, speaker_id, audio_name):
                print(f"Layered_Occurrence_Count_Populator: skipping '{speaker_id}/{audio_name}' — already in voice state for '{speaker_id}'")
                continue

            print(f"Layered_Occurrence_Count_Populator: processing '{speaker_id}/{audio_name}'...")
            distribution, frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
            timepoint_mask = [p is not None for p in timepoint_phonemes]
            if state["frequency_bucket_centers"] is None:
                state["frequency_bucket_centers"] = frequency_bucket_centers

            updated_counts, updated_timepoints_count = Accumulate_Frequency_Occurrence_Counts(
                distribution, frequency_bucket_centers,
                state["frequency_amount_occurrence_counts"],
                state["total_voiced_frequency_timepoints_count"],
                timepoint_mask
            )
            state["frequency_amount_occurrence_counts"] = updated_counts
            state["total_voiced_frequency_timepoints_count"] = updated_timepoints_count
            Mark_Processed(state, speaker_id, audio_name)
            Save_State(Get_Speaker_State_Path(speaker_id), state)
            print(f"Layered_Occurrence_Count_Populator: '{speaker_id}/{audio_name}' complete — '{speaker_id}' voiced timepoints: {int(state['total_voiced_frequency_timepoints_count'])}")

    print("Layered_Occurrence_Count_Populator: voice run complete")


def Run_Phoneme_Analysis(speaker_audio_dict):
    phoneme_states = {phoneme: Load_State(Get_Phoneme_State_Path(phoneme)) for phoneme in VOICED_PHONEMES}

    for speaker_id, audio_names in speaker_audio_dict.items():
        for audio_name in audio_names:
            if any(Is_Already_Processed(state, speaker_id, audio_name) for state in phoneme_states.values()):
                print(f"Layered_Occurrence_Count_Populator: skipping '{speaker_id}/{audio_name}' — already in phoneme states")
                continue

            print(f"Layered_Occurrence_Count_Populator: processing '{speaker_id}/{audio_name}'...")
            distribution, frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
            for state in phoneme_states.values():
                if state["frequency_bucket_centers"] is None:
                    state["frequency_bucket_centers"] = frequency_bucket_centers

            for phoneme in VOICED_PHONEMES:
                phoneme_mask = [p == phoneme for p in timepoint_phonemes]
                if not any(phoneme_mask):
                    continue
                state = phoneme_states[phoneme]
                updated_counts, updated_timepoints_count = Accumulate_Frequency_Occurrence_Counts(
                    distribution, frequency_bucket_centers,
                    state["frequency_amount_occurrence_counts"],
                    state["total_voiced_frequency_timepoints_count"],
                    phoneme_mask
                )
                state["frequency_amount_occurrence_counts"] = updated_counts
                state["total_voiced_frequency_timepoints_count"] = updated_timepoints_count

            for phoneme, state in phoneme_states.items():
                Mark_Processed(state, speaker_id, audio_name)
                Save_State(Get_Phoneme_State_Path(phoneme), state)

            print(f"Layered_Occurrence_Count_Populator: '{speaker_id}/{audio_name}' complete")

    print("Layered_Occurrence_Count_Populator: phoneme run complete")


# --- entry point ---

def Run_Layered_Occurrence_Count_Population(speaker_audio_dict, subdistribution_layer):
    if subdistribution_layer == "universal":
        Run_Universal_Analysis(speaker_audio_dict)
    elif subdistribution_layer == "voice":
        Run_Voice_Analysis(speaker_audio_dict)
    elif subdistribution_layer == "phoneme":
        Run_Phoneme_Analysis(speaker_audio_dict)
    else:
        print(f"WARNING: Layered_Occurrence_Count_Populator: unrecognized subdistribution_layer '{subdistribution_layer}'")


# --- subtractive offset helpers ---

def Get_Subdistribution_Offsets_From_State(state, tier):
    ratios = Convert_Occurrence_Counts_To_Ratios(state["frequency_amount_occurrence_counts"], state["total_voiced_frequency_timepoints_count"])
    tiers = Extract_Frequency_Subdistributions(ratios)
    matching = next((t for t in tiers if t.Occurrence_Ratio_Threshold == tier), None)
    if matching is None:
        raise ValueError(f"Layered_Occurrence_Count_Populator: threshold {tier} not found in populated tiers. Available: {[t.Occurrence_Ratio_Threshold for t in tiers]}")
    return matching.Subdistribution


def Get_Universal_Subdistribution_Offsets(tier):
    state = Load_State(Get_Universal_State_Path())
    if state["frequency_amount_occurrence_counts"] is None:
        raise ValueError("Layered_Occurrence_Count_Populator: universal state is empty — run universal population first")
    return Get_Subdistribution_Offsets_From_State(state, tier)


def Get_Speaker_Subdistribution_Offsets(speaker_id, tier):
    path = Get_Subtractive_Speaker_State_Path(speaker_id, tier)
    state = Load_State(path)
    if state["frequency_amount_occurrence_counts"] is None:
        raise ValueError(f"Layered_Occurrence_Count_Populator: subtractive voice state for speaker '{speaker_id}' is empty — run subtractive voice population first")
    return Get_Subdistribution_Offsets_From_State(state, tier)


# --- subtractive analysis runners ---

def Run_Subtractive_Voice_Analysis(speaker_audio_dict, subtractive_subdistribution_tier):
    universal_offsets = Get_Universal_Subdistribution_Offsets(subtractive_subdistribution_tier)
    speaker_states = {speaker_id: Load_State(Get_Subtractive_Speaker_State_Path(speaker_id, subtractive_subdistribution_tier)) for speaker_id in speaker_audio_dict}

    for speaker_id, audio_names in speaker_audio_dict.items():
        state = speaker_states[speaker_id]
        for audio_name in audio_names:
            if Is_Already_Processed(state, speaker_id, audio_name):
                print(f"Layered_Occurrence_Count_Populator: skipping '{speaker_id}/{audio_name}' — already in subtractive voice state for '{speaker_id}'")
                continue

            print(f"Layered_Occurrence_Count_Populator: processing '{speaker_id}/{audio_name}'...")
            distribution, frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
            timepoint_mask = [p is not None for p in timepoint_phonemes]
            if state["frequency_bucket_centers"] is None:
                state["frequency_bucket_centers"] = frequency_bucket_centers

            updated_counts, updated_timepoints_count = Accumulate_Frequency_Occurrence_Counts(
                distribution, frequency_bucket_centers,
                state["frequency_amount_occurrence_counts"],
                state["total_voiced_frequency_timepoints_count"],
                timepoint_mask,
                universal_offsets
            )
            state["frequency_amount_occurrence_counts"] = updated_counts
            state["total_voiced_frequency_timepoints_count"] = updated_timepoints_count
            Mark_Processed(state, speaker_id, audio_name)
            Save_State(Get_Subtractive_Speaker_State_Path(speaker_id, subtractive_subdistribution_tier), state)
            print(f"Layered_Occurrence_Count_Populator: '{speaker_id}/{audio_name}' complete — '{speaker_id}' voiced timepoints: {int(state['total_voiced_frequency_timepoints_count'])}")

    print("Layered_Occurrence_Count_Populator: subtractive voice run complete")


def Run_Subtractive_Phoneme_Analysis(speaker_audio_dict, subtractive_subdistribution_tier, subtract_voice_for_phoneme):
    universal_offsets = Get_Universal_Subdistribution_Offsets(subtractive_subdistribution_tier)
    phoneme_states = {phoneme: Load_State(Get_Subtractive_Phoneme_State_Path(phoneme, subtractive_subdistribution_tier)) for phoneme in VOICED_PHONEMES}

    for speaker_id, audio_names in speaker_audio_dict.items():
        if subtract_voice_for_phoneme:
            voice_offsets = Get_Speaker_Subdistribution_Offsets(speaker_id, subtractive_subdistribution_tier)
            combined_offsets = [u + v for u, v in zip(universal_offsets, voice_offsets)]
        else:
            combined_offsets = universal_offsets

        for audio_name in audio_names:
            if any(Is_Already_Processed(state, speaker_id, audio_name) for state in phoneme_states.values()):
                print(f"Layered_Occurrence_Count_Populator: skipping '{speaker_id}/{audio_name}' — already in subtractive phoneme states")
                continue

            print(f"Layered_Occurrence_Count_Populator: processing '{speaker_id}/{audio_name}'...")
            distribution, frequency_bucket_centers, timepoint_phonemes = Process_Audio(speaker_id, audio_name)
            for state in phoneme_states.values():
                if state["frequency_bucket_centers"] is None:
                    state["frequency_bucket_centers"] = frequency_bucket_centers

            for phoneme in VOICED_PHONEMES:
                phoneme_mask = [p == phoneme for p in timepoint_phonemes]
                if not any(phoneme_mask):
                    continue
                state = phoneme_states[phoneme]
                updated_counts, updated_timepoints_count = Accumulate_Frequency_Occurrence_Counts(
                    distribution, frequency_bucket_centers,
                    state["frequency_amount_occurrence_counts"],
                    state["total_voiced_frequency_timepoints_count"],
                    phoneme_mask,
                    combined_offsets
                )
                state["frequency_amount_occurrence_counts"] = updated_counts
                state["total_voiced_frequency_timepoints_count"] = updated_timepoints_count

            for phoneme, state in phoneme_states.items():
                Mark_Processed(state, speaker_id, audio_name)
                Save_State(Get_Subtractive_Phoneme_State_Path(phoneme, subtractive_subdistribution_tier), state)

            print(f"Layered_Occurrence_Count_Populator: '{speaker_id}/{audio_name}' complete")

    print("Layered_Occurrence_Count_Populator: subtractive phoneme run complete")


# --- subtractive entry point ---

def Run_Subtractive_Layered_Occurrence_Count_Population(speaker_audio_dict, subdistribution_layer, subtractive_subdistribution_tier, subtract_voice_for_phoneme=False):
    if subdistribution_layer == "universal":
        Run_Universal_Analysis(speaker_audio_dict)
    elif subdistribution_layer == "voice":
        Run_Subtractive_Voice_Analysis(speaker_audio_dict, subtractive_subdistribution_tier)
    elif subdistribution_layer == "phoneme":
        Run_Subtractive_Phoneme_Analysis(speaker_audio_dict, subtractive_subdistribution_tier, subtract_voice_for_phoneme)
    else:
        print(f"WARNING: Layered_Occurrence_Count_Populator: unrecognized subdistribution_layer '{subdistribution_layer}'")
