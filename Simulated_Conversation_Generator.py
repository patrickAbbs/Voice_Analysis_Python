import json
import os
import random

import soundfile

from Global_Hyperparameters import Phoneme_Corpus_Directory
from Layered_Occurrence_Count_Populator import CORPUS_AUDIO_EXTENSION

Conversation_Sequence_Json_Directory = "tmp/media/conversation_sequence_json/"


# --- seeded duration generation ---

def _Compute_Seeded_Duration(duration_seeds):
    base_duration = duration_seeds["base_duration"]
    deviation_ratio = duration_seeds["deviation_ratio"]
    power_curve = duration_seeds["power_curve"]

    base_duration_multiplier = ((random.random() ** power_curve) * (1 - deviation_ratio)) + deviation_ratio
    if random.random() < 0.5:
        base_duration_multiplier = 1.0 / base_duration_multiplier

    return base_duration_multiplier * base_duration


# --- speaker/audio selection ---

def _Get_Speaker_Audio_Names(speaker_id, speaker_audio_names_cache):
    if speaker_id not in speaker_audio_names_cache:
        speaker_directory = Phoneme_Corpus_Directory + speaker_id + "/"
        speaker_audio_names_cache[speaker_id] = sorted(
            file_name[:-len(CORPUS_AUDIO_EXTENSION)]
            for file_name in os.listdir(speaker_directory)
            if file_name.endswith(CORPUS_AUDIO_EXTENSION)
        )
    return speaker_audio_names_cache[speaker_id]


def _Get_Audio_Duration(speaker_id, audio_name, audio_duration_cache):
    cache_key = (speaker_id, audio_name)
    if cache_key not in audio_duration_cache:
        audio_path = Phoneme_Corpus_Directory + speaker_id + "/" + audio_name + CORPUS_AUDIO_EXTENSION
        audio_duration_cache[cache_key] = soundfile.info(audio_path).duration
    return audio_duration_cache[cache_key]


def _Select_Speaker(speaker_weights, previous_speaker_id):
    candidate_speaker_ids = [speaker_id for speaker_id in speaker_weights if speaker_id != previous_speaker_id]
    if not candidate_speaker_ids:
        # only one speaker is available overall (or previous speaker filled the whole pool) — allow a repeat rather than erroring
        candidate_speaker_ids = list(speaker_weights.keys())
    candidate_weights = [speaker_weights[speaker_id] for speaker_id in candidate_speaker_ids]
    return random.choices(candidate_speaker_ids, weights=candidate_weights, k=1)[0]


def _Select_Audio(speaker_id, speaker_audio_names_cache, conversation_selected_audios):
    available_audio_names = _Get_Speaker_Audio_Names(speaker_id, speaker_audio_names_cache)
    selected_audio_names = conversation_selected_audios.setdefault(speaker_id, set())

    if len(selected_audio_names) >= len(available_audio_names):
        selected_audio_names.clear()

    candidate_audio_names = [audio_name for audio_name in available_audio_names if audio_name not in selected_audio_names]
    audio_name = random.choice(candidate_audio_names)
    selected_audio_names.add(audio_name)
    return audio_name


# --- conversation generation ---

def _Generate_Conversation(speaker_weights, turn_duration_seeds, conversation_duration_seeds, speaker_audio_names_cache, audio_duration_cache):
    conversation_duration = _Compute_Seeded_Duration(conversation_duration_seeds)

    conversation_selected_audios = {}
    turns = []
    total_duration = 0.0
    previous_speaker_id = None

    while total_duration < conversation_duration:
        speaker_id = _Select_Speaker(speaker_weights, previous_speaker_id)
        turn_duration_threshold = _Compute_Seeded_Duration(turn_duration_seeds)

        turn_audio_names = []
        turn_elapsed_duration = 0.0
        while turn_elapsed_duration < turn_duration_threshold and total_duration < conversation_duration:
            audio_name = _Select_Audio(speaker_id, speaker_audio_names_cache, conversation_selected_audios)
            audio_duration = _Get_Audio_Duration(speaker_id, audio_name, audio_duration_cache)

            turn_audio_names.append(audio_name)
            turn_elapsed_duration += audio_duration
            total_duration += audio_duration

        turns.append((speaker_id, turn_audio_names))
        previous_speaker_id = speaker_id

    return turns


# --- output file naming ---

def _Format_Value_For_Filename(value):
    return str(value).replace(".", "o")


def _Build_Output_File_Name(speaker_weights, conversation_duration_seeds):
    ordered_speaker_ids = sorted(speaker_weights, key=lambda speaker_id: speaker_weights[speaker_id], reverse=True)
    speaker_segments = [f"{speaker_id}_{_Format_Value_For_Filename(speaker_weights[speaker_id])}" for speaker_id in ordered_speaker_ids]
    return "_".join(speaker_segments) + f"_{_Format_Value_For_Filename(conversation_duration_seeds['base_duration'])}.json"


# --- entry point ---

def Generate_Simulated_Conversation_Set(conversation_count, speaker_weights, turn_duration_seeds, conversation_duration_seeds):
    speaker_audio_names_cache = {}
    audio_duration_cache = {}

    conversations = []
    for conversation_index in range(conversation_count):
        turns = _Generate_Conversation(speaker_weights, turn_duration_seeds, conversation_duration_seeds, speaker_audio_names_cache, audio_duration_cache)
        conversations.append(turns)
        print(f"Simulated_Conversation_Generator: conversation {conversation_index + 1}/{conversation_count} complete — {len(turns)} turns")

    os.makedirs(Conversation_Sequence_Json_Directory, exist_ok=True)
    output_path = Conversation_Sequence_Json_Directory + _Build_Output_File_Name(speaker_weights, conversation_duration_seeds)

    with open(output_path, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"Simulated_Conversation_Generator: saved {conversation_count} conversations to '{output_path}'")
    return conversations
