import json
import os

from Global_Hyperparameters import Subdistribution_Display_Colors

COLOR_ASSIGNMENTS_PATH = "tmp/media/output/color_assignments.json"


def Load_Color_Assignments():
    if not os.path.exists(COLOR_ASSIGNMENTS_PATH):
        return {"speakers": {}, "phonemes": {}}
    with open(COLOR_ASSIGNMENTS_PATH, "r") as f:
        return json.load(f)


def Save_Color_Assignments(assignments):
    os.makedirs(os.path.dirname(COLOR_ASSIGNMENTS_PATH), exist_ok=True)
    with open(COLOR_ASSIGNMENTS_PATH, "w") as f:
        json.dump(assignments, f, indent=2)


def Get_Speaker_Color(speaker_id):
    assignments = Load_Color_Assignments()
    if speaker_id not in assignments["speakers"]:
        index = len(assignments["speakers"])
        assignments["speakers"][speaker_id] = Subdistribution_Display_Colors[index % len(Subdistribution_Display_Colors)]
        Save_Color_Assignments(assignments)
    return assignments["speakers"][speaker_id]


def Get_Phoneme_Color(phoneme):
    assignments = Load_Color_Assignments()
    if phoneme not in assignments["phonemes"]:
        index = len(assignments["phonemes"])
        assignments["phonemes"][phoneme] = Subdistribution_Display_Colors[index % len(Subdistribution_Display_Colors)]
        Save_Color_Assignments(assignments)
    return assignments["phonemes"][phoneme]
