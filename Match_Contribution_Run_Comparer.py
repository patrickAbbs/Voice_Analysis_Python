import json
import os

from Global_Hyperparameters import Analysis_Directory_Path, Analysis_Directory_Folder


_RUN_CONFIGURATION_FILENAME = "run_configuration.json"

# the four Run_Element_Match_Contribution_Type_Exploration() parameters that get compared/overwritten wholesale on every run, as opposed to voice_ids/conversation_json_file_name which get appended to a running log instead
_COMPARED_CONFIGURATION_KEYS = ("aggregate_match_types", "cross_type_hyperparameters", "chart_type_inclusions", "metric_inclusions")

# sentinel distinguishing "key absent on this side of the comparison" from any real stored value (including None)
_MISSING = object()


# --- generic nested-dict diffing, shared by the stored-vs-new confirmation prompt and Compare_Match_Contribution_Runs ---

def _All_Are_Dicts_Or_Missing(values):
    return all(value is _MISSING or isinstance(value, dict) for value in values) and any(isinstance(value, dict) for value in values)


def _Find_Differing_Leaves(run_values, path):
    if _All_Are_Dicts_Or_Missing(run_values):
        # union of child keys, in first-seen order across the runs (so output order tracks comparison_run_set/stored-then-new order)
        ordered_keys = []
        seen_keys = set()
        for value in run_values:
            if isinstance(value, dict):
                for key in value.keys():
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ordered_keys.append(key)

        differing_leaves = []
        for key in ordered_keys:
            child_values = [(value.get(key, _MISSING) if isinstance(value, dict) else _MISSING) for value in run_values]
            differing_leaves.extend(_Find_Differing_Leaves(child_values, path + [key]))
        return differing_leaves

    first_value = run_values[0]
    if all(value == first_value for value in run_values[1:]):
        return []
    return [(path, run_values)]


def _Format_Diff_Value(value):
    return "<not set>" if value is _MISSING else str(value)


def _Print_Diff_Tree(run_labels, differing_leaves):
    if not differing_leaves:
        return

    print(" | ".join(run_labels))
    print()

    formatted_entries = []
    global_value_width = 0
    for path, values in differing_leaves:
        formatted_values = [_Format_Diff_Value(value) for value in values]
        formatted_entries.append((path, formatted_values))
        global_value_width = max(global_value_width, max(len(value) for value in formatted_values))

    max_leaf_depth = max(len(path) - 1 for path, _ in differing_leaves)
    value_line_indent = "    " * (max_leaf_depth + 1)

    previous_path = []
    for index, (path, formatted_values) in enumerate(formatted_entries):
        common_prefix_length = 0
        while (
            common_prefix_length < len(previous_path)
            and common_prefix_length < len(path)
            and previous_path[common_prefix_length] == path[common_prefix_length]
        ):
            common_prefix_length += 1

        if common_prefix_length == 0 and index > 0:
            print()

        for depth in range(common_prefix_length, len(path)):
            is_leaf_key = depth == len(path) - 1
            print(("    " * depth) + str(path[depth]) + (":" if is_leaf_key else ""))

        padded_values = [value.ljust(global_value_width) for value in formatted_values]
        print(value_line_indent + " | ".join(padded_values))

        previous_path = path

    print()


def _Confirm(question):
    response = input(f"{question} [y/N]: ").strip().lower()
    return response in ("y", "yes")


# --- entry point 1: called by Run_Element_Match_Contribution_Type_Exploration() to log/reconcile each run's configuration ---

def Record_Run_Configuration(voice_ids, conversation_json_file_name, aggregate_match_types, cross_type_hyperparameters, chart_type_inclusions, metric_inclusions):
    analysis_directory = Analysis_Directory_Path + Analysis_Directory_Folder + "/"
    os.makedirs(analysis_directory, exist_ok=True)
    configuration_path = analysis_directory + _RUN_CONFIGURATION_FILENAME

    new_configuration_values = {
        "aggregate_match_types": aggregate_match_types,
        "cross_type_hyperparameters": cross_type_hyperparameters,
        "chart_type_inclusions": chart_type_inclusions,
        "metric_inclusions": metric_inclusions,
    }

    stored_configuration = None
    if os.path.isfile(configuration_path):
        with open(configuration_path, "r") as f:
            stored_configuration = json.load(f)

    stored_pairings = stored_configuration.get("voice_id_conversation_pairings", []) if stored_configuration is not None else []

    if stored_configuration is not None:
        differing_leaves = []
        for key in _COMPARED_CONFIGURATION_KEYS:
            differing_leaves.extend(_Find_Differing_Leaves([stored_configuration.get(key, _MISSING), new_configuration_values[key]], [key]))

        if differing_leaves:
            print(f"Match_Contribution_Run_Comparer: WARNING - this run's configuration differs from the configuration already stored in '{configuration_path}':")
            _Print_Diff_Tree(["stored", "new"], differing_leaves)
            if not _Confirm("Overwrite the stored configuration with this run's values?"):
                print("Match_Contribution_Run_Comparer: run aborted by user")
                return False

    pairing_already_logged = any(
        existing_voice_ids == voice_ids and existing_conversation_json_file_name == conversation_json_file_name
        for existing_voice_ids, existing_conversation_json_file_name in stored_pairings
    )
    if pairing_already_logged:
        print(f"Match_Contribution_Run_Comparer: WARNING - the voice_ids/conversation_json_file_name pairing {(voice_ids, conversation_json_file_name)} is already logged for '{analysis_directory}'.")
        if not _Confirm("Continue anyway?"):
            print("Match_Contribution_Run_Comparer: run aborted by user")
            return False
    else:
        stored_pairings.append([voice_ids, conversation_json_file_name])

    output_configuration = dict(new_configuration_values)
    output_configuration["voice_id_conversation_pairings"] = stored_pairings
    with open(configuration_path, "w") as f:
        json.dump(output_configuration, f, indent=2)

    return True


# --- entry point 2: user-callable, surfaces differences between the recorded configurations of multiple analysis directories ---

def Compare_Match_Contribution_Runs(comparison_run_set):
    run_configurations = {}
    missing_run_folders = []
    for run_folder in comparison_run_set:
        run_directory = Analysis_Directory_Path + run_folder + "/"
        configuration_path = run_directory + _RUN_CONFIGURATION_FILENAME
        if not os.path.isdir(run_directory) or not os.path.isfile(configuration_path):
            missing_run_folders.append(run_folder)
            continue
        with open(configuration_path, "r") as f:
            run_configurations[run_folder] = json.load(f)

    if missing_run_folders:
        print(f"Match_Contribution_Run_Comparer: the following comparison run folder(s) do not exist or do not contain a '{_RUN_CONFIGURATION_FILENAME}' file, aborting: {', '.join(missing_run_folders)}")
        return

    # --- voice_ids/conversation_json_file_name pairings logged for some but not all runs ---
    runs_by_pairing = {}
    for run_folder in comparison_run_set:
        for voice_ids, conversation_json_file_name in run_configurations[run_folder].get("voice_id_conversation_pairings", []):
            pairing_key = (tuple(voice_ids), conversation_json_file_name)
            runs_by_pairing.setdefault(pairing_key, []).append(run_folder)

    for (voice_ids, conversation_json_file_name), runs_with_pairing in runs_by_pairing.items():
        if 0 < len(runs_with_pairing) < len(comparison_run_set):
            print(f"{(list(voice_ids), conversation_json_file_name)} | {', '.join(runs_with_pairing)}")

    # --- configuration entries that differ across the compared runs ---
    differing_leaves = []
    for key in _COMPARED_CONFIGURATION_KEYS:
        run_values = [run_configurations[run_folder].get(key, _MISSING) for run_folder in comparison_run_set]
        differing_leaves.extend(_Find_Differing_Leaves(run_values, [key]))

    _Print_Diff_Tree(comparison_run_set, differing_leaves)
