# Voice Analysis Python

A Python tool for comparing voice recordings by analyzing their frequency content over time. The pipeline takes `.wav` files, produces spectrograms, translates those into bucketed frequency distributions, and then extracts tiered subdistributions for cross-audio comparison.

## Entry Point

Run `analysis_runner.py` directly. It calls each pipeline stage in sequence and writes output charts to disk.

## Pipeline

### 1. `Spectrogram_Generator.py`
Loads `.wav` files from `Audio_Directory` and produces a STFT spectrogram per file.
- Multi-channel audio is collapsed to mono by averaging channels.
- FFT points are set to `2 * window_size_in_samples` to increase frequency resolution.
- Display types: `linear`, `logarithmic` (power `0.30102999566`), `decibel` (shifted to 0–80 range).
- Output: one combined PNG with all audio spectrograms as subplots.

### 2. `Frequency_Distribution_Generator.py`
Translates each spectrogram into bucketed frequency progressions, then normalizes to distributions.
- Buckets are defined by `Frequency_Distribution_Bucket_Increment` (center step) and `Frequency_Distribution_Bucket_Range` (window around each center). Uses a windowed-average approach — each bucket value is the average of all spectrogram frequency bins within that window.
- Normalization: each timepoint's bucket values are divided by the sum across all buckets, giving a ratio (share of total energy) per frequency per timepoint.
- Distribution types stored: `linear`, `logarithmic`, `decibel`. Only types in `Distribution_Types` are passed downstream.
- Output: one combined PNG showing the bucketed frequency distribution over time per audio file.

### 3. `Subdistribution_Extractor.py`
For each frequency bucket, determines the highest amplitude ratio that occurs at least X% of voiced timepoints — producing one "subdistribution" per threshold tier.
- A timepoint is counted as voiced only if the sum of distribution ratios within `Subdistribution_Voiced_Frequency_Limit` meets `Subdistribution_Timepoint_Voiced_Ratio_Minimum`.
- Thresholds are defined in `Subdistribution_Thresholds` (e.g. `[0.9, 0.8, 0.7, 0.6, 0.5]`).
- Output charts: one "self" chart per audio file (all threshold tiers stacked) and one "cross" chart per threshold tier (all audio files side by side).

**Refactored internals:** The original `Extract_Frequency_Amount_Occurrence_Ratios` was split into two exported functions:
- `Accumulate_Frequency_Occurrence_Counts(distribution, bucket_centers, existing_counts, existing_timepoints_count, timepoint_mask, frequency_ratio_offsets, frequency_ratio_cumulation_weight=0.0, existing_cumulative_ratios=None)` — accumulates counts into an existing state. `timepoint_mask` (optional bool list) skips timepoints; `frequency_ratio_offsets` (optional float list, one per voiced bucket) is subtracted from each bucket's ratio after the cumulative update; `frequency_ratio_cumulation_weight` (0–1) applies exponential smoothing to each bucket's ratio before counting — when non-zero, each valid timepoint's ratio is an EWA of the previous smoothed value and the raw value, making the logged ratio comparable to similarly weighted runtime values in analysis modules; `existing_cumulative_ratios` (optional list of `float | None`) seeds the per-bucket EWA state and is updated across calls to preserve continuity across audio files. Returns a 3-tuple `(counts, timepoints_count, current_cumulative_ratios)`.
- `Convert_Occurrence_Counts_To_Ratios(counts, timepoints_count)` — converts raw counts to occurrence ratios.
- `Extract_Frequency_Amount_Occurrence_Ratios` is now a thin wrapper over both (backward compatible; ignores the third return value).

### 4. `Subdistribution_Difference_Analyzer.py`
Compares the extracted subdistributions between every pair of audio files.
- For each pair, produces a per-bucket signed difference chart (`A − B`) per threshold tier. Bars are colored by which audio is higher.
- Also produces a summary chart showing the L1 distance (total absolute difference) between each pair at each threshold tier.
- Output: one PNG per audio pair + one summary PNG, all in `Analysis_Directory`.

### 5. `Layered_Occurrence_Count_Populator.py`
Processes audio from the TIMIT phoneme corpus (`../Phoneme_Corpus/data/TRAIN/DR1/{speaker_id}/{audio_name}.WAV.wav`) and accumulates `frequency_amount_occurrence_counts` state into JSON files for later use by `Layered_Subdistribution_Generator`.

**Entry point 1:** `Run_Layered_Occurrence_Count_Population(speaker_audio_dict, subdistribution_layer, frequency_ratio_cumulation_half_life=None)`

- `speaker_audio_dict`: `dict[str, list[str]]` — maps speaker IDs to audio filenames (without extension).
- `subdistribution_layer`: `"universal"` | `"voice"` | `"phoneme"`
  - `"universal"` — accumulates all voiced timepoints into one global JSON.
  - `"voice"` — one JSON per speaker.
  - `"phoneme"` — one JSON per voiced phoneme (32 phonemes tracked).
- `frequency_ratio_cumulation_half_life`: optional float. When set, converts to a per-timepoint EWA weight via `Convert_Half_Life_To_Cumulation_Weight` and passes it to `Accumulate_Frequency_Occurrence_Counts`, so the logged frequency ratios are exponentially smoothed in the same style as runtime values in analysis modules. `None` = no smoothing (default, weight = 0).
- Only timepoints annotated as voiced phonemes in the corresponding `.PHN` file are included. Unvoiced/silence timepoints are skipped via a `timepoint_mask` passed to `Accumulate_Frequency_Occurrence_Counts`.
- Persists after each audio so progress survives mid-run failures. Skips already-processed audios on re-run (tracked in each JSON's `processed_audios` field). The per-bucket EWA state (`current_cumulative_frequency_ratios`) is also persisted so continuity is maintained when resuming.
- Output JSON files written to `Json_Directory`. File naming: no suffix when `frequency_ratio_cumulation_half_life=None`; otherwise appends `_{half_life}` with `.` replaced by `o` (e.g. half_life `0.2` → suffix `_0o2`):
  - `Universal_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
  - `Speaker_{id}_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
  - `Phoneme_{label}_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
- JSON structure: `{ "processed_audios": {speaker_id: [audio_name]}, "total_voiced_frequency_timepoints_count": float, "frequency_amount_occurrence_counts": [{float_key: int}], "frequency_bucket_centers": [float], "current_cumulative_frequency_ratios": [float | null] }`

**Entry point 2:** `Run_Subtractive_Layered_Occurrence_Count_Population(speaker_audio_dict, subdistribution_layer, subtractive_subdistribution_tier, subtract_voice_for_phoneme=False, frequency_ratio_cumulation_half_life=None)`

Same pipeline as entry point 1, but subtracts a pre-computed subdistribution from each bucket's frequency ratio before accumulation, so only the residual above that baseline is counted.

- `subtractive_subdistribution_tier`: float threshold (e.g. `0.9`) identifying which tier of the universal/voice subdistribution to subtract.
- `frequency_ratio_cumulation_half_life`: same as entry point 1; the same half_life is used when loading the universal/speaker offset files to ensure the subtracted baseline matches the accumulated data.
- `"universal"` pathway — identical to entry point 1 (no subtraction needed at universal level).
- `"voice"` pathway — loads the universal subdistribution at `subtractive_subdistribution_tier` once and passes it as `frequency_ratio_offsets` to `Accumulate_Frequency_Occurrence_Counts` for every audio. Writes to `Speaker_{id}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}[_{half_life}].json`.
- `"phoneme"` pathway — subtracts the universal subdistribution offset; if `subtract_voice_for_phoneme=True`, also subtracts the per-speaker subtractive voice subdistribution (loaded from the speaker's `_Subtractive_` JSON, which must already exist). Writes to `Phoneme_{label}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}[_{half_life}].json`.
- `{tier}` in filenames is the threshold value with `.` replaced by `o` (e.g. `0.9` → `0o9`). Half_life suffix follows the same convention.

**Helper:** `Format_Half_Life_For_Filename(half_life)` — returns `""` if `None`, otherwise `"_" + str(half_life).replace(".", "o")`. Exported for use by other modules that load half_life-keyed JSON files.

### 6. `Layered_Subdistribution_Generator.py`
Loads the JSON files produced by `Layered_Occurrence_Count_Populator` and generates tiered subdistribution charts.

**Entry point 1:** `Run_Layered_Subdistribution_Generation(subdistribution_layer, voice_set, phoneme_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts)`

- `"universal"` — generates one chart from the universal JSON.
- `"voice"` — for each speaker in `voice_set`, generates original and/or subtractive charts. Subtractive = voice tier minus universal tier per bucket.
- `"phoneme"` — same as voice but for each phoneme in `phoneme_set`.
- `allow_negative_subtractive_subdistributions`: if `False`, subtractive values are clamped to 0.
- `generate_original_subdistribution_charts` / `generate_subtractive_subdistribution_charts`: boolean flags controlling which chart types are produced.
- Output PNGs written to `Analysis_Directory` with `Analysis_Run_Name` prefix:
  - `{run_name}_universal_subdistributions.png`
  - `{run_name}_voice_original_subdistributions_{speaker_id}.png`
  - `{run_name}_voice_subtractive_subdistributions_{speaker_id}.png`
  - `{run_name}_phoneme_original_subdistributions_{phoneme}.png`
  - `{run_name}_phoneme_subtractive_subdistributions_{phoneme}.png`

**Entry point 2:** `Run_Subtractive_Layered_Subdistribution_Generation(subdistribution_layer, voice_set, phoneme_set)`

Loads the `_Subtractive_` JSON files produced by `Run_Subtractive_Layered_Occurrence_Count_Population` and generates tiered subdistribution charts. Key differences from entry point 1:

- Each subdistribution tier is loaded from its own per-tier JSON file and extracted using only that tier's threshold (rather than computing all tiers from one file). This is required because each `_Subtractive_` JSON was built with a different offset subtracted.
- No universal subtraction step at chart generation time — that subtraction already happened during population.
- `"universal"` pathway — identical to entry point 1.
- `"voice"` — for each speaker in `voice_set`, loads `Speaker_{id}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}.json` for each tier and generates one chart.
- `"phoneme"` — same pattern for each phoneme in `phoneme_set`.
- Output PNGs:
  - `{run_name}_subtractive_voice_subdistributions_{speaker_id}.png`
  - `{run_name}_subtractive_phoneme_subdistributions_{phoneme}.png`

### 7. `Voice_Subdistribution_Deviation_Tracker.py` *(effectively deprecated — superseded by `Element_Match_Contribution_Type_Explorer`)*
Tracks how a comparative speaker's per-bucket frequency ratios deviate from a reference voice's subdistribution baseline over time, producing an exponentially weighted cumulative progression per frequency bucket.

**Entry point:** `Run_Voice_Subdistribution_Deviation_Tracking(voice_id, comparative_voices_audio_set, occurrence_ratio_threshold, cumulation_half_life)`

- `voice_id`: reference speaker ID whose persisted JSON (`Speaker_{id}_Frequency_Amount_Occurrence_Counts.json`) is loaded and used to extract a single `Subdistribution_Tier` at `occurrence_ratio_threshold`. This gives the per-bucket baseline — the highest frequency ratio that occurs at least `occurrence_ratio_threshold` fraction of the time for the reference speaker.
- `comparative_voices_audio_set`: `dict[str, list[str]]` — maps each comparative speaker ID to their audio filenames. All keys are processed.
- `occurrence_ratio_threshold`: float (e.g. `0.6`) — used both as the subdistribution extraction threshold and as the neutral midpoint in the deviation normalization.
- `cumulation_half_life`: float — converted to a per-timepoint `cumulation_weight` via `Convert_Half_Life_To_Cumulation_Weight(Spectrogram_Window_Jump_In_Seconds, cumulation_half_life)` using an exponential decay formula (`0.5 ^ (window_duration / half_life)`).

**Per-speaker processing:**

For each comparative speaker, a `cumulative_above_subdistribution_value_and_deviation_progressions` dict is initialized: keys are voiced `frequency_bucket_centers`, values are `(first_list, second_list)` tuples initialized to `([occurrence_ratio_threshold], [0.0])`.

Audio files are processed in order via `Process_Audio()`. Valid timepoints (passing the voiced-ratio minimum and voiced-phoneme mask checks from `Accumulate_Frequency_Occurrence_Counts`) append new values to both lists for every frequency bucket:

- **First list** — exponentially weighted average of whether the comparative speaker's ratio exceeds the reference baseline at this bucket: `Weighted_Average(last_value, cumulation_weight, (1.0 if ratio > baseline else 0.0), 1 - cumulation_weight)`. Represents the cumulative fraction of time the comparative speaker is above the reference voice's threshold.
- **Second list** — deviation of the first list value from `occurrence_ratio_threshold`, normalized to `[-1, 1]`: positive when above threshold `((first - threshold) / (1 - threshold))`, negative when below `((threshold - first) / threshold * -1)`.

**Output chart** (one PNG per comparative speaker):
- Two-subplot line graph. Each line = one frequency bucket, colored on a purple (lowest frequency) → orange (highest frequency) gradient.
- Subplot 1: first list values, y-axis `[0, 1]`.
- Subplot 2: second list values, y-axis `[-1, 1]`, with a zero baseline.
- X-axis for both: time in seconds (`datapoint_index * Spectrogram_Window_Jump_In_Seconds`).
- Saved to `{Analysis_Directory}{Analysis_Run_Name}_voice_comparative_progression_{voice_id}_{speaker_id}_{occurrence_ratio_threshold}.png`.

**Shared helpers** (defined here for reuse by future modules):
- `Convert_Half_Life_To_Cumulation_Weight(processing_window_duration, half_life)` — exponential decay weight.
- `Weighted_Average(value_1, weight_1, value_2, weight_2)` — scalar weighted average.

### 8. `Element_Match_Contribution_Type_Explorer.py`
Compares multiple computational approaches for scoring how well a comparative speaker's frequency ratios match a reference voice's statistical distribution, running all selected variants in parallel and generating charts for side-by-side comparison. Replaces the former `Occurrence_Ratio_Divergence_Match_Score_Tracker`.

**Entry point:** `Run_Element_Match_Contribution_Type_Exploration(voice_id, comparative_voices_audio_set, aggregate_match_types, cross_type_hyperparameters)`

- `voice_id`: reference speaker whose JSON is loaded and converted to inverted occurrence ratios.
- `comparative_voices_audio_set`: `dict[str, list[str]]` — same format as other modules.
- `aggregate_match_types`: `dict[str, dict]` — one entry per variant. Each value has:
  - `"include_variant"`: bool — whether to run this variant in the current execution.
  - `"hyperparameters"`: dict of variant-specific parameters (see variants below).
- `cross_type_hyperparameters`: parameters shared across all variants:
  - `"occurrence_ratio_cumulation_half_life"`: float — converted to per-timepoint exponential decay weight for the runtime cumulative ratio tracking.
  - `"use_bell_curve_percentile_projection"`: bool — if `True`, uses the Gaussian z-score approximation to compute `value_2` (same bell-curve logic as the former module); if `False`, uses direct bisect-based lookup into the inverted occurrence ratios.
  - `"voice_profile_cumulation_half_life"`: float or `None` — selects which half_life-keyed `Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts` JSON to load as the reference profile. `None` loads the default (no-half_life) file. Must match the `frequency_ratio_cumulation_half_life` used when populating that JSON.

**Shared per-timepoint computation:**

For variants that use `cumulative_comparative_occurrence_ratios` (all except `raw_distance`), `value_2` is computed once per bucket per timepoint (lookup or bell-curve path) and the cumulative ratio is updated via `Weighted_Average`. All speakers are fully processed before any charts are generated, so global y-axis bounds can be computed.

**Variants:**

- **`weighted_binary_match_contribution`** — the original match-score logic. Hyperparameters: `positive_contribution_range`, `positive_weight_power_curve`, `negative_weight_proximity_half_distance_increment`. Per-bucket: unsigned `match_contribution_weights`. Overall per-timepoint: fraction of total weight falling in the positive zone `[0.5 ± pcr/2]`. Range [0, 1].

- **`occurrence_percentile_deviation`** — `1.0 - |0.5 - ratio| * 2.0`. No hyperparameters. Range [0, 1]; value is 1 when ratio = 0.5 and 0 when ratio = 0 or 1. Overall per-timepoint: average across buckets.

- **`occurrence_percentile_inverse_deviation`** — `(-1.0 / (deviation ^ power_curve)) + 1.0`, clamped at `inverse_deviation_minimum`. Hyperparameters: `deviation_power_curve`, `inverse_deviation_minimum`. Value is 0 at perfect match (deviation = 1) and grows more negative as deviation falls. Overall per-timepoint: average across buckets.

- **`occurrence_percentile_half_distance`** — `-log_{0.5}(deviation)`, clamped at `half_distance_minimum`. Hyperparameters: `half_distance_minimum`. Value is 0 at perfect match and grows more negative as deviation falls. Overall per-timepoint: average across buckets.

- **`raw_distance`** — uses `cumulative_raw_distances` instead of `cumulative_comparative_occurrence_ratios`. `value_2 = -|median - timepoint_frequency_ratio|` where median is the per-bucket frequency ratio key with inverted occurrence ratio closest to 0.5. No hyperparameters. Always ≤ 0. Overall per-timepoint: average across buckets.

- **`accumulative_deviation`** — an exponentially decaying running sum (not a `Weighted_Average`-based EWA, except when `use_average_element_deviations` is set) of a per-bucket deviation measure, tracked independently of `cumulative_comparative_occurrence_ratios`. Hyperparameters:
  - `"decay_half_life"`: float — converted to `accumulative_deviation_decay_rate` via `Convert_Half_Life_To_Cumulation_Weight(Spectrogram_Window_Jump_In_Seconds, decay_half_life)`.
  - `"deviation_type"`: `"occurrence_percentile_inverse_deviation"` | `"occurrence_percentile_half_distance"` | `"raw_distance"` — selects which of those three variants' calculation is used to compute `current_timepoint_deviation` for a bucket at a timepoint. Unlike the standalone variants of the same name, this calculation is always applied directly to the current timepoint's raw ratio (`value_2` or `timepoint_frequency_ratio`) rather than to a cumulative EWA of it. When the type is `occurrence_percentile_inverse_deviation` or `occurrence_percentile_half_distance`, its hyperparameters are read from that variant's own entry in `aggregate_match_types` (which does not need `include_variant: True` to supply them).
  - `"use_non_directional_element_deviations"`: bool — if `False`, `current_timepoint_deviation` is sign-flipped (`* -1.0`) whenever the timepoint's raw ratio is below the bucket's median (or bell-curve center).
  - `"use_average_element_deviations"`: bool — if `True`, the new per-bucket value is `Weighted_Average(previous_value, decay_rate, current_timepoint_deviation, 1.0 - decay_rate)`. If `False`, it is `(previous_value + current_timepoint_deviation) * decay_rate`.
  - Per-bucket state: `element_accumulative_deviations`, seeded at `0.0`. Overall per-timepoint: `average_element_accumulative_deviations` — the average of the absolute values of that timepoint's new per-bucket entries, negated (always ≤ 0).

**Output charts (three per run):**
- **Per-speaker overall** (`{run_name}_element_match_overall_{voice_id}_{speaker_id}.png`): one subplot per included variant, single overall timepoint line in speaker color. `weighted_binary_match_contribution` and `occurrence_percentile_deviation` subplots use y ∈ [0, 1]. The rest (`occurrence_percentile_inverse_deviation`, `occurrence_percentile_half_distance`, `raw_distance`, `accumulative_deviation`) use y ∈ [global_min, 0], where global_min is the lowest value observed across all speakers for that variant.
- **Per-speaker per-bucket** (`{run_name}_element_match_per_bucket_{voice_id}_{speaker_id}.png`): one subplot per included variant, one line per frequency bucket (purple→orange gradient). `weighted_binary_match_contribution` subplot shows signed weights (positive in positive zone, negative in negative zone) with symmetric y-axis `[-global_abs_max, global_abs_max]`. `occurrence_percentile_deviation` subplot uses y ∈ [0, 1]. `occurrence_percentile_inverse_deviation`, `occurrence_percentile_half_distance`, and `raw_distance` use y ∈ [global_min, 0]. `accumulative_deviation` depends on its own `use_non_directional_element_deviations` hyperparameter: if `False`, y ∈ [global_min, 0] (same rule as the other deviation variants); if `True`, y ∈ [-global_abs_max, global_abs_max] (symmetric, same rule as `weighted_binary_match_contribution`'s bucket chart).
- **Combined** (`{run_name}_element_match_combined_{voice_id}.png`): one subplot per included variant, all speakers' overall lines overlaid; `voice_id`'s own line is drawn at 1.5× thickness. Y-axis bounds match the per-speaker overall chart.

**`Subdistribution_Extractor.py` change:** `Convert_Occurrence_Counts_To_Ratios` gained an `invert=False` parameter. When `True`, each ratio `v` is stored as `1 - v`. All existing callers use the default and are unaffected.

### 9. `Occurrence_Ratio_Percentile_Shape_Visualizer.py`
Visualizes the distribution shape of observed frequency ratios per bucket for a given speaker, to support identifying compact mathematical representations of those distributions.

**Entry point:** `Visualize_Occurrence_Ratio_Percentile_Shapes(voice_ids, proximity_density_distance=0.001, voice_profile_cumulation_half_life=None)`

- `voice_ids`: list of speaker IDs to process. One chart is generated per speaker.
- `proximity_density_distance`: radius used for neighbor counting in subplot 3 (see below).
- `voice_profile_cumulation_half_life`: float or `None` — selects which half_life-keyed JSON to load, using the same convention as `Element_Match_Contribution_Type_Explorer`. `None` loads the default (no-half_life) file.

**Per-speaker processing:**

Loads `Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts.json`. For each frequency bucket, recovers the original per-timepoint frequency ratio observations by reconstructing a sorted (descending) list from the cumulative occurrence counts: the gap between adjacent entries' counts gives how many timepoints had exactly that ratio value. The resulting list has length equal to `total_voiced_frequency_timepoints_count`.

**Output chart** (3-subplot PNG per speaker, `{run_name}_occurrence_ratio_percentile_shapes_{voice_id}.png`):

- **Subplot 1 — Raw percentile shapes**: line per bucket, x ∈ [0, 1] (percentile, 0 = highest observed ratio), y = frequency ratio value. Shows absolute scale differences between buckets.
- **Subplot 2 — Min-max normalized per bucket**: each bucket's curve independently scaled to [0, 1], isolating shape from amplitude. Allows direct comparison of distribution shape across frequencies.
- **Subplot 3 — Proximity density**: x = frequency ratio value, y = normalized neighbor count. For each datapoint, counts how many other datapoints fall within `proximity_density_distance`, then normalizes so all counts for that bucket sum to 1.0. Computed in O(n log n) via `numpy.searchsorted` on the sorted ratio array. Reveals the density shape of the distribution as a function of ratio value rather than percentile rank.

## Data Model

All pipeline state lives in `Audio_Analysis_Data` (defined in `analysis_runner.py`):

| Field | Type | Set by |
|---|---|---|
| `Audio_File_Name` | str | constructor |
| `Spectrogram_Data` | `Spectrogram_Data` | step 1 |
| `Frequency_Bucket_Centers` | list[float] | step 2 |
| `Typed_Bucketed_Frequency_Progressions` | `Typed_Bucketed_Frequency_Progressions` | step 2 |
| `Typed_Bucketed_Frequency_Distributions` | `Typed_Bucketed_Frequency_Distributions` | step 2 |
| `Typed_Tiered_Subdistributions` | `Typed_Tiered_Subdistributions` | step 3 |

## Configuration (`Global_Hyperparameters.py`)

| Parameter | Default | Notes |
|---|---|---|
| `Audio_Directory` | `../voice_modulation_audio/` | path to `.wav` files |
| `Audio_File_Set` | `["normal_1", "no_nose_1", "round_1"]` | filenames without extension |
| `Analysis_Directory` | `tmp/media/voice_modulation/` | output chart directory |
| `Analysis_Run_Name` | `subdistributions_test` | prefix for output filenames |
| `Spectrogram_Window_Size_In_Seconds` | `0.05` | 50ms; speech standard is ~25ms |
| `Spectrogram_Window_Jump_In_Seconds` | `0.01` | 10ms hop |
| `Spectrogram_Displayed_Frequency_Maximum` | `4000` | Hz cap for spectrogram chart |
| `Frequency_Distribution_Bucket_Range` | `250.0` | Hz window per bucket |
| `Frequency_Distribution_Bucket_Increment` | `25.0` | Hz step between bucket centers |
| `Subdistribution_Voiced_Frequency_Limit` | `4000` | Hz; only buckets below this are used |
| `Subdistribution_Timepoint_Voiced_Ratio_Minimum` | `0.5` | min voiced-range ratio to count a timepoint |
| `Subdistribution_Thresholds` | `[0.9, 0.8, 0.7, 0.6, 0.5]` | occurrence ratio tiers |
| `Distribution_Types` | `["logarithmic"]` | which type(s) flow into subdistribution step |
| `Chart_Image_Resolution` | `250` | DPI for saved PNGs |

## Color Assignment (`Color_Assignment_Manager.py`)

Speaker and phoneme colors are assigned permanently on first encounter and persisted to `tmp/media/output/color_assignments.json`, so the same entity always gets the same color across all chart types and analysis runs.

- `Get_Speaker_Color(speaker_id)` — returns the assigned color for a speaker (or audio file name). Assigns one if not yet seen, using the next available slot in `Subdistribution_Display_Colors` (cyclic).
- `Get_Phoneme_Color(phoneme)` — same for phoneme labels, tracked in a separate list so phoneme indices don't collide with speaker indices.
- JSON structure: `{ "speakers": { id: color_name }, "phonemes": { label: color_name } }`
- Fixed path `tmp/media/output/color_assignments.json` (not inside `Json_Directory`) so assignments persist across runs with different `Json_Directory` configurations.

`Subdistribution_Display_Colors` in `Global_Hyperparameters.py` has been extended from 10 to 50 named matplotlib colors to minimize recycling. All chart-generation code in `Subdistribution_Extractor`, `Subdistribution_Difference_Analyzer`, and `Layered_Subdistribution_Generator` now calls `Get_Speaker_Color` / `Get_Phoneme_Color` instead of computing colors by position at call time.

## Shared Helpers (`Global_Helper_Functions.py`)

Contains general-purpose utility functions imported by multiple modules:
- `Convert_Half_Life_To_Cumulation_Weight(processing_window_duration, half_life)` — computes an exponential decay weight as `0.5 ^ (window_duration / half_life)`. Used wherever an EWA half_life is converted to a per-timepoint weight.
- `Weighted_Average(value_1, weight_1, value_2, weight_2)` — scalar weighted average `(v1*w1 + v2*w2) / (w1+w2)`.

These were previously defined in `Voice_Subdistribution_Deviation_Tracker.py`, which created circular import constraints. They now live here so any module can import them without dependency issues.

## Dependencies

`numpy`, `soundfile`, `librosa`, `matplotlib`. A `venv` is present at the repo root.

## Output Files

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (pipeline):
- `{run_name}_spectrograms.png`
- `{run_name}_frequency_distribution.png`
- `{run_name}self_subdistributions_{audio_name}.png` (one per audio file)
- `{run_name}cross_subdistributions_{threshold}.png` (one per threshold tier)
- `{run_name}subdistribution_diff_{a}_vs_{b}.png` (one per audio pair)
- `{run_name}subdistribution_diff_summary.png`

Written to `Json_Directory` (corpus analysis). Base names shown; suffix `_{half_life}` (with `.` → `o`) is appended when a `frequency_ratio_cumulation_half_life` is set (e.g. `_0o2` for half_life 0.2):
- `Universal_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
- `Speaker_{id}_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
- `Phoneme_{label}_Frequency_Amount_Occurrence_Counts[_{half_life}].json`
- `Speaker_{id}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}[_{half_life}].json`
- `Phoneme_{label}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}[_{half_life}].json`

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (layered subdistribution charts):
- `{run_name}universal_subdistributions.png`
- `{run_name}voice_original_subdistributions_{speaker_id}.png`
- `{run_name}voice_subtractive_subdistributions_{speaker_id}.png`
- `{run_name}phoneme_original_subdistributions_{phoneme}.png`
- `{run_name}phoneme_subtractive_subdistributions_{phoneme}.png`

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (deviation tracking):
- `{run_name}_voice_comparative_progression_{voice_id}_{speaker_id}_{occurrence_ratio_threshold}.png` (one per comparative speaker)

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (element match type exploration):
- `{run_name}_element_match_overall_{voice_id}_{speaker_id}.png` (one per comparative speaker)
- `{run_name}_element_match_per_bucket_{voice_id}_{speaker_id}.png` (one per comparative speaker)
- `{run_name}_element_match_combined_{voice_id}.png` (combined all-speakers chart)

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (percentile shape visualizer):
- `{run_name}_occurrence_ratio_percentile_shapes_{voice_id}.png` (one per voice_id)
