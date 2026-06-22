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
- `Accumulate_Frequency_Occurrence_Counts(distribution, bucket_centers, existing_counts, existing_timepoints_count, timepoint_mask, frequency_ratio_offsets)` — accumulates counts into an existing state. `timepoint_mask` (optional bool list) skips timepoints; `frequency_ratio_offsets` (optional float list, one per voiced bucket) is subtracted from each bucket's ratio before the threshold check — used by subtractive population runs.
- `Convert_Occurrence_Counts_To_Ratios(counts, timepoints_count)` — converts raw counts to occurrence ratios.
- `Extract_Frequency_Amount_Occurrence_Ratios` is now a thin wrapper over both (backward compatible).

### 4. `Subdistribution_Difference_Analyzer.py`
Compares the extracted subdistributions between every pair of audio files.
- For each pair, produces a per-bucket signed difference chart (`A − B`) per threshold tier. Bars are colored by which audio is higher.
- Also produces a summary chart showing the L1 distance (total absolute difference) between each pair at each threshold tier.
- Output: one PNG per audio pair + one summary PNG, all in `Analysis_Directory`.

### 5. `Layered_Occurrence_Count_Populator.py`
Processes audio from the TIMIT phoneme corpus (`../Phoneme_Corpus/data/TRAIN/DR1/{speaker_id}/{audio_name}.WAV.wav`) and accumulates `frequency_amount_occurrence_counts` state into JSON files for later use by `Layered_Subdistribution_Generator`.

**Entry point 1:** `Run_Layered_Occurence_Count_Population(speaker_audio_dict, subdistribution_layer)`

- `speaker_audio_dict`: `dict[str, list[str]]` — maps speaker IDs to audio filenames (without extension).
- `subdistribution_layer`: `"universal"` | `"voice"` | `"phoneme"`
  - `"universal"` — accumulates all voiced timepoints into one global JSON.
  - `"voice"` — one JSON per speaker.
  - `"phoneme"` — one JSON per voiced phoneme (32 phonemes tracked).
- Only timepoints annotated as voiced phonemes in the corresponding `.PHN` file are included. Unvoiced/silence timepoints are skipped via a `timepoint_mask` passed to `Accumulate_Frequency_Occurrence_Counts`.
- Persists after each audio so progress survives mid-run failures. Skips already-processed audios on re-run (tracked in each JSON's `processed_audios` field).
- Output JSON files written to `tmp/media/output/`:
  - `Universal_Frequency_Amount_Occurrence_Counts.json`
  - `Speaker_{id}_Frequency_Amount_Occurrence_Counts.json`
  - `Phoneme_{label}_Frequency_Amount_Occurrence_Counts.json`
- JSON structure: `{ "processed_audios": {speaker_id: [audio_name]}, "total_voiced_frequency_timepoints_count": float, "frequency_amount_occurrence_counts": [{float_key: int}], "frequency_bucket_centers": [float] }`

**Entry point 2:** `Run_Subtractive_Layered_Occurrence_Count_Population(speaker_audio_dict, subdistribution_layer, subtractive_subdistribution_tier, subtract_voice_for_phoneme=False)`

Same pipeline as entry point 1, but subtracts a pre-computed subdistribution from each bucket's frequency ratio before accumulation, so only the residual above that baseline is counted.

- `subtractive_subdistribution_tier`: float threshold (e.g. `0.9`) identifying which tier of the universal/voice subdistribution to subtract.
- `"universal"` pathway — identical to entry point 1 (no subtraction needed at universal level).
- `"voice"` pathway — loads the universal subdistribution at `subtractive_subdistribution_tier` once and passes it as `frequency_ratio_offsets` to `Accumulate_Frequency_Occurrence_Counts` for every audio. Writes to `Speaker_{id}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}.json`.
- `"phoneme"` pathway — subtracts the universal subdistribution offset; if `subtract_voice_for_phoneme=True`, also subtracts the per-speaker subtractive voice subdistribution (loaded from the speaker's `_Subtractive_` JSON, which must already exist). Writes to `Phoneme_{label}_Subtractive_Frequency_Amount_Occurrence_Counts_{tier}.json`.
- `{tier}` in filenames is the threshold value with `.` removed (e.g. `0.9` → `09`).

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

### 7. `Voice_Subdistribution_Deviation_Tracker.py`
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

Written to `tmp/media/output/` (corpus analysis):
- `Universal_Frequency_Amount_Occurrence_Counts.json`
- `Speaker_{id}_Frequency_Amount_Occurrence_Counts.json`
- `Phoneme_{label}_Frequency_Amount_Occurrence_Counts.json`

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (layered subdistribution charts):
- `{run_name}universal_subdistributions.png`
- `{run_name}voice_original_subdistributions_{speaker_id}.png`
- `{run_name}voice_subtractive_subdistributions_{speaker_id}.png`
- `{run_name}phoneme_original_subdistributions_{phoneme}.png`
- `{run_name}phoneme_subtractive_subdistributions_{phoneme}.png`

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (deviation tracking):
- `{run_name}_voice_comparative_progression_{voice_id}_{speaker_id}_{occurrence_ratio_threshold}.png` (one per comparative speaker)
