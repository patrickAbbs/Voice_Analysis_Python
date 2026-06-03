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
- `Accumulate_Frequency_Occurrence_Counts(distribution, bucket_centers, existing_counts, existing_timepoints_count, timepoint_mask)` — accumulates counts into an existing state; accepts an optional boolean `timepoint_mask` to skip timepoints.
- `Convert_Occurrence_Counts_To_Ratios(counts, timepoints_count)` — converts raw counts to occurrence ratios.
- `Extract_Frequency_Amount_Occurrence_Ratios` is now a thin wrapper over both (backward compatible).

### 4. `Subdistribution_Difference_Analyzer.py`
Compares the extracted subdistributions between every pair of audio files.
- For each pair, produces a per-bucket signed difference chart (`A − B`) per threshold tier. Bars are colored by which audio is higher.
- Also produces a summary chart showing the L1 distance (total absolute difference) between each pair at each threshold tier.
- Output: one PNG per audio pair + one summary PNG, all in `Analysis_Directory`.

### 5. `Layered_Occurrence_Count_Populator.py`
Processes audio from the TIMIT phoneme corpus (`../Phoneme_Corpus/data/TRAIN/DR1/{speaker_id}/{audio_name}.WAV.wav`) and accumulates `frequency_amount_occurrence_counts` state into JSON files for later use by `Layered_Subdistribution_Generator`.

Entry point: `Run_Layered_Occurence_Count_Population(speaker_audio_dict, subdistribution_layer)`

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

### 6. `Layered_Subdistribution_Generator.py`
Loads the JSON files produced by `Layered_Occurrence_Count_Populator` and generates tiered subdistribution charts, with optional universal subtraction to isolate voice- or phoneme-specific frequency patterns.

Entry point: `Run_Layered_Subdistribution_Generation(subdistribution_layer, voice_set, phoneme_set, allow_negative_subtractive_subdistributions, generate_original_subdistribution_charts, generate_subtractive_subdistribution_charts)`

- `"universal"` — generates one chart from the universal JSON.
- `"voice"` — for each speaker in `voice_set`, generates original and/or subtractive charts. Subtractive = voice tier minus universal tier per bucket.
- `"phoneme"` — same as voice but for each phoneme in `phoneme_set`.
- `allow_negative_subtractive_subdistributions`: if `False`, subtractive values are clamped to 0.
- `generate_original_subdistribution_charts` / `generate_subtractive_subdistribution_charts`: boolean flags controlling which chart types are produced.
- Output PNGs written to `Analysis_Directory` with `Analysis_Run_Name` prefix:
  - `{run_name}universal_subdistributions.png`
  - `{run_name}voice_original_subdistributions_{speaker_id}.png`
  - `{run_name}voice_subtractive_subdistributions_{speaker_id}.png`
  - `{run_name}phoneme_original_subdistributions_{phoneme}.png`
  - `{run_name}phoneme_subtractive_subdistributions_{phoneme}.png`

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
