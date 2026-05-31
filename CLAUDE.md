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

**Known bug (line 18):** The occurrence count increment logic is incorrect for entries where a new frequency ratio is lower than an existing one. A new entry at ratio 0.1 inherits the count of the nearest-above entry (0.2) rather than starting from 0, which means it may be undercounted for all thresholds below 0.2.

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

All written to `Analysis_Directory` with `Analysis_Run_Name` as prefix:
- `{run_name}_spectrograms.png`
- `{run_name}_frequency_distribution.png`
- `{run_name}self_subdistributions_{audio_name}.png` (one per audio file)
- `{run_name}cross_subdistributions_{threshold}.png` (one per threshold tier)
