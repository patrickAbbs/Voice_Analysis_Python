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
Processes audio from the TIMIT phoneme corpus (`{Phoneme_Corpus_Directory}{speaker_id}/{audio_name}.WAV.wav`, default `../Phoneme_Corpus/data/TRAIN/DR1/`) and accumulates `frequency_amount_occurrence_counts` state into JSON files for later use by `Layered_Subdistribution_Generator`. `Phoneme_Corpus_Directory` lives in `Global_Hyperparameters.py` so it can be referenced by other modules (e.g. `Simulated_Conversation_Generator`) without duplicating the path.

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

### 8. `Simulated_Conversation_Generator.py`
Generates semi-random speaker ↔ audio sequences ("simulated conversations") for use as the input to `Element_Match_Contribution_Type_Explorer` — saved to a JSON file rather than passed in-memory.

**Entry point:** `Generate_Simulated_Conversation_Set(conversation_count, speaker_weights, turn_duration_seeds, conversation_duration_seeds)`

- `conversation_count`: positive int — number of conversations (top-level sequences) to generate.
- `speaker_weights`: `dict[str, float]` — speaker_id → selection weight (0–1). Drives weighted-random speaker selection for each turn.
- `turn_duration_seeds` / `conversation_duration_seeds`: each a `dict` with `"base_duration"`, `"deviation_ratio"`, `"power_curve"` — used to semi-randomly compute a target duration (seconds) via `_Compute_Seeded_Duration`: `multiplier = (random()^power_curve * (1 - deviation_ratio)) + deviation_ratio`, then a 50% chance to invert the multiplier (`1/multiplier`), then `duration = multiplier * base_duration`.

**Per-conversation generation:**
- A `conversation_duration` threshold is computed from `conversation_duration_seeds`. Audio files are added turn by turn until total selected duration exceeds this threshold — checked before every audio pick (not just at turn boundaries), so a turn can be cut off mid-turn to enforce the conversation-level cutoff.
- Each turn picks a speaker via weighted random selection from `speaker_weights`, excluding the immediately-preceding turn's speaker (falls back to allowing a repeat if excluding it would leave no candidates, e.g. only one speaker configured).
- Each turn independently computes a `turn_duration` threshold (same seeded-duration logic, via `turn_duration_seeds`) and adds audio files from the speaker's corpus folder (`Phoneme_Corpus_Directory + speaker_id + "/"`) until that threshold is met or the conversation threshold is hit first.
- Audio selection per speaker excludes files already used by that speaker within the current conversation; once all of that speaker's files have been used, the exclusion set resets so all files become selectable again. Speaker file listings and per-file durations (`soundfile.info`) are cached across the whole run, not per-conversation.

**Output:** one JSON file in `Conversation_Sequence_Json_Directory` (`tmp/media/conversation_sequence_json/`), containing a list of conversations, each a list of `[speaker_id, audio_list]` pairs (plain JSON arrays — read back as tuples by `Element_Match_Contribution_Type_Explorer.Load_Comparative_Voices_Audio_Set`). Filename is built from `speaker_weights` entries ordered by weight descending (`{speaker_id}_{weight}` per entry, `.` → `o`) joined by `_`, followed by `_{conversation_duration_seeds["base_duration"]}.json` — e.g. `{"FCJF0": 0.6, "MEDR0": 0.4}` with base_duration `15.0` → `FCJF0_0o6_MEDR0_0o4_15o0.json`.

### 9. `Element_Match_Contribution_Type_Explorer.py`
Compares multiple computational approaches for scoring how well a comparative speaker's frequency ratios match a reference voice's statistical distribution, running all selected variants in parallel and generating charts for side-by-side comparison. Replaces the former `Occurrence_Ratio_Divergence_Match_Score_Tracker`.

**Entry point:** `Run_Element_Match_Contribution_Type_Exploration(voice_ids, conversation_json_file_name, aggregate_match_types, cross_type_hyperparameters, chart_type_inclusions, metric_inclusions)`

- `voice_ids`: `list[str]` — one or more reference speakers. Each voice_id's JSON is loaded and converted to inverted occurrence ratios, and the whole per-sequence timepoint pass (from `comparative_voices_audio_set`) is re-run once per voice_id against that voice_id's own baseline — functionally equivalent to calling the (former single-`voice_id`) entry point once per list entry: each voice_id still gets its own `Per-speaker overall`/`Per-speaker per-bucket`/`Combined` charts, with y-axis bounds computed independently per voice_id (not shared across voice_ids). A voice_id with no persisted `Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts...json` is skipped with a warning rather than aborting the whole run; the run only aborts if *no* voice_id has data. Audio → bucketed frequency distribution (`Process_Audio`) is voice_id-independent, so each `(speaker_id, audio_name)` pair referenced anywhere in `comparative_voices_audio_set` is processed at most once and the result is cached and reused across every voice_id, rather than being recomputed per voice_id. This assumes every voice_id's persisted JSON shares the same `frequency_bucket_centers` (true whenever they came from the same corpus-population run under the same `Global_Hyperparameters` config).
- `conversation_json_file_name`: filename (e.g. `"FCJF0_0o6_MEDR0_0o4_5o0.json"`) of a JSON file in `Conversation_Sequence_Json_Directory` (`tmp/media/conversation_sequence_json/`) — typically produced by `Simulated_Conversation_Generator.Generate_Simulated_Conversation_Set`, though any file with the same shape works. Loaded and parsed by `Load_Comparative_Voices_Audio_Set(conversation_json_file_name)` into the internal `comparative_voices_audio_set`: `list[list[tuple[str, list[str]]]]` — differs from the `dict[str, list[str]]` format used by other modules. Each top-level list entry ("sequence") is processed as one continuous timepoint stream (its own `speaker_data` in `all_results`, its own set of output charts) and is a list of `(speaker_id, audio_list)` sub-sequence tuples processed in order, allowing a single sequence to alternate between speakers (simulating a "conversation") instead of covering only one speaker. State that accumulates over time (`cumulative_comparative_occurrence_ratios`, `element_accumulative_deviations`, etc.) is **not** reset at sub-sequence boundaries — it runs continuously across the whole sequence regardless of which speaker is talking. Example (pre-JSON-load shape): `[[("FCJF0", ["SA1", "SA2"])], [("FCJF0", ["SA1"]), ("MEDR0", ["SI744"]), ("FCJF0", ["SI1027"])]]` — the first sequence is a plain single-speaker run (equivalent to the old dict-entry semantics); the second alternates FCJF0 → MEDR0 → FCJF0 within one continuous run.
- `aggregate_match_types`: `dict[str, dict]` — one entry per variant. Each value has:
  - `"include_variant"`: bool — whether to run this variant in the current execution.
  - `"hyperparameters"`: dict of variant-specific parameters (see variants below). Every variant's `hyperparameters` also accepts `"chart_y_minimum"` (float, default `-inf`): a floor on how deep that variant's overall-value y-axis is allowed to go — if the true observed minimum is lower (more negative) than `chart_y_minimum`, the axis is clamped to `chart_y_minimum` instead and out-of-range points simply don't render. This only affects the `overall_ylims` used by `Per-speaker overall`, `All-speaker overall`, and `Combined` (not per-bucket charts) and never affects the underlying stored values — anything clamped off-chart is still counted toward the aggregate metrics below. `deviation_scaled_percentile_proximity` ignores `chart_y_minimum`: its axis minimum is always fixed at `0.0` (see its variant description below).
- `cross_type_hyperparameters`: parameters shared across all variants:
  - `"occurrence_ratio_cumulation_half_life"`: float — converted to per-timepoint exponential decay weight for the runtime cumulative ratio tracking.
  - `"use_bell_curve_percentile_projection"`: bool — if `True`, uses the Gaussian z-score approximation to compute `value_2` (same bell-curve logic as the former module); if `False`, uses direct bisect-based lookup into the inverted occurrence ratios.
  - `"voice_profile_cumulation_half_life"`: float or `None` — selects which half_life-keyed `Speaker_{voice_id}_Frequency_Amount_Occurrence_Counts` JSON to load as the reference profile. `None` loads the default (no-half_life) file. Must match the `frequency_ratio_cumulation_half_life` used when populating that JSON.
  - `"use_signal_rate_simulation"`: bool (default `False`) — selects an alternate per-timepoint `new_ratio` computation, applied identically across all variants (see **Signal rate simulation** below). Fully independent of every other option; toggling it does not change which chart types or variants run.
  - `"include_non_voiced_timepoints"`: bool (default `False`) — requires `"use_signal_rate_simulation"` to also be `True`; if `True` while `use_signal_rate_simulation` is `False`, a warning is printed and the option is ignored for the run (falls back to the normal skip-non-voiced behavior). See **Including non-voiced timepoints** below.
  - `"continuous_voice_profiling"`: `dict` — see **Continuous voice profiling** below. Sub-keys:
    - `"use_continuous_voice_profiling"`: bool (default `False`) — if `True` and `"use_bell_curve_percentile_projection"` is also `True`, each voice_id's bell curve projections are learned on-the-fly from that voice's own speaking turns instead of looked up from the persisted JSON. If `True` while `use_bell_curve_percentile_projection` is `False`, a warning is printed and continuous profiling is ignored for the run.
    - `"continue_voice_profiles_across_conversations"`: bool (default `False`) — if `True`, a voice's learned profile persists across every sequence in `comparative_voices_audio_set`; if `False`, it resets at the start of each sequence.
    - `"use_cumulative_signal_rate_distribution_ratios"`: bool (default `False`) — if `True`, profile updates use `signal_rate_ratio` instead of `timepoint_ratio`. Requires `"use_signal_rate_simulation"` to also be `True`; if not, a warning is printed and `timepoint_ratio` is used instead.
    - `"voice_profile_timepoints_threshold"`: int (default `0`) — a voice's learned profile is only used for scoring once its `voice_timepoints_count` exceeds this value; before that, the voice is fully ignored (NaN) for comparison at every timepoint.
- `chart_type_inclusions`: `dict[str, bool]` — one entry per chart type, gating which of the five chart types (see **Output charts** below) are generated; missing keys default to `False`, same convention as `aggregate_match_types`'s `"include_variant"`. Keys: `"combined_overall"` (→ `Combined`), `"all_speaker_overall"` (→ `All-speaker overall`), `"per_speaker_overall"` (→ `Per-speaker overall`), `"per_speaker_per_bucket"` (→ `Per-speaker per-bucket`), `"continuous_voice_profile_convergence"` (→ `Continuous voice profile convergence`; also requires `cross_type_hyperparameters["continuous_voice_profiling"]["use_continuous_voice_profiling"]` to be `True`, otherwise it's treated as excluded regardless of this key's value). If every effective key is `False` (or omitted), the run aborts immediately after variant validation — before any audio processing or per-voice_id computation — since none of that work would feed a chart. When only some chart types are excluded, their type-specific y-axis-bounds computation (`_Compute_Global_Ylims` for the per-voice_id `overall_ylims`/`per_bucket_ylims`, and the separate joint computation for `all_speaker_overall_ylims`) is skipped too, but the underlying per-voice_id per-timepoint pass (`_Process_Sequences_For_Voice`) still always runs for every voice_id, since its output (`all_results`) is shared by every chart type and there's no way to know in advance which of it a given chart type needs less of.
- `metric_inclusions`: `dict[str, bool]` — one entry per aggregate metric (see **Aggregate metrics** below), gating which are computed and displayed in the `All-speaker overall` chart's subplot titles; missing keys default to `False`. Metrics are only ever computed for that one chart type — if `chart_type_inclusions["all_speaker_overall"]` is `False`, `metric_inclusions` has no effect regardless of its values.

**Shared per-timepoint computation:**

For variants that use `cumulative_comparative_occurrence_ratios`, `value_2` is computed once per bucket per timepoint (lookup or bell-curve path) and the cumulative ratio is updated via `Weighted_Average`. `accumulative_deviation` also reuses this same `value_2` per timepoint but tracks its own separate state (see below) rather than appending to `cumulative_comparative_occurrence_ratios`. All speakers are fully processed before any charts are generated, so global y-axis bounds can be computed.

**Signal rate simulation** (`cross_type_hyperparameters["use_signal_rate_simulation"]`): an alternate, fully self-contained way of computing each bucket's per-timepoint `new_ratio`, swapped in wherever `value_2`/`new_ratio` would otherwise be computed from the raw `timepoint_ratio` — every downstream variant consumes `new_ratio` the same way regardless of which path produced it, so none of the variant-specific logic below needs to know which mode is active except `accumulative_deviation` (see its bullet). Two pieces of running state are tracked, both local to `_Process_Sequences_For_Voice`:
- `total_distribution_signal_rate`: a single float. Updated every valid timepoint via `Weighted_Average(total_distribution_signal_rate, occurrence_ratio_cumulation_weight, 1.0, 1.0 - occurrence_ratio_cumulation_weight)` — an EWA that converges toward `1.0`, independent of any bucket's actual ratio.
- `distribution_ratio_signal_rates`: `dict[frequency_bucket, float]`. Updated every valid timepoint, per bucket, via `Weighted_Average(distribution_ratio_signal_rates[freq_center], occurrence_ratio_cumulation_weight, timepoint_ratio, 1.0 - occurrence_ratio_cumulation_weight)`.

Both are reset to `0.0` together at the start of every sequence (a new conversation always starts this tracking clean); a voice's own turns do not additionally reset this state (unlike `accumulative_deviation`'s `use_self_tracking_reset` — signal-rate tracking is speaker-agnostic and carries over continuously across every turn within a sequence, regardless of who is talking). Outside of that per-sequence reset, both values persist for the remainder of the sequence.

Both are fully updated for a timepoint (across every bucket) before any bucket proceeds to the lookup step. Each bucket then computes `signal_rate_ratio = distribution_ratio_signal_rates[freq_center] / total_distribution_signal_rate` and passes that (instead of `timepoint_ratio`) into `_Bell_Curve_Value_2`/`_Lookup_Closest_Value`; the result is assigned directly to `new_ratio` — no intermediate `value_2`, and no further `Weighted_Average` against the previous cumulative ratio (unlike the default path). `new_ratio` is still appended to `cumulative_comparative_occurrence_ratios` exactly as in the default path. Because `total_distribution_signal_rate` is an independent EWA toward `1.0` rather than the literal sum of that timepoint's bucket signal rates, `signal_rate_ratio` is not a true proportion — it can transiently exceed what a normalized share would allow, most visibly as a pronounced transient in the first several timepoints of a sequence, before both EWAs settle.

**Including non-voiced timepoints** (`cross_type_hyperparameters["include_non_voiced_timepoints"]`, requires `use_signal_rate_simulation`): normally a timepoint failing the voiced-ratio or voiced-phoneme check is skipped outright (`continue`, no state updates, doesn't advance `processed_timepoint_count`). With this option on, every timepoint is included on the timeline instead:
- The signal-rate state updates always run, but for a non-voiced timepoint the second `Weighted_Average` value is `0.0` in place of the normal `1.0` (for `total_distribution_signal_rate`) / `timepoint_ratio` (for `distribution_ratio_signal_rates[freq_center]`). Since both the numerator (`distribution_ratio_signal_rates[freq_center]`) and denominator (`total_distribution_signal_rate`) shrink by the same `occurrence_ratio_cumulation_weight` factor each non-voiced timepoint, `signal_rate_ratio` — and everything computed from it — holds constant for the duration of a non-voiced stretch. Because both values have shrunk during the stretch, the first voiced timepoint after it pulls the ratio back with more relative effect than a single voiced timepoint would mid-stretch — i.e. matching is more responsive to change right after a pause.
- Divide-by-zero guard: `total_distribution_signal_rate` can only become nonzero via a voiced timepoint; before the first voiced timepoint since the run (or the current sequence) started, it stays exactly `0.0`, so `signal_rate_ratio`'s division would fail. Any timepoint where `total_distribution_signal_rate == 0.0` is still counted (`processed_timepoint_count` still advances, still occupies a slot in `speaker_segments`) but every per-bucket and overall value across every included variant is set to `float('nan')` instead of computed — matplotlib renders `NaN` as a gap in the line, and `_Compute_Global_Ylims`'s min/max scans naturally ignore `NaN` (comparisons against `NaN` are always `False`), so no other code needed to change to support this.
- `accumulative_deviation` reads its own previous per-bucket value each timepoint; if that value is `NaN` (only possible immediately after this leading null stretch), it's treated the same as "no accumulation has happened yet" (`0.0`) rather than letting `NaN` propagate forward through every future timepoint for that bucket.

**Continuous voice profiling** (`cross_type_hyperparameters["continuous_voice_profiling"]`, requires `use_bell_curve_percentile_projection`): an alternative to looking up each voice_id's `(center, lower_standard_deviation, upper_standard_deviation)` bell curve projections from the persisted JSON (`_Extract_Bell_Curve_Projections`) — instead, those three values are learned on-the-fly per frequency bucket from the voice's own speaking turns within `comparative_voices_audio_set`, to investigate how the profiling process would perform if a voice were learned live rather than pre-trained. The static JSON-derived projections are still loaded regardless (as the `Continuous voice profile convergence` chart's ground truth), but every scoring use of `bell_curve_projections` is replaced by the live-learned equivalent when this pathway is active.

- **State**: tracked separately per voice_id (reset per `_Process_Sequences_For_Voice` call) and, within that, per frequency bucket per bell curve point (`lower_standard_deviation` target percentile `0.15865`, `median` target percentile `0.5`, `upper_standard_deviation` target percentile `0.84135`). A voice-level `voice_timepoints_count` starts at `0`; each bucket/point pair tracks `projected_distribution_ratio` and `cumulative_occurrence_percentile`, both starting at `0.0`. Whether this state persists across sequences or resets each sequence is controlled by `continue_voice_profiles_across_conversations` (see above).
- **Update** (`_Update_Continuous_Voice_Profile`): runs once per voiced timepoint (`timepoint_is_voiced`) during a turn where the active speaker is the voice_id itself. First, `voice_timepoints_count` is incremented; `weight_1 = 1.0 - (1.0 / voice_timepoints_count)`, `weight_2 = 1.0 - weight_1` (so `weight_1` grows toward `1.0` as more timepoints accumulate, stabilizing the estimate). Then per bucket per point: `cumulative_occurrence_percentile` is updated via `Weighted_Average(cumulative_occurrence_percentile, weight_1, 1.0 if ratio < projected_distribution_ratio else 0.0, weight_2)` — the ratio being `signal_rate_ratio` if `use_cumulative_signal_rate_distribution_ratios` else `timepoint_ratio` — then `projected_distribution_ratio` is nudged via `Weighted_Average(projected_distribution_ratio, weight_1, projected_distribution_ratio + (1.0 if cumulative_occurrence_percentile < target_percentile else -1.0), weight_2)`. The update happens before this same timepoint is scored, so a voice's own-turn comparisons always use its freshly updated profile.
- **Readiness gate**: `is_voice_profile_ready = voice_timepoints_count > voice_profile_timepoints_threshold`. At every timepoint (any speaker, not just the voice_id's own turns), if not ready, every score for that voice_id at that timepoint is set to `NaN` (same mechanism as the `include_non_voiced_timepoints` divide-by-zero guard) instead of computed — a voice_id is effectively invisible in every chart until its profile stabilizes. Once ready, `_Continuous_Voice_Profile_Bell_Curve_Projections` converts the three per-point `projected_distribution_ratio` values into the same `(center, lower_standard_deviation, upper_standard_deviation)` shape `_Extract_Bell_Curve_Projections` returns (`center` = median's ratio; `lower_standard_deviation`/`upper_standard_deviation` = the median ratio's distance to the lower/upper point's ratio) and substitutes it for `bell_curve_projections` in `_Bell_Curve_Value_2`, and for `bucket_medians` (used by `accumulative_deviation`'s non-directional sign flip) wherever that's needed.
- **Convergence chart values** (only computed when `chart_type_inclusions["continuous_voice_profile_convergence"]` is also `True`): at every timepoint, once ready, `_Continuous_Voice_Profile_Convergence_Values` computes `-abs(live_value - static_value)` per bucket for each of the three points (in the same converted `(center, lower_standard_deviation, upper_standard_deviation)` form described above, diffed against the static `_Extract_Bell_Curve_Projections` result for that voice_id), then averages across buckets — one aggregate value per point per timepoint. Before ready, all three are `NaN`.

**Variants:**

- **`weighted_binary_match_contribution`** — the original match-score logic. Hyperparameters: `positive_contribution_range`, `positive_weight_power_curve`, `negative_weight_proximity_half_distance_increment`. Per-bucket: unsigned `match_contribution_weights`, unaffected by the shift described next (still its own separate `[-max, max]` per-bucket scale, used only by the `Per-speaker per-bucket` chart). Overall per-timepoint: `(fraction of total weight falling in the positive zone [0.5 ± pcr/2]) - 1.0` — the `-1.0` shift remaps the underlying [0, 1] fraction (1 = best) onto the same "0 is best, negative is worst" scale every other variant uses, range [-1, 0]. Because of this, its overall y-axis bounds are no longer a hardcoded `(0.0, 1.0)`; they're computed the same data-driven way as every other variant (see `chart_y_minimum` above).

- **`occurrence_percentile_deviation`** — internal `deviation = |0.5 - ratio| * 2.0` (0 at a perfect match, 1 at the worst match). This unflipped, 0-is-best form is also what's fed to `occurrence_percentile_inverse_deviation`/`occurrence_percentile_half_distance` below and to `accumulative_deviation`'s own `deviation_type: "occurrence_percentile_deviation"` option. No hyperparameters. What's actually stored/charted for this variant is the negated value (`-1.0 * deviation`, range [-1, 0]) so its own chart matches the same "0 is best, more negative is worse" y-axis convention shared by every other `occurrence_percentile_*`/`accumulative_deviation` variant, rather than the [0, 1]-with-1-as-best scale it used before. Overall per-timepoint: average of the negated per-bucket values across buckets.

- **`occurrence_percentile_inverse_deviation`** — `(-1.0 / (closeness ^ power_curve)) + 1.0`, clamped at `inverse_deviation_minimum`, where `closeness = 1.0 - deviation` recovers the original 1-is-best scale from `occurrence_percentile_deviation`'s unflipped internal `deviation`. Hyperparameters: `deviation_power_curve`, `inverse_deviation_minimum`. Value is 0 at perfect match (closeness = 1) and grows more negative as closeness falls. Overall per-timepoint: average across buckets.

- **`occurrence_percentile_half_distance`** — `-log_{0.5}(closeness)` (same `closeness = 1.0 - deviation` as above), clamped at `half_distance_minimum`. Hyperparameters: `half_distance_minimum`. Value is 0 at perfect match and grows more negative as closeness falls. Overall per-timepoint: average across buckets.

- **`accumulative_deviation`** — an exponentially decaying running sum (not a `Weighted_Average`-based EWA, except when `use_average_element_deviations` is set) of a per-bucket deviation measure, tracked independently of `cumulative_comparative_occurrence_ratios`. Hyperparameters:
  - `"decay_half_life"`: float — converted to `accumulative_deviation_decay_rate` via `Convert_Half_Life_To_Cumulation_Weight(Spectrogram_Window_Jump_In_Seconds, decay_half_life)`.
  - `"deviation_type"`: `"occurrence_percentile_inverse_deviation"` | `"occurrence_percentile_half_distance"` | `"occurrence_percentile_deviation"` — selects which calculation computes `current_timepoint_deviation` for a bucket at a timepoint. Unlike the standalone variants of the same name, this is always applied directly to the current timepoint's raw ratio (`value_2`, or `new_ratio` when `use_signal_rate_simulation` is `True` — see **Signal rate simulation** above) rather than to a cumulative EWA of it. For the first two, hyperparameters are read from that variant's own entry in `aggregate_match_types` (which does not need `include_variant: True` to supply them); `occurrence_percentile_deviation` takes none and computes `-1.0 * _Occurrence_Percentile_Deviation(value_2 or new_ratio)` — negated so it matches the other two options' 0-is-best, negative-is-worse polarity.
  - `"use_non_directional_element_deviations"`: bool — if `False`, `current_timepoint_deviation` is sign-flipped (`* -1.0`) whenever the timepoint's raw ratio is below the bucket's median (or bell-curve center) — or, when `use_signal_rate_simulation` is `True`, whenever `new_ratio < 0.5` (`bucket_medians` is not used in that mode).
  - `"use_average_element_deviations"`: bool — if `True`, the new per-bucket value is `Weighted_Average(previous_value, decay_rate, current_timepoint_deviation, 1.0 - decay_rate)`. If `False`, it is `(previous_value + current_timepoint_deviation) * decay_rate`.
  - `"use_self_tracking_reset"`: bool — if `True`, `previous_element_accumulative_deviation` is forced to `0.0` for every frequency bucket at the first valid timepoint of any turn whose speaker is the `voice_id` being compared against, instead of continuing from the last accumulated value — i.e. tracking restarts every time the reference voice starts speaking. If `False`, tracking runs continuously across the whole sequence regardless of who's speaking.
  - Per-bucket state: `element_accumulative_deviations`, seeded at `0.0`. Overall per-timepoint: `average_element_accumulative_deviations` — the average of the absolute values of that timepoint's new per-bucket entries, negated (always ≤ 0).

- **`deviation_scaled_percentile_proximity`** — requires `use_bell_curve_percentile_projection: True` in `cross_type_hyperparameters`; if `include_variant: True` but that's `False`, a warning is printed and the variant is excluded from the run (same "excluded, not aborted" treatment as `continuous_voice_profiling`/`include_non_voiced_timepoints`'s misconfiguration warnings). Hyperparameters: `percentile_proximity_power_curve`, `deviation_scaling_power_curve`. Per bucket per timepoint: `deviation = _Occurrence_Percentile_Deviation(new_ratio or value_2)` (the same 0-at-perfect-match, 1-at-worst-match value used elsewhere); `percentile_proximity = 1.0 - (deviation ** percentile_proximity_power_curve)`; the bell curve's `lower_standard_deviation` (if the ratio is below 0.5) or `upper_standard_deviation` (if above) is looked up from that timepoint's active bell curve projections (so it also works under continuous voice profiling) and floored at a tiny epsilon to avoid a divide-by-zero on a degenerate zero standard deviation; `bell_curve_deviation_scaling_multiplier = 1.0 / (standard_deviation ** deviation_scaling_power_curve)`; the bucket's value is `percentile_proximity * bell_curve_deviation_scaling_multiplier`. This rewards a close-to-median percentile more heavily for frequency buckets where the reference voice's distribution is narrower (more selective) at that bucket. Per-bucket state: `deviation_scaled_percentile_proximities`, seeded at `0.0`. Overall per-timepoint: `average_deviation_scaled_percentile_proximities` — the plain average (not negated) of that timepoint's per-bucket values. Unlike every other variant, this one is never negative and has **no capped maximum** (the scaling multiplier grows indefinitely as a bucket's standard deviation shrinks toward `0`), so its y-axis bounds are the odd ones out: `0.0` on the bottom (fixed, not `chart_y_minimum`-adjustable) and the observed maximum across all processed sequences on top — for both the overall charts and the `Per-speaker per-bucket` chart. It is not offered as a `deviation_type` option for `accumulative_deviation`, since that variant's accumulation logic assumes a `0`-capped per-timepoint deviation.

**Speaker segments:** for each sequence, `speaker_segments` is a `list[tuple[speaker_id, start_index, end_index]]` recording which contiguous range of per-timepoint data-array indices (1-based; index 0 is always the seed value) belongs to each sub-sequence tuple, in order — including repeats when the same speaker appears in multiple non-adjacent sub-sequences. This drives the per-chart speaker coloring/annotations described below and is stored in each sequence's `speaker_data["speaker_segments"]`.

**Speaker-segment annotations:** `_Draw_Speaker_Segment_Annotations(axis, speaker_segments, include_leading_line=False)` draws a vertical dotted line (colored to the incoming speaker) at each genuine speaker change, plus a speaker_id label centered below the subplot at the midpoint of each sub-sequence's timepoint span. When `include_leading_line=True` it additionally draws one at the very start of the chart, colored to the *first* speaker in the sequence — offset half a `Spectrogram_Window_Jump_In_Seconds` step forward from x=0 so it doesn't render on top of (and get hidden by) the axis's left spine. Used by `Per-speaker overall` and `All-speaker overall` only (see below); `Per-speaker per-bucket` and `Combined` keep the transition-only behavior (`include_leading_line=False`).

**Output charts (five chart types per run):**
- **Per-speaker overall** (`{run_name}_element_match_overall_{voice_id}_{sequence_index}_{sub_sequence_speaker_ids}.png`, e.g. `..._FCJF0_1_FCJF0_MEDR0_FCJF0_FECD0.png`; one per voice_id per sequence): one subplot per included variant, single overall timepoint line drawn in one solid color — the color assigned to `voice_id` (the baseline being compared to), for the line's entire length, regardless of which speaker is talking at a given point. Every included variant's subplot (including `weighted_binary_match_contribution`, since its overall value is now on the same "0 is best" scale as the rest) uses y ∈ [global_min, 0], where global_min is the lowest value observed across all sequences for that variant (scoped to this voice_id only), floored at that variant's `chart_y_minimum` if one is set — except `deviation_scaled_percentile_proximity`, whose subplot uses y ∈ [0.0, global_max] instead (see its variant description above). Speaker-segment annotations are drawn with `include_leading_line=True`.
- **All-speaker overall** (`{run_name}_element_match_all_speaker_overall_{voice_ids_joined}_{sequence_index}_{sub_sequence_speaker_ids}.png`; one per sequence per run, overlaying every successfully-loaded voice_id — new in the `voice_ids` refactor): same subplot layout as `Per-speaker overall`, but each subplot overlays one line per voice_id (each in that voice_id's own color, legended) for that sequence — effectively a combined overlay of that sequence's `Per-speaker overall` charts across all `voice_ids`. Y-axis bounds are computed jointly across all included voice_ids (not per-voice_id) so every overlaid line stays visible on a shared scale. Speaker-segment annotations (including the leading line) are drawn once, sourced from any one voice_id's `speaker_segments` for that sequence (identical across voice_ids since segment boundaries are audio-derived, not voice_id-derived). Subplot titles are `"{variant} | {sequence_label} | {metric_values}"` (the trailing `| {metric_values}` segment is omitted if no metrics are enabled) — see **Aggregate metrics** below.
- **Per-speaker per-bucket** (`{run_name}_element_match_per_bucket_{voice_id}_{sequence_index}_{sub_sequence_speaker_ids}.png`; one per voice_id per sequence): one subplot per included variant, one line per frequency bucket (purple→orange gradient) — bucket line coloring is unaffected by speaker segments. `weighted_binary_match_contribution` subplot shows signed weights (positive in positive zone, negative in negative zone) with symmetric y-axis `[-global_abs_max, global_abs_max]`. `occurrence_percentile_deviation`, `occurrence_percentile_inverse_deviation`, and `occurrence_percentile_half_distance` use y ∈ [global_min, 0]. `accumulative_deviation` depends on its own `use_non_directional_element_deviations` hyperparameter: if `True`, y ∈ [global_min, 0] (values stay non-positive since no sign-flip is applied); if `False`, y ∈ [-global_abs_max, global_abs_max] (symmetric, since the sign-flip makes values genuinely bidirectional — same rule as `weighted_binary_match_contribution`'s bucket chart). `deviation_scaled_percentile_proximity` uses y ∈ [0.0, global_max] (never negative, no capped maximum). Speaker-segment annotations drawn with `include_leading_line=False` (unchanged from before the `voice_ids` refactor).
- **Combined** (`{run_name}_element_match_combined_{voice_id}.png`; one per voice_id): one subplot per included variant, all of that voice_id's sequences' overall lines overlaid, each recolored per `speaker_segments` (legend deduplicated per speaker_id); segments where the speaker equals `voice_id` are drawn at 1.5× thickness, others at 0.75×. No vertical speaker-change lines or speaker_id labels (would create too much visual noise with multiple overlaid sequences) — `include_leading_line` is not applicable here. Y-axis bounds match this voice_id's `Per-speaker overall` chart.
- **Continuous voice profile convergence** (`{run_name}_continuous_voice_profile_convergence_{voice_ids_joined}.png`; one file per run, requires `use_continuous_voice_profiling`): one subplot per sequence in `comparative_voices_audio_set` (not per variant). Each voice_id gets three lines per subplot — one per bell curve point (`lower_standard_deviation`, `median`, `upper_standard_deviation`) — all in that voice_id's color but with a distinct line style (dotted/solid/dashed respectively), plotting the convergence values described above. A line renders only from the timepoint its voice_id's profile becomes ready onward (`NaN` before that). Y-axis is shared across every subplot: max `0.0`, min = the lowest value observed across all subplots/lines. Speaker-segment annotations drawn with `include_leading_line=True`, sourced from any one voice_id's `speaker_segments` per sequence (same rationale as `All-speaker overall`).

**Aggregate metrics** (`_Compute_All_Speaker_Metrics`, gated by `metric_inclusions`): computed once per `(variant, sequence_index)` pair — i.e. once per subplot of the `All-speaker overall` chart — from that variant's overall per-timepoint value lists (`_OVERALL_KEYS[variant]`) across every successfully-loaded voice_id, plus that sequence's `speaker_segments`. For every metric, an audio period whose `speaker_id` has no corresponding entry among the compared `voice_ids` is excluded entirely (neither numerator nor denominator), and any timepoint where the speaker's own voice's value is `NaN` (e.g. a leading non-voiced timepoint under `include_non_voiced_timepoints`) is skipped the same way. Results are rounded to 3 decimal places and joined as `"{metric_name} {value}"` comma-separated, in `METRIC_ORDER` (`match_ratio`, `transition_duration`, `match_differentiation`) regardless of `metric_inclusions` dict order; a metric with no eligible data (e.g. zero eligible timepoints) reports `nan`.
- `"match_ratio"`: `(timepoints where the current speaker's own voice has the strict — non-tied — highest value) / (total eligible timepoints)`. A timepoint where the speaker's own voice exactly ties another voice's value does not count toward the numerator, but still counts toward the denominator.
- `"transition_duration"`: averaged per speaker turn (each eligible entry in `speaker_segments`, including the very first turn of a sequence): the elapsed time from the start of that turn until the first timepoint where the turn's speaker's own voice reaches the strict highest value (same tie rule as `match_ratio`). If that voice never reaches it within the turn, the turn's full duration is used instead of an unbounded/missing value. Elapsed timepoint counts are converted to seconds via `Spectrogram_Window_Jump_In_Seconds`.
- `"match_differentiation"`: for each eligible timepoint, `(speaker's own voice's value) / (other voice's value)` is computed for every other compared voice and averaged; those per-timepoint averages are then averaged across all eligible timepoints in the sequence. A pairing where the other voice's value is exactly `0.0` is skipped (divide-by-zero guard not specified by the original request) rather than aborting the whole timepoint.

**`Subdistribution_Extractor.py` change:** `Convert_Occurrence_Counts_To_Ratios` gained an `invert=False` parameter. When `True`, each ratio `v` is stored as `1 - v`. All existing callers use the default and are unaffected.

### 10. `Occurrence_Ratio_Percentile_Shape_Visualizer.py`
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
| `Phoneme_Corpus_Directory` | `../Phoneme_Corpus/data/TRAIN/DR1/` | TIMIT phoneme corpus root; `{speaker_id}/{audio_name}.WAV.wav` underneath. Used by `Layered_Occurrence_Count_Populator` and `Simulated_Conversation_Generator` |

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

Written to `Conversation_Sequence_Json_Directory` (`tmp/media/conversation_sequence_json/`) (simulated conversation generation):
- `{speaker_id}_{weight}_..._{conversation_base_duration}.json` (one per `Generate_Simulated_Conversation_Set` run, speaker segments ordered by weight descending)

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (element match type exploration):
- `{run_name}_element_match_overall_{voice_id}_{sequence_index}_{sub_sequence_speaker_ids}.png` (one per voice_id per sequence)
- `{run_name}_element_match_all_speaker_overall_{voice_ids_joined}_{sequence_index}_{sub_sequence_speaker_ids}.png` (one per sequence, overlaying all voice_ids)
- `{run_name}_element_match_per_bucket_{voice_id}_{sequence_index}_{sub_sequence_speaker_ids}.png` (one per voice_id per sequence)
- `{run_name}_element_match_combined_{voice_id}.png` (one per voice_id, combining all sequences)
- `{run_name}_continuous_voice_profile_convergence_{voice_ids_joined}.png` (one per run, requires `use_continuous_voice_profiling`)

Written to `Analysis_Directory` with `Analysis_Run_Name` as prefix (percentile shape visualizer):
- `{run_name}_occurrence_ratio_percentile_shapes_{voice_id}.png` (one per voice_id)
