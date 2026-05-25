import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Chart_Image_Resolution, Audio_Directory, Audio_File_Set, Analysis_Directory, Analysis_Run_Name, Spectrogram_Window_Size_In_Seconds, Spectrogram_Window_Jump_In_Seconds, Spectrogram_Display_Type, Spectrogram_Displayed_Frequency_Maximum


def Generate_Audio_Spectrogram(audio_file_analysis_entry, audio_directory, audio_file_name, subplot_number):
    audio_file_path = audio_directory + audio_file_name + ".wav"
    audio_soundfile = soundfile.read(audio_file_path)
    audio_waveform_channels = audio_soundfile[0]
    sample_rate = audio_soundfile[1]
    audio_file_analysis_entry["sample_rate"] = sample_rate

    if audio_waveform_channels.ndim > 1:
        audio_waveform = numpy.mean(audio_waveform_channels, axis=1)
    else:
        audio_waveform = audio_waveform_channels

    spectrogram_window_size_in_audio_samples = round(sample_rate * Spectrogram_Window_Size_In_Seconds)
    spectrogram_jump_length_in_audio_samples = round(sample_rate * Spectrogram_Window_Jump_In_Seconds)
    spectrogram_fft_data_points = 2 * spectrogram_window_size_in_audio_samples  # Number of fft points; Kept 2x to increase the frequency resolution

    audio_file_analysis_entry["spectrogram_frequencies"] = librosa.fft_frequencies(sr=sample_rate, n_fft=spectrogram_fft_data_points)

    fft_spectrogram_data = librosa.stft(audio_waveform, n_fft=spectrogram_fft_data_points, win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples, window='hann')
    audio_file_analysis_entry["typed_spectrogram_data"] = {}
    audio_file_analysis_entry["typed_spectrogram_data"]["linear"] = numpy.abs(fft_spectrogram_data)
    audio_file_analysis_entry["typed_spectrogram_data"]["logarithmic"] = audio_file_analysis_entry["typed_spectrogram_data"]["linear"] ** 0.30102999566
    audio_file_analysis_entry["typed_spectrogram_data"]["decibel"] = librosa.amplitude_to_db(audio_file_analysis_entry["typed_spectrogram_data"]["linear"], ref=numpy.max)
    #NOTE [2026-05-25]: below +80.0 increment to all decibel entries effectively translates decibel entries from being a -80.0 -> 0.0 spectrum to being a 0.0 -> 80.0 spectrum, since by default the highest amplitude value in the spectrogram is set to 0.0 and spectrogram values have a minimum bound of -80.0
    audio_file_analysis_entry["typed_spectrogram_data"]["decibel"] += 80.0
    pyplot.subplot(subplot_number)
    pyplot.title(audio_file_name)
    pyplot.ylim(0, Spectrogram_Displayed_Frequency_Maximum)
    if Spectrogram_Display_Type in audio_file_analysis_entry["typed_spectrogram_data"]:
        librosa.display.specshow(audio_file_analysis_entry["typed_spectrogram_data"][Spectrogram_Display_Type], n_fft=spectrogram_fft_data_points,
                                 win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples,
                                 x_axis='time', y_axis='hz', sr=sample_rate)
    else:
        print(f"WARNING: Spectrogram_Generator is attempting to Generate_Audio_Spectrogram(), but Spectrogram_Display_Type '{Spectrogram_Display_Type}' is not recognized; no spectrogram visual will be generated")



def Generate_Audio_Spectrogram_Set(audios_analysis_data):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(Audio_File_Set) * 100) + 11

    for audio_file_name in Audio_File_Set:
        Generate_Audio_Spectrogram(audios_analysis_data[audio_file_name], Audio_Directory, audio_file_name, subplot_number)
        subplot_number += 1
    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "_spectrograms.png", dpi=Chart_Image_Resolution)
    pyplot.close()
    return audios_analysis_data