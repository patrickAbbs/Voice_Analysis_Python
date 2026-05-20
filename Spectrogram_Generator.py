import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Chart_Image_Resolution, Audio_Directory, Audio_File_Set, Analysis_Directory, Analysis_Run_Name

Window_Size_In_Seconds = 0.05  # 50ms
Window_Jump_In_Seconds = 0.008  # 8ms

Displayed_Frequency_Maximum = 4000
Spectrogram_Display_Type = "decibel"

def Generate_Audio_Spectrogram(audio_directory, audio_file_name, subplot_number):
    audio_file_path = audio_directory + audio_file_name + ".wav"
    audio_soundfile = soundfile.read(audio_file_path)
    audio_waveform_channels = audio_soundfile[0]
    sample_rate = audio_soundfile[1]

    if audio_waveform_channels.ndim > 1:
        audio_waveform = numpy.mean(audio_waveform_channels, axis=1)
    else:
        audio_waveform = audio_waveform_channels

    spectrogram_window_size_in_audio_samples = round(sample_rate * Window_Size_In_Seconds)
    spectrogram_jump_length_in_audio_samples = round(sample_rate * Window_Jump_In_Seconds)
    spectrogram_fft_data_points = 2 * spectrogram_window_size_in_audio_samples  # Number of fft points; Kept 2x to increase the frequency resolution

    fft_spectrogram_data = librosa.stft(audio_waveform, n_fft=spectrogram_fft_data_points, win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples, window='hann')
    typed_spectrogram_data = {}
    typed_spectrogram_data["linear"] = numpy.abs(fft_spectrogram_data)
    typed_spectrogram_data["logarithmic"] = typed_spectrogram_data["linear"] ** 0.30102999566
    typed_spectrogram_data["decibel"] = librosa.amplitude_to_db(typed_spectrogram_data["linear"], ref=numpy.max)
    pyplot.subplot(subplot_number)
    pyplot.title(audio_file_name)
    pyplot.ylim(0, Displayed_Frequency_Maximum)
    if(Spectrogram_Display_Type in typed_spectrogram_data):
        librosa.display.specshow(typed_spectrogram_data[Spectrogram_Display_Type], n_fft=spectrogram_fft_data_points,
                                 win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples,
                                 x_axis='time', y_axis='hz', sr=sample_rate)
    else:
        print(f"WARNING: Spectrogram_Generator is attempting to Generate_Audio_Spectrogram(), but Spectrogram_Display_Type '{Spectrogram_Display_Type}' is not recognized; no spectrogram visual will be generated")

    return typed_spectrogram_data



def Generate_Audio_Spectrogram_Set(audios_analysis_data):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(Audio_File_Set) * 100) + 11

    for audio_file_name in Audio_File_Set:
        audios_analysis_data[audio_file_name]["typed_spectrogram_data"] = Generate_Audio_Spectrogram(Audio_Directory, audio_file_name, subplot_number)
        subplot_number += 1
    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "_spectrograms.png", dpi=Chart_Image_Resolution)
    pyplot.close()
    return audios_analysis_data