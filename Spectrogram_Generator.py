import numpy
import soundfile
import matplotlib.pyplot as pyplot
import librosa

from Global_Hyperparameters import Chart_Image_Resolution, Audio_Directory, Analysis_Directory, Analysis_Run_Name, Spectrogram_Window_Size_In_Seconds, Spectrogram_Window_Jump_In_Seconds, Spectrogram_Display_Type, Spectrogram_Displayed_Frequency_Maximum

class Spectrogram_Data:
    def __init__(self, sample_rate, window_duration, jump_duration, frequencies, spectrogram):
        self.Sample_Rate = sample_rate
        self.Window_Duration = window_duration
        self.Jump_Duration = jump_duration
        self.Frequencies = frequencies
        self.Spectrogram = spectrogram

def Generate_Audio_Spectrogram(audio_directory, audio_file_name, extension=".wav"):
    audio_file_path = audio_directory + audio_file_name + extension
    audio_soundfile = soundfile.read(audio_file_path)
    audio_waveform_channels = audio_soundfile[0]
    sample_rate = audio_soundfile[1]

    if audio_waveform_channels.ndim > 1:
        audio_waveform = numpy.mean(audio_waveform_channels, axis=1)
    else:
        audio_waveform = audio_waveform_channels

    spectrogram_window_size_in_audio_samples = round(sample_rate * Spectrogram_Window_Size_In_Seconds)
    spectrogram_jump_length_in_audio_samples = round(sample_rate * Spectrogram_Window_Jump_In_Seconds)
    spectrogram_fft_data_points = 2 * spectrogram_window_size_in_audio_samples  # Number of fft points; Kept 2x to increase the frequency resolution

    spectrogram_frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=spectrogram_fft_data_points)

    fft_spectrogram_data = librosa.stft(audio_waveform, n_fft=spectrogram_fft_data_points, win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples, window='hann')
    linear_spectrogram_data = numpy.abs(fft_spectrogram_data)

    return Spectrogram_Data(sample_rate, Spectrogram_Window_Size_In_Seconds, Spectrogram_Window_Jump_In_Seconds, spectrogram_frequencies, linear_spectrogram_data)


def Generate_Audio_Spectrogram_Chart(all_audios_analysis_data):
    pyplot.figure(figsize=(20, 24))
    subplot_number = (len(all_audios_analysis_data) * 100) + 11

    for audio_analysis_entry in all_audios_analysis_data:
        spectrogram_window_size_in_audio_samples = round(audio_analysis_entry.Spectrogram_Data.Sample_Rate * Spectrogram_Window_Size_In_Seconds)
        spectrogram_jump_length_in_audio_samples = round(audio_analysis_entry.Spectrogram_Data.Sample_Rate * Spectrogram_Window_Jump_In_Seconds)
        spectrogram_fft_data_points = 2 * spectrogram_window_size_in_audio_samples  # Number of fft points; Kept 2x to increase the frequency resolution
        # displayed_spectrogram_data = None
        if Spectrogram_Display_Type == "linear":
            displayed_spectrogram_data = audio_analysis_entry.Spectrogram_Data.Spectrogram
        elif Spectrogram_Display_Type == "logarithmic":
            displayed_spectrogram_data = audio_analysis_entry.Spectrogram_Data.Spectrogram ** 0.30102999566
        elif Spectrogram_Display_Type == "decibel":
            displayed_spectrogram_data = librosa.amplitude_to_db(audio_analysis_entry.Spectrogram_Data.Spectrogram, ref=numpy.max)
            #NOTE [2026-05-25]: below +80.0 increment to all decibel entries effectively translates decibel entries from being a -80.0 -> 0.0 spectrum to being a 0.0 -> 80.0 spectrum, since by default the highest amplitude value in the spectrogram is set to 0.0 and spectrogram values have a minimum bound of -80.0
            displayed_spectrogram_data += 80.0
        else:
            print(f"WARNING: Spectrogram_Generator is attempting to Generate_Audio_Spectrogram(), but Spectrogram_Display_Type '{Spectrogram_Display_Type}' is not recognized; no spectrogram visual will be generated")
            return

        pyplot.subplot(subplot_number)
        pyplot.title(audio_analysis_entry.Audio_File_Name)
        pyplot.ylim(0, Spectrogram_Displayed_Frequency_Maximum)
        librosa.display.specshow(displayed_spectrogram_data, n_fft=spectrogram_fft_data_points,
                                 win_length=spectrogram_window_size_in_audio_samples, hop_length=spectrogram_jump_length_in_audio_samples,
                                 x_axis='time', y_axis='hz', sr=audio_analysis_entry.Spectrogram_Data.Sample_Rate)
        subplot_number += 1

    pyplot.tight_layout()
    pyplot.savefig(Analysis_Directory + Analysis_Run_Name + "_spectrograms.png", dpi=Chart_Image_Resolution)
    pyplot.close()


def Generate_Audio_Spectrogram_Set(all_audios_analysis_data):
    for audio_analysis_entry in all_audios_analysis_data:
        audio_analysis_entry.Spectrogram_Data = Generate_Audio_Spectrogram(Audio_Directory, audio_analysis_entry.Audio_File_Name)
    Generate_Audio_Spectrogram_Chart(all_audios_analysis_data)
    return all_audios_analysis_data
