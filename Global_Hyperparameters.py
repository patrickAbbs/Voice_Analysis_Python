
Chart_Image_Resolution = 250

#NOTE [2026-05-25]: below window size and jump are somewhat arbitrary, and there are meaningful differences even for relatively small alterations. From some quick searching, it sounds like "standard practice" for speech tends to be around ~0.025 size and ~0.01 jump (with my current go-to being 0.05 size and 0.01 jump), which makes the frequency resolution significantly blurrier than my current go-to but apparently tends to be better for surfacing rapid phonemes such as plosives
Spectrogram_Window_Size_In_Seconds = 0.05  # 50ms
Spectrogram_Window_Jump_In_Seconds = 0.01  # 8ms
Spectrogram_Displayed_Frequency_Maximum = 4000
Spectrogram_Display_Type = "decibel"

Frequency_Distribution_Bucket_Range = 250.0
Frequency_Distribution_Bucket_Increment = 25.0
Frequency_Distribution_Displayed_Frequency_Maximum = 4000
Frequency_Distribution_Frequency_Maximum = 10000
Frequency_Distribution_Display_Type = "decibel"




Audio_Directory = "../voice_modulation_audio/"
Audio_File_Set = ["normal_1", "no_nose_1", "round_1", "scratchy_1", "throat_close_1"]
#Audio_File_Set = ["normal_1"]
#Audio_File_Set = ["round_1", "round_2", "round_3"]

Analysis_Directory = "tmp/media/voice_modulation/"
Analysis_Run_Name = "all_1s_decibel_distribution_test"



