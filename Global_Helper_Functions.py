import math


def Convert_Half_Life_To_Cumulation_Weight(processing_window_duration, half_life):
    return math.pow(0.5, processing_window_duration / half_life)


def Weighted_Average(value_1, weight_1, value_2, weight_2):
    return (value_1 * weight_1 + value_2 * weight_2) / (weight_1 + weight_2)
