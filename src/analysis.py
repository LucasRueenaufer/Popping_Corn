#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:27:10 2026

@author: LucasR

This project aims to quantify the popping of popcorn by analizing audio
recordings of the cooking process. We will use librosa to handle the 
mp3 files and will continue the data science part with SciPy and NumPy
"""

import librosa
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

data_location ="../data/"

def extract_data_from_mp3(name: str):
    """
    Loads the mp3 recording of the popcorn and extracts a numpy array
    with timestamps of popping events.    
    
    Parameters
    ----------
    name : str
        filename.

    Returns
    -------
    times : numpy.array
        array with timestamps of popcorn events
    """
    
    # Loading mp3 and introducing rolling average of waveenergy
    waveform, sample_rate = librosa.load(data_location+name)
    energy = librosa.feature.rms(y=waveform)[0]
    
    # finding peaks using scipy. Choose peak-threshold of 2.5 times over mean
    threshhold = np.mean(energy) * 2.5
    peaks = find_peaks(energy, height=threshhold)[0]
    times = librosa.frames_to_time(peaks, sr=sample_rate)
    return times

plt.hist(extract_data_from_mp3("Popping1.mp3"))
plt.show()
