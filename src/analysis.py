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
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

data_folder =Path("../data/")
generate_frec_plots = False

def extract_data_from_mp3(path):
    """
    Loads the mp3 recording of the popcorn and extracts a numpy array
    with timestamps of popping events.    
    
    Parameters
    ----------
    name : Path object
        filename.

    Returns
    -------
    times : numpy.array
        array with timestamps of popcorn events
    """
    
    # Define Kernel size for energy calc
    frame_length=64
    hop_length=frame_length
    
    # Loading mp3 and introducing rolling average of waveenergy
    waveform, sample_rate = librosa.load(path)
    energy = librosa.feature.rms(y=waveform,frame_length=frame_length, hop_length=hop_length)[0]
    
    #additional frec plots by generating timing
    times = librosa.frames_to_time(np.arange(len(energy)),sr=sample_rate,hop_length=hop_length)
    
    # print energy over time
    if generate_frec_plots:
        plt.plot(times,energy)
        plt.show()
        
   
    
    # finding peaks using scipy. Choose peak-threshold of 4 times over mean
    threshhold = np.mean(energy) * 4
    peaks = find_peaks(energy, height=threshhold)[0]
    times = librosa.frames_to_time(peaks, sr=sample_rate)
    return times

def data_analysis():
    mp3_files = list(data_folder.glob("*.mp3"))
    for file_path in mp3_files:
        print("Processing:", file_path.name)
        timestamp_events = extract_data_from_mp3(file_path)
        plt.hist(timestamp_events,cumulative=True,bins=20)
        plt.title(f"Popping against time for {file_path.stem}",)
        plt.xlabel("Time in s")
        plt.ylabel("Pop Events")
        plt.show()
        plt.hist(np.diff(timestamp_events),bins=10)
        plt.show()
    
    pass

data_analysis()

