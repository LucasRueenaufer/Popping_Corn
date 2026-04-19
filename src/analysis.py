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
generate_frec_plots = True

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
    
    
    
    
    # finding peaks using scipy. Choose peak-threshhold of 4 times over mean
    threshhold = np.mean(energy) * 4
    peaks = find_peaks(energy, height=threshhold)[0]
    times = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=hop_length)
    
    # optionally print energy over time with threshhold
    times_energy = librosa.frames_to_time(np.arange(len(energy)),sr=sample_rate,hop_length=hop_length)
    if generate_frec_plots:
        plt.plot(times_energy,energy)
        plt.axhline(y=threshhold, color='r', linestyle='--')
        plt.show()
        
    return times

def data_analysis():
    """
    SUMMARY.

    Returns
    -------
    None
    """
    # find my mp3 files

    mp3_files = list(data_folder.glob("*.mp3"))
    
    #big data extraction for each file
    for file_path in mp3_files:
        
        # extract time data of popping events
        print("Processing:", file_path.name)
        timestamp_events = extract_data_from_mp3(file_path)
        
        # create gaussian fit to data
        fit_time = np.linspace(min(timestamp_events),max(timestamp_events),1024)
        fit_mean, fit_sigma = norm.fit(timestamp_events)
        print(fit_mean,fit_sigma)
        
        #plot histogram
        plt.hist(timestamp_events,bins=20,density=True,label="Registered Pops")
 
        #plot fit
        plt.title(f"Popping against time for {file_path.stem}",)
        plt.xlabel("Time to Pop in s")
        plt.ylabel("Frequency of Pops")
        plt.plot(fit_time,norm.pdf(fit_time,fit_mean,fit_sigma),label=f"Best Fit $N({fit_mean:.2f},{fit_sigma:.2f})$")
        plt.legend()
        
        plt.show()
    
    pass

data_analysis()

