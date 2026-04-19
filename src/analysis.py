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
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path

data_folder =Path("../data/")
generate_frec_plots = True
generate_fit_plots = True
answer_verbose = True

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
    frame_length=256
    hop_length=frame_length
    
    # Loading mp3 and introducing rolling average of waveenergy
    waveform, sample_rate = librosa.load(path)
    energy = librosa.feature.rms(y=waveform,frame_length=frame_length, hop_length=hop_length)[0]
    
    # finding peaks using scipy. Choose peak-threshhold of 5 times over mean
    threshhold = np.mean(energy) * 5
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
    Runs the main analysis of the data on all datasets.

    Returns
    -------
    None
    """
    # read experimental data from file
    data = np.genfromtxt(data_folder/"data.csv", delimiter=",", dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    
    #init array with results
    results=[]
    
    # extract everything for each row in data
    for row in data:
        filename, num_total_exp, num_missed_exp = row
        
        file_path = data_folder / filename
        # extract time data of popping events
        if answer_verbose:
            print("Processing:", file_path.name)
        timestamp_events = extract_data_from_mp3(file_path)
        num_pops = len(timestamp_events)
        
        # Call for fit on right truncated gaussian
        fit_parameters=fit_trunc_gaussian(timestamp_events)
        
        # Determine number of kernels missed due to cutoff by using CDF
        num_missed = int(num_pops/norm.cdf((np.max(timestamp_events)-fit_parameters[0])/fit_parameters[1])-num_pops)
        
        #optional histogram plotting
        if generate_fit_plots:
            # create time-array for plotting
            fit_time = np.linspace(min(timestamp_events),max(timestamp_events),1024)
        
            #plot histogram
            plt.hist(timestamp_events,bins=20,density=True,label="Registered Pops")
 
            #plot fit
            plt.title(f"Popping against time for {file_path.stem}",)
            plt.xlabel("Time to Pop in s")
            plt.ylabel("Frequency of Pops")
            plt.plot(fit_time,trunc_gaussian(fit_time,fit_parameters),label=f"Best Fit for \nmu={fit_parameters[0]:.2f}, sig={fit_parameters[1]:.2f}")
            plt.legend()
            plt.show()
        
        # We can estimate that ~9%  of kernes do not pop at all. We can now estimate
        # total kernals, popped kernels and unpopped kernels. Print only if needed
        
        if answer_verbose:

            num_total = num_total_exp
            print("# Total Kernels: ",num_total)
            print("# Popped Kernels: ",num_pops,"(",round(num_pops/num_total*100,1),"%)")
            print("# Unpopped Kernels: ",num_total-num_pops,"(",round(num_pops/num_total*100,1),"%)\nOf which are:")
            print("\t Missed Kernels: ",num_missed,"(",round(num_missed/num_total*100,1),"%)")
            print("\t Unpoppable Kernels: ",num_total-num_missed-num_pops,"(",round((num_total-num_missed-num_pops)/num_total*100,1),"%)")
        
        #save important values from dataset
        results.append([num_pops, num_missed,num_total_exp, num_missed_exp])
    pass

def fit_trunc_gaussian(timestamp_events):
    """
    SUMMARY.

    Parameters
    ----------
    timestamp_events : TYPE
        DESCRIPTION.

    Returns
    -------
    tuple : (mean, sigma)
    """
    
    # select starting parameters for minimization by naive gaussian fit
   
    mean_init, sigma_init = norm.fit(timestamp_events)
    
    # Minimize negative log-likelihood through scipy.minimize
    result = minimize(
        negative_log_likelihood,
        x0=[mean_init,sigma_init],
        args=(timestamp_events),
        bounds=[(None, None), (1e-6, None)]
        )
    
    return result.x #return results of minimization

def trunc_gaussian(x,parameters):
    """
    Just gives back a gaussian, which is truncated at the right side
    (highest value for x)

    Parameters
    ----------
    x : TYPE
        array of x-axis values
    parameters : tuple
        (mean, sigma)
    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    #extract parameters
    mean, sigma = parameters
    
    #return right truncated gaussian
    return norm.pdf(x, mean, sigma)/norm.cdf((np.max(x)-mean)/sigma)

def negative_log_likelihood(parameters,timestamps_events):
    """
    Returns the negative log-likelihood for a truncated gaussian

    Parameters
    ----------
    parameters : tuple
        (mean, sigma)
    timestamps_events : array
        timestamp data

    Returns
    -------
    returns float with neg log likelihood
    """
    # extract parameters
    mean, sigma = parameters
    
    # small sanity check
    if sigma <=0:
        print("Something went wrong while fitting.")
        pass
    
    # pass parameters to pdf ad cdf function
    log_pdf = norm.logpdf(timestamps_events, mean, sigma)
    log_cdf = norm.logcdf((np.max(timestamps_events)-mean)/sigma)
    
    return -(np.sum(log_pdf)-len(timestamps_events)*log_cdf) #return the negative
    # of the logarithm of the log likelihood
    
data_analysis()

