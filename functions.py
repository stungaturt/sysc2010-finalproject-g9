import numpy as np
import matplotlib as plt
import pandas as pd
import scipy.signal as sc
from data_loader import *


def remove_baseline(signal):
    '''Remove the baseline from a provided signal'''
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
        
    mean_val = sum(signal) / len(signal)
    return [x - mean_val for x in signal]


def normalize(signal):
    '''Normalize a provided signal'''
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    max_val = max(abs(x) for x in signal)
    if max_val == 0:
        return signal
    return [x / max_val for x in signal]

def lowpass(signal, fs, cutoff):
    "Does a lowpass filter for the giving signal"
    b, a = sc.butter(4, cutoff / (fs / 2), btype='low')
    return sc.filtfilt(b,a, signal)

def highpass(signal, fs, cutoff):
    "Does a highpass filter for the giving signal"
    b, a = sc.butter(4, cutoff / (fs / 2), btype='high')
    return sc.filtfilt(b,a, signal)

def bandpass(signal, fs, lowc, highc):
    "Does a bandpass filter for the giving signal"
    b, a = sc.butter(4, [lowc/(fs/2), highc/(fs/2)], btype='band')
    return sc.filtfilt(b,a, signal)

def compute_fft(signal, fs):
    n = len(signal)
    freq = np.fft.fftfreq(n, d = 1/fs)
    fft_value = np.abs(np.fft.fft(signal))

    return freq[:n//2], fft_value[:n//2]

def detect_peaks(signal, threshold):
    '''
    Return list of peaks in an ECG signal
        
    A peak is such that it is a point above the threshold value,
    and is greater than the values to its left and right
    '''
    if len(signal) < 3:
        return []
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return peaks

def rr_intervals(peaks, fs):
    '''
    Calculate the time between consecutive peaks (RR intervals)

    peaks is a list of detected peak positions
    fs is the sampling frequency

    Returns a list of RR intervals in seconds
    '''
    intervals = []
    for i in range(1, len(peaks)):
        interval = (peaks[i]-peaks[i-1])/fs
        intervals.append(interval)

    return intervals

def heart_rate(peaks, fs):
    '''
    peaks is a list of detected peak positions
    fs is the sampling frequency
        
    This function should return the heart rate in (BPM)
    '''
    if len(peaks) < 2:
        raise ValueError("Need more peaks to calculate Heart Rate")
            
    rr = rr_intervals(peaks, fs)
    avg_rr = sum(rr)/len(rr)
    heartRate = 60/avg_rr
        
    return heartRate

def features(signal):
    return{"mean": np.mean(signal),
           "std": np.std(signal),
           "rms": np.sqrt(np.mean(signal**2))}