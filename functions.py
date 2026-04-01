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

def lowpass(signal, fs, cutoff = 5):
      b, a = sc.butter(4, cutoff / (fs / 2), btype='low')
      return sc.filtfilt(b,a, signal)

def highpass(signal, fs, cutoff = 0.5):
      b, a = sc.butter(4, cutoff / (fs / 2), btype='high')
      return sc.filtfilt(b,a, signal)

def bandpass(signal, fs, lowc = 0.5, ):
      b, a = sc.butter(4, cutoff / (fs / 2), btype='low')
      return sc.filtfilt(b,a, signal)

def compute_fft(signal, fs):
      n = len(signal)
      freq = np.fft.fftfreq(n, d = 1/fs)
      fft_value = np.abs(np.fft.fft(signal))

      return freq[:n//2], fft_value[:n//2]

