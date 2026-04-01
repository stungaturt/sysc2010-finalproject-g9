from data_loader import load_csv
from functions import *

def process_signal(file, fs):
    time, signal = load_csv(file)
    signal = normalize(signal)
    filtered = bandpass(signal, fs)
    freq, fft_vals = compute_fft(filtered, fs)
    stats = features(filtered)
    return time, signal, filtered, freq, fft_vals, stats