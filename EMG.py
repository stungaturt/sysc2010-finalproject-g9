import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EMG.csv")
df.columns = df.columns.str.strip()
print("Columns:", df.columns)
emg_col = "underhand"

signal = df[emg_col]
jitter = np.diff(signal, prepend=signal.iloc[1])
noise = np.random.normal(0, 0.02 * signal.std(), size=len(signal))
df[emg_col] = signal + noise + 0.5*jitter

df.to_csv("EMG_raw.csv", index=False)

plt.plot(signal, label="Original")
plt.plot(df[emg_col], label="Noisy", alpha=0.7)
plt.legend()
plt.title("EMG Signal with Noise")
plt.show()