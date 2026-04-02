import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("env_temp_humidity_clean.csv")
df.columns = df.columns.str.strip()
print("Columns:", df.columns)
temp = "signal"

signal = df[temp]
noise = np.random.normal(0, 0.02 * signal.std(), size=len(signal))
df[temp] = signal + noise

df.to_csv("temp_raw.csv", index=False)

plt.plot(signal, label="Original")
plt.plot(df[temp], label="Noisy", alpha=0.7)
plt.legend()
plt.title("EMG Signal with Noise")
plt.show()