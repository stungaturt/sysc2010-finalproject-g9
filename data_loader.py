import pandas as pd
import numpy as np

def load_csv(file):
    data = pd.read_csv(file)

    time = data.iloc[:,0].values
    signal = data.iloc[:,1].values

    return time, signal