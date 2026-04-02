import pandas as pd
import numpy as np

def load_csv(file):
    try: 
        data = pd.read_csv(file, header=None)

        time = data.iloc[:,0].values
        signal = data.iloc[:,1].values

        return time, signal

    except pd.errors.EmptyDataError:
        return None, None
    except Exception as e:
        return None, None