################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller

# adfuller test for stationarity (unit-root test)
def adf_test(df: object):
    print("Running adf test for all columns in dataset")
    for col in df.columns:
        donor_series = df[col]
        adf_test = adfuller(donor_series)
        if adf_test[1] < 0.05:
            print(f'{col}: Stationary')
        elif adf_test[1] >= 0.05:
            print(f'{col}: Non-stationary ({adf_test[1]})')
        else:
            raise ValueError('Adf-test not performed')