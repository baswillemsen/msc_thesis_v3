################################
### import relevant packages ###
################################
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from definitions import paths, verbatim

data_path, figures_path, output_path = paths()
pr_results, save_figs, show_plots = verbatim()


def pivot_target(df: object, target_country: str, target_var: str):
    return df[df['country'] == target_country][target_var]


def pivot_donors(df: object, donor_countries: list):
    donors = df.copy()
    donors = donors[donors['country'].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index='date', columns=['country'], values=donors.columns[2:])
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)
    donors = donors.dropna(axis=1)
    return donors


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
