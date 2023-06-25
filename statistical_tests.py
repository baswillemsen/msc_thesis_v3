################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd

from definitions import *

from statsmodels.tsa.stattools import adfuller


# adfuller test for stationarity (unit-root test)
def adf_test(sign_level: str):

    # load data
    df = pd.read_csv(os.path.join(data_path, data_file), delimiter=',', header=0, encoding='latin-1')
    country_list = []
    series_list = []
    diff_level_list = []
    stat_list = []
    p_value_list = []

    # for every country, series combination perform adf-test until stationary
    for country in df['country'].unique():
        for series in df.columns[2:]:
            for diff_level in [0, 1, 2]:
                print(country, series, diff_level)

                country_list.append(country)
                series_list.append(series)
                diff_level_list.append(diff_level)

                if diff_level == 0:
                    df_country_series = np.log(df[df['country'] == country].set_index('date', drop=True)[series])
                else:
                    df_country_series = np.log(df[df['country'] == country].set_index('date', drop=True)[series]).diff(diff_level).dropna()
                adf_res = adfuller(df_country_series)
                p_value_list.append(adf_res[1])
                if adf_res[1] < sign_level:
                    stat_list.append(1)
                    break
                elif adf_res[1] >= sign_level:
                    stat_list.append(0)
                else:
                    raise ValueError('Adf-test not performed')

    df_stat = pd.DataFrame(list(zip(country_list, series_list, diff_level_list, stat_list, p_value_list)),
                           columns=['country', 'series', 'diff_level', 'stationary', 'p_value'])
    df_stat.to_csv(f'{output_path}stationarity_results.csv')

    # df_stat_group = df_stat[df_stat['stationary']==1]
    # print(df_stat_group.groupby(by=['series', 'stationary']).sum())
    #
    return df_stat[df_stat['stationary']==1]


if __name__ == "__main__":
    adf_test(sign_level=0.1)