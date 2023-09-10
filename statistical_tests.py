################################
### import relevant packages ###
################################
import os
import sys
import pandas as pd
import numpy as np

from definitions import tables_path, show_results, data_path, save_results, \
    country_col, date_col
from helper_functions import read_data, get_timescale, get_trans

from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro


tables_path_cor = f'{tables_path}methodology/'


# adfuller test for stationarity (unit-root test)
def adf_test(timeframe: str, sign_level: float = 0.05):

    df = read_data(source_path=data_path, file_name=f'total_{timeframe}')
    trans = get_trans(timeframe=timeframe)

    country_list = []
    series_list = []
    log_list = []
    diff_level_list = []
    diff_order_list = []
    reg_list = []
    stat_list = []
    p_value_list = []

    # for every country, series combination perform adf-test until stationary
    for country in df[country_col].unique():
        df_country = df[df[country_col] == country].set_index(date_col, drop=True)

        for series in trans.keys():
            df_country_series = df_country[series]

            for log in [0, 1]:
                if (log == 1) and (sum(df_country_series <= 0) == 0):
                    df_country_series = np.log(df_country_series)

                diff_timescope = get_timescale(timeframe)
                for diff_level in [0, 1, 2, diff_timescope, 2*diff_timescope]:

                    for diff_order in [1, 2]:

                        # for reg in ['c', 'ct', 'ctt', 'n']:
                        for reg in ['c']:

                            # print(country, series, diff_level)
                            country_list.append(country)
                            series_list.append(series)
                            log_list.append(log)
                            diff_level_list.append(diff_level)
                            diff_order_list.append(diff_order)
                            reg_list.append(reg)

                            if diff_level == 0:
                                df_country_series_diff = df_country_series.dropna()
                                df_country_series_diff_diff = df_country_series_diff.dropna()
                            else:
                                df_country_series_diff = df_country_series.diff(periods=diff_level).dropna()

                                if diff_order == 1:
                                    df_country_series_diff_diff = df_country_series_diff.dropna()
                                elif diff_order == 2:
                                    df_country_series_diff_diff = df_country_series_diff.diff(periods=diff_level).dropna()

                            print(df_country_series_diff_diff)
                            adf_res = adfuller(x=df_country_series_diff_diff, regression=reg)
                            p_value_list.append(adf_res[1])
                            if adf_res[1] < sign_level:
                                stat_list.append(1)
                                break
                            elif adf_res[1] >= sign_level:
                                stat_list.append(0)
                            else:
                                raise ValueError('Adf-test not performed')

    df_stat = pd.DataFrame(list(zip(country_list, series_list, log_list, diff_level_list, diff_order_list,
                                    reg_list, stat_list, p_value_list)),
                           columns=['country', 'series', 'log', 'diff_level', 'diff_order',
                                    'regression', 'stationary', 'p_value'])
    df_stat_group = df_stat.groupby(by=['series', 'log', 'diff_level', 'diff_order', 'regression']).mean()

    if show_results:
        print(df_stat)
        print(df_stat_group)
    if save_results:
        df_stat.to_csv(f'{tables_path_cor}{timeframe}_stat_results.csv')
        df_stat_group.to_csv(f'{tables_path_cor}{timeframe}_stat_results_grouped.csv')


def shapiro_wilk_test(df: object, target_impl_year: int, alpha: float):
    df['diff'] = df['pred'] - df['act']
    shap = shapiro(df['diff'].loc[:target_impl_year])
    if shap[1] > alpha:
        print(f"Shapiro-Wilk test: Errors are normally distributed (p-value={round(shap[1],3)})")
    else:
        print(f"Shapiro-Wilk test: Errors NOT normally distributed (p-value={round(shap[1],3)})")


def stat_test(x: list, sign_level: float):
    adf_res = adfuller(x=x, regression='c')
    if adf_res[1] < sign_level:
        return True
    else:
        return False


if __name__ == "__main__":
    adf_test(timeframe=sys.argv[1])