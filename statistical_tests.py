################################
### import relevant packages ###
################################
import os
import pandas as pd
import numpy as np

from definitions import tables_path, show_results, data_path, sign_level, save_results
from helper_functions import read_data

from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro


tables_path_cor = f'{tables_path}methodology/'


# # adfuller test for stationarity (unit-root test)
# def adf_test(sign_level: str):
#
#     # load data
#     df = pd.read_csv(os.path.join(data_path, data_file), delimiter=',', header=0, encoding='latin-1')
#     country_list = []
#     series_list = []
#     diff_level_list = []
#     stat_list = []
#     p_value_list = []
#
#     # for every country, series combination perform adf-test until stationary
#     for country in df['country'].unique():
#         for series in df.columns[2:]:
#             for diff_level in [0, 1, 2]:
#                 print(country, series, diff_level)
#
#                 country_list.append(country)
#                 series_list.append(series)
#                 diff_level_list.append(diff_level)
#
#                 if diff_level == 0:
#                     df_country_series = np.log(df[df['country'] == country].set_index('date', drop=True)[series])
#                 else:
#                     df_country_series = np.log(df[df['country'] == country].set_index('date', drop=True)[series]).diff(diff_level).dropna()
#                 adf_res = adfuller(df_country_series)
#                 p_value_list.append(adf_res[1])
#                 if adf_res[1] < sign_level:
#                     stat_list.append(1)
#                     break
#                 elif adf_res[1] >= sign_level:
#                     stat_list.append(0)
#                 else:
#                     raise ValueError('Adf-test not performed')
#
#     df_stat = pd.DataFrame(list(zip(country_list, series_list, diff_level_list, stat_list, p_value_list)),
#                            columns=['country', 'series', 'diff_level', 'stationary', 'p_value'])
#     df_stat.to_csv(f'{output_path}stationarity_results.csv')
#
#     df_stat_group = df_stat[df_stat['stationary']==1]
#     print(df_stat_group.groupby(by=['series', 'stationary']).sum())
#
#     return df_stat[df_stat['stationary'] == 1]


# adfuller test for stationarity (unit-root test)
def adf_test(df: object, country_col: str, time_col: str, sign_level: str):

    country_list = []
    series_list = []
    log_list = []
    diff_level_list = []
    reg_list = []
    stat_list = []
    p_value_list = []

    # for every country, series combination perform adf-test until stationary
    for country in df[country_col].unique():
        df_country = df[df['country'] == country].set_index(time_col, drop=True)

        # for series in df_country.drop(country_col, axis=1).columns:
        for series in ['co2', 'gdp', 'pop']:
            df_country_series = df_country[series]

            for log in [0, 1]:
                if (log == 1) and (sum(df_country_series <= 0) == 0):
                    df_country_series = np.log(df_country_series)

                for diff_level in [0, 1, 2]:

                    # for reg in ['c', 'ct', 'ctt', 'n']:
                    for reg in ['c']:

                        # print(country, series, diff_level)
                        country_list.append(country)
                        series_list.append(series)
                        log_list.append(log)
                        diff_level_list.append(diff_level)
                        reg_list.append(reg)

                        if diff_level == 0:
                            df_country_series_diff = df_country_series.dropna()
                        else:
                            df_country_series_diff = df_country_series.diff(periods=diff_level).dropna()

                        adf_res = adfuller(x=df_country_series_diff, regression=reg)
                        p_value_list.append(adf_res[1])
                        if adf_res[1] < sign_level:
                            stat_list.append(1)
                            break
                        elif adf_res[1] >= sign_level:
                            stat_list.append(0)
                        else:
                            raise ValueError('Adf-test not performed')

    df_stat = pd.DataFrame(list(zip(country_list, series_list, log_list, diff_level_list, reg_list,
                                    stat_list, p_value_list)),
                           columns=['country', 'series', 'log', 'diff_level', 'regression', 'stationary', 'p_value'])
    df_stat_group = df_stat.groupby(by=['series', 'log', 'diff_level', 'regression']).mean()

    if show_results:
        print(df_stat)
        print(df_stat_group)
    if save_results:
        df_stat.to_csv(f'{tables_path_cor}stationarity_results.csv')
        df_stat_group.to_csv(f'{tables_path_cor}stationarity_results_grouped.csv')


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
    df = read_data(source_path=data_path, file_name='total_m')
    adf_test(df=df, country_col='country', time_col='date', sign_level=sign_level)