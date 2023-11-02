# import relevant packages
import pandas as pd
import numpy as np

from definitions import show_output, save_output, country_col, date_col, timeframe_val, sign_level
from util_general import read_data, get_timescale, get_trans, get_impl_date, get_data_path, get_table_path

from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from scipy.stats import ttest_1samp, ttest_ind, norm
from statsmodels.stats.stattools import durbin_watson


# adfuller test for stationarity (unit-root test)
def adf_test(sign_level: float = sign_level):

    for timeframe in timeframe_val:
        table_path_meth = get_table_path(timeframe=timeframe, folder='methodology')

        df = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}')
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
                if sum(df_country_series <= 0) != 0:
                    continue
                else:
                    for log in [0, 1]:
                        if log == 1:
                            df_country_series = np.log(df_country_series)

                        diff_timescope = get_timescale(timeframe)
                        for diff_level in [0, 1, 2, diff_timescope, 2*diff_timescope]:

                            for diff_order in [1, 2]:

                                for reg in ['c', 'ct', 'ctt', 'n']:

                                    print(timeframe, country, series, diff_level, diff_order)
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

        if show_output:
            print(df_stat)
            print(df_stat_group)
        if save_output:
            df_stat.to_csv(f'{table_path_meth}/{timeframe}_stat_results.csv')
            df_stat_group.to_csv(f'{table_path_meth}/{timeframe}_stat_results_grouped.csv')


# shapiro wilk test for normality of the residuals
def shapiro_wilk_test(df: object, treatment_country: str, alpha: float):
    shap = shapiro(df['error'].loc[:get_impl_date(treatment_country)])
    if shap[1] > alpha:
        normal_errors = 1
        print(f'Shapiro-Wilk test: Errors are normally distributed (p-value={round(shap[1],3)})')
    else:
        normal_errors = 0
        print(f'Shapiro-Wilk test: Errors NOT normally distributed (p-value={round(shap[1],3)})')

    return normal_errors, shap[1]


# Durbin Watson test for non-serially correlated errors
def durbin_watson_test(df: object, treatment_country: str, alpha: float):
    db = durbin_watson(df['error'].loc[:get_impl_date(treatment_country)])
    if 1 < db < 3:
        serial_corr = 0
        print(f'Shapiro-Wilk test: Errors are NOT serially correlated  (p-value={round(db,3)})')
    else:
        serial_corr = 1
        print(f'Shapiro-Wilk test: Errors are serially correlated  (p-value={round(db,3)})')

    return serial_corr, db


def stat_test(x: list, sign_level: float):
    adf_res = adfuller(x=x, regression='c')
    if adf_res[1] < sign_level:
        return 'stationary'
    else:
        return 'non_stationary'


# one-sample t-test to see if results are significant
def sign_test_result(df: object, treatment_country: str, hyp: bool = None):

    df_pre = df[df.index < get_impl_date(treatment_country=treatment_country)]
    df_post = df[df.index >= get_impl_date(treatment_country=treatment_country)]

    att_mean = df_post['error'].mean()
    att_std = df_post['error'].std()
    print(f'ATT (mean): {round(att_mean,4)}')
    print(f'ATT (std):  {round(att_std,4)}')

    if hyp is None:
        p_value = ttest_1samp(df_post['error'], popmean=0).pvalue
    else:
        T = len(df)
        T_pre = len(df_pre)
        T_post = len(df_post)
        gamma_1 = sum(df_pre['error']**2) / T_pre
        gamma_2 = sum(df_post['error'] ** 2) / T_post
        omega_hat_sqrt = np.sqrt(gamma_1 / (T_pre/T) + gamma_2 / (T_post/T))
        test_stat = att_mean / (omega_hat_sqrt / np.sqrt(T))
        p_value = 2 * (1 - norm.cdf(abs(test_stat), loc=0, scale=1))

    if p_value < sign_level:
        significant = 1
        print(f'Result is significant (p-value={p_value})')
    else:
        significant = 0
        print(f'Result is NOT significant (p-value={p_value})')

    return att_mean, att_std, significant, p_value


if __name__ == "__main__":
    adf_test()