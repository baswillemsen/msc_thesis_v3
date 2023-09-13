################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from definitions import data_source_path, data_path, corr_country_names, sign_level, fake_num, \
    country_col, year_col, quarter_col, month_col, date_col
from helper_functions import read_data, select_country_year_measure, month_name_to_num, rename_order_scale, \
    downsample_month_to_quarter, quarter_to_month, upsample_quarter_to_month, get_timeframe_col, get_trans
from statistical_tests import stat_test


# Monthly CO2 data
def preprocess_co2_m(source_file: str, source_country_col: str, source_year_col: str, var_name: str):
    # read data
    co2_m_raw = read_data(source_path=data_source_path, file_name=source_file)
    co2_m = co2_m_raw.copy()

    # lowercase, replace country names
    co2_m[source_country_col] = co2_m[source_country_col].str.lower()
    co2_m = co2_m.replace({source_country_col: corr_country_names})

    # select countries and year
    co2_m = select_country_year_measure(df=co2_m, country_col=source_country_col, year_col=source_year_col)
    # pivot
    co2_m = co2_m.melt(id_vars=[source_country_col, source_year_col],
                       value_vars=co2_m.drop([source_country_col, source_year_col], axis=1),
                       value_name=var_name)
    co2_m[month_col] = co2_m.apply(lambda row: month_name_to_num(row.variable), axis=1)
    co2_m[date_col] = pd.to_datetime(dict(year=co2_m[source_year_col], month=co2_m[month_col], day=1))
    co2_m = co2_m.drop('variable', axis=1)

    # rename, order and scale: output = [index, country, date, value]
    co2_m = rename_order_scale(df=co2_m, source_country_col=source_country_col, source_year_col=source_year_col,
                               var_name=var_name, var_scale=1e6, timeframe='m')
    # downsample monthly to quarterly
    co2_q = downsample_month_to_quarter(df_m=co2_m, var_name=var_name)

    # export to csv
    co2_m.to_csv(f'{data_path}{var_name}_m.csv')
    co2_q.to_csv(f'{data_path}{var_name}_q.csv')

    return co2_m, co2_q


# Quarterly GDP data
def preprocess_WB_q(source_file: str, source_country_col: str, source_time_col: str,
                    source_measure_col: str, source_incl_measure: list, var_name: str):
    # read data
    df_q_raw = read_data(source_path=data_source_path, file_name=source_file)
    df_q = df_q_raw.copy()

    # lowercase, replace country names
    df_q[source_country_col] = df_q[source_country_col].str.lower()
    df_q = df_q.replace({source_country_col: corr_country_names})

    # transform
    df_q[year_col] = df_q[source_time_col].str[:4].astype(int)
    df_q[quarter_col] = df_q[source_time_col].str[6:].astype(int)
    df_q[month_col] = df_q.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    df_q[date_col] = pd.to_datetime(dict(year=df_q[year_col], month=df_q[month_col], day=1)).dt.to_period('M')
    # df_q[date_col] = pd.to_datetime(dict(year=df_q[year_col], month=df_q[month_col], day=1))
    df_q[var_name] = df_q['Value']

    # select countries and year
    df_q = select_country_year_measure(df=df_q, country_col=source_country_col, year_col=year_col,
                                       measure_col=source_measure_col, incl_measure=source_incl_measure)

    # rename, order and scale: output = [index, country, date, value]
    df_q = rename_order_scale(df=df_q, source_country_col=source_country_col, source_year_col=year_col,
                              timeframe='q', var_name=var_name, var_scale=1e6)

    # upsample monthly to quarterly
    df_m = upsample_quarter_to_month(df_q=df_q, var_name=var_name)
    df_q[date_col] = pd.to_datetime(dict(year=df_q[year_col], month=df_q[quarter_col].apply(quarter_to_month), day=1))

    # export to csv
    df_q.to_csv(f'{data_path}{var_name}_q.csv')
    df_m.to_csv(f'{data_path}{var_name}_m.csv')

    return df_m, df_q


def total_join(co2: object, pop: object, gdp: object, key_cols: list, timeframe: str):
    total = co2.copy()
    total = total.merge(gdp, how='left', on=key_cols)

    total = total.merge(pop, how='left', on=key_cols)

    total[f'co2_cap'] = total[f'co2'] / total[f'pop']
    total[f'gdp_cap'] = total[f'gdp'] / total[f'pop']

    total = total.dropna(axis=0, how='any').reset_index(drop=True)
    total.to_csv(f'{data_path}total_{timeframe}.csv', header=True, index=False)

    return total


def make_stat(df: object, timeframe: str):

    for stat in ['stat', 'non_stat']:

        country_list = []
        date_list = []
        year_list = []
        period_list = []
        period_col = get_timeframe_col(timeframe=timeframe)
        trans = get_trans(timeframe=timeframe)

        # cov = df.columns.drop(['country', 'year'])
        vars = trans.keys()
        for series in vars:
            globals()[f"{series}_list"] = []

        for country in df[country_col].unique():

            df_country = df[df[country_col] == country]
            country_list += list(df_country[country_col])
            date_list += list(df_country[date_col])
            year_list += list(df_country[year_col])
            period_list += list(df_country[period_col])

            for series in vars:
                df_country_series = df_country[series]
                log, diff_level, diff_order = trans[series]

                # log the series if necessary
                if log:
                    df_country_series_log = np.log(df_country_series)
                else:
                    df_country_series_log = df_country_series

                # difference the series
                i = 1
                df_country_series_diff = df_country_series_log.copy()
                if diff_level != 0:
                    while i <= diff_order:
                        df_country_series_diff = df_country_series_diff.diff(periods=diff_level)
                        i += 1

                # if diff_level == 0:
                #     df_country_series_diff = df_country_series_log
                #     df_country_series_diff_diff = df_country_series_diff
                # else:
                #     df_country_series_diff = df_country_series.diff(periods=diff_level)
                #
                #     if diff_order == 1:
                #         df_country_series_diff_diff = df_country_series_diff
                #     elif diff_order == 2:
                #         df_country_series_diff_diff = df_country_series_diff.diff(periods=diff_level)

                # if series is non-stationary input fake number -99999
                if stat == 'stat':
                    if stat_test(x=df_country_series_diff.dropna(), sign_level=sign_level) == 'stationary':
                        globals()[f"{series}_list"] += list(df_country_series_diff)
                    elif stat_test(x=df_country_series_diff.dropna(), sign_level=sign_level) == 'non_stationary':
                        globals()[f"{series}_list"] += [fake_num]*len(df_country_series_diff)

                elif stat == 'non_stat':
                    globals()[f"{series}_list"] += list(df_country_series_diff)
                else:
                    raise ValueError('Define stat as being "stat" or "non_stat"')

        # put together in dataframe
        total_stat = pd.DataFrame(list(zip(country_list, date_list, year_list, period_list)),
                                  columns=[country_col, date_col, year_col, period_col])
        total_stat['co2'] = co2_list
        total_stat['gdp'] = gdp_list
        total_stat['pop'] = pop_list
        # total_stat['co2_cap'] = co2_cap_list
        # total_stat['gdp_cap'] = gdp_cap_list

        total_stat = total_stat.dropna(axis=0, how='any').reset_index(drop=True)
        total_stat.to_csv(f'{data_path}total_{timeframe}_{stat}.csv', header=True, index=False)

    return total_stat


def preprocess():
    co2_m, co2_q = preprocess_co2_m(source_file='co2_m_2000_2021',
                                    source_country_col='Name',
                                    source_year_col='Year',
                                    var_name='co2'
                                    )

    gdp_m, gdp_q = preprocess_WB_q(source_file='gdp_q_1990_2022',
                                   source_country_col='Country',
                                   source_time_col='TIME',
                                   source_measure_col='MEASURE',
                                   source_incl_measure=['CPCARSA'],
                                   var_name='gdp'
                                   )

    pop_m, pop_q = preprocess_WB_q(source_file='pop_q_1995_2022',
                                   source_country_col='Country',
                                   source_time_col='TIME',
                                   source_measure_col='MEASURE',
                                   source_incl_measure=['PERSA'],
                                   var_name='pop'
                                   )

    # total monthly
    timeframe = 'm'
    total_m = total_join(co2=co2_m, pop=pop_m, gdp=gdp_m,
                         key_cols=[country_col, date_col, year_col, month_col], timeframe=timeframe)
    total_m_stat = make_stat(df=total_m, timeframe=timeframe)
    print(total_m_stat)

    # total quarterly
    timeframe = 'q'
    total_q = total_join(co2=co2_q, pop=pop_q, gdp=gdp_q,
                         key_cols=[country_col, date_col, year_col, quarter_col], timeframe=timeframe)
    total_q_stat = make_stat(df=total_q, timeframe=timeframe)
    print(total_q_stat)


if __name__ == "__main__":
    preprocess()