################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd

from definitions import data_source_path, corr_country_names, sign_level, fake_num, \
        country_col, year_col, quarter_col, month_col, date_col, incl_countries, show_results
from helper_functions import read_data, select_country_year_measure, month_name_to_num, rename_order_scale, \
    downsample_month_to_quarter, quarter_to_month, upsample_quarter_to_month, get_timeframe_col, get_trans, \
    get_data_path, get_fig_path, interpolate_series
from statistical_tests import stat_test
from plot_functions import plot_series


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
    co2_m[month_col] = co2_m.apply (lambda row: month_name_to_num(row.variable), axis=1)
    co2_m[date_col] = pd.to_datetime(dict(year=co2_m[source_year_col], month=co2_m[month_col], day=1))
    co2_m = co2_m.drop('variable', axis=1)

    # rename, order and scale: output = [index, country, date, value]
    co2_m = rename_order_scale(df=co2_m, source_country_col=source_country_col, source_year_col=source_year_col,
                               var_name=var_name, var_scale=1e6, timeframe='m')
    # downsample monthly to quarterly
    co2_q = downsample_month_to_quarter(df_m=co2_m, var_name=var_name, agg='sum')

    # export to csv
    co2_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    co2_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return co2_m, co2_q


def preprocess_brent_m(source_file: str, source_date_col: str, source_measure_col: str, var_name: str):
    # read data
    brent_m_raw = read_data(source_path=data_source_path, file_name=source_file)
    brent_m = brent_m_raw.copy()

    brent_m = brent_m.rename(columns={source_date_col: date_col, source_measure_col: var_name})

    brent_m[date_col] = pd.to_datetime(brent_m[date_col])
    brent_m[year_col] = brent_m[date_col].dt.year
    brent_m[month_col] = brent_m[date_col].dt.month

    # select years
    brent_m = select_country_year_measure(df=brent_m, year_col=year_col)

    # order
    brent_m = brent_m[[date_col, year_col, month_col, var_name]].reset_index(drop=True)

    # downsample to q
    brent_q = downsample_month_to_quarter(df_m=brent_m, var_name=var_name, agg='mean')

    # export to csv
    brent_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    brent_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return brent_m, brent_q


def preprocess_infl_m(source_file: str, source_country_col: str, var_name: str):
    # read data
    infl_m = read_data(source_path=data_source_path, file_name=source_file)

    # lowercase, replace country names
    infl_m[source_country_col] = infl_m[source_country_col].str.lower()
    infl_m = infl_m.replace({source_country_col: corr_country_names})

    infl_m = infl_m.melt(id_vars=[source_country_col],
                         value_vars=infl_m.drop([source_country_col], axis=1),
                         value_name=var_name)

    infl_m[date_col] = pd.to_datetime(infl_m['variable'])
    infl_m[year_col] = infl_m[date_col].dt.year
    infl_m[month_col] = infl_m[date_col].dt.month
    infl_m = infl_m.drop('variable', axis=1)

    # select years
    infl_m = select_country_year_measure(df=infl_m, year_col=year_col)

    # order
    infl_m = rename_order_scale(df=infl_m, source_country_col=source_country_col,
                                var_name=var_name, var_scale=1e-2, timeframe='m')

    # downsample monthly to quarterly
    infl_q = downsample_month_to_quarter(df_m=infl_m, var_name=var_name, agg='mean')

    # export to csv
    infl_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    infl_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return infl_m, infl_q


def preprocess_ind_prod_m(source_file: str, source_country_col: str, var_name: str):
    ind_prod_m = read_data(source_path=data_source_path, file_name=source_file)

    # lowercase, replace country names
    ind_prod_m[source_country_col] = ind_prod_m[source_country_col].str.lower()
    ind_prod_m = ind_prod_m.replace({source_country_col: corr_country_names})

    ind_prod_m = ind_prod_m.melt(id_vars=[source_country_col],
                                 value_vars=ind_prod_m.drop([source_country_col], axis=1),
                                 value_name=var_name)

    ind_prod_m[date_col] = pd.to_datetime(ind_prod_m['variable'])
    ind_prod_m[year_col] = ind_prod_m[date_col].dt.year
    ind_prod_m[month_col] = ind_prod_m[date_col].dt.month
    ind_prod_m = ind_prod_m.drop('variable', axis=1)

    ind_prod_m = select_country_year_measure(df=ind_prod_m, year_col=year_col, country_col=source_country_col)
    ind_prod_m = ind_prod_m.replace({':': np.nan})

    ind_prod_m = rename_order_scale(df=ind_prod_m, source_country_col=source_country_col,
                                    var_name=var_name, var_scale=1e-2, timeframe='m')
    ind_prod_m['ind_prod'] = interpolate_series(series=ind_prod_m['ind_prod'], method='median')

    ind_prod_q = downsample_month_to_quarter(df_m=ind_prod_m, var_name=var_name, agg='mean')

    # export to csv
    ind_prod_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    ind_prod_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return ind_prod_m, ind_prod_q


# Quarterly GDP data
def preprocess_WB_q(source_file: str, source_country_col: str, source_time_col: str,
                    source_measure_col: str, source_incl_measure: list, var_name: str, var_scale: float):
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
    df_q[var_name] = df_q['Value']

    # select countries and year
    df_q = select_country_year_measure(df=df_q, country_col=source_country_col, year_col=year_col,
                                       measure_col=source_measure_col, incl_measure=source_incl_measure)

    # rename, order and scale: output = [index, country, date, value]
    df_q = rename_order_scale(df=df_q, source_country_col=source_country_col, source_year_col=year_col,
                              timeframe='q', var_name=var_name, var_scale=var_scale)

    # upsample monthly to quarterly
    df_m = upsample_quarter_to_month(df_q=df_q, var_name=var_name)
    df_q[date_col] = pd.to_datetime(dict(year=df_q[year_col], month=df_q[quarter_col].apply(quarter_to_month), day=1))

    # export to csv
    df_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    df_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return df_m, df_q


def total_join(co2: object, pop: object, gdp: object, ind_prod: object,
               infl: object, brent: object, key_cols: list, timeframe: str):
    total = co2.copy()
    total = total.merge(gdp, how='left', on=key_cols)
    total = total.merge(ind_prod, how='left', on=key_cols)
    total = total.merge(infl, how='left', on=key_cols)
    total = total.merge(pop, how='left', on=key_cols)
    total = total.merge(brent, how='left', on=key_cols.remove(country_col))

    total[f'co2_cap'] = total[f'co2'] / total[f'pop']
    total[f'gdp_cap'] = total[f'gdp'] / total[f'pop']

    total = total.dropna(axis=0, how='any', subset=total.columns.drop(['infl', 'ind_prod']))
    total = total.fillna(fake_num).reset_index(drop=True)
    # total = total.reset_index(drop=True)
    total.to_csv(f'{get_data_path(timeframe=timeframe)}/total_{timeframe}.csv', header=True, index=False)

    return total


def make_stat(df: object, timeframe: str):
    i = 1
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

        for country in incl_countries:
            data_path_cor = get_data_path(timeframe=timeframe, country=country)
            fig_path_cor = get_fig_path(timeframe=timeframe, folder='data', country=country)

            df_country = df[df[country_col] == country]
            country_list += list(df_country[country_col])
            date_list += list(df_country[date_col])
            year_list += list(df_country[year_col])
            period_list += list(df_country[period_col])

            for series in vars:
                print(timeframe, stat, country, series)
                df_country_series = df_country.set_index(date_col)[series]
                var_name = f'{country}_{timeframe}_{series}_act'
                df_country_series.to_csv(f'{data_path_cor}{var_name}.csv')
                plot_series(i=i, series=df_country_series, timeframe=timeframe,
                            target_country=country, var_name=var_name)
                i += 1

                log, diff_level, diff_order = trans[series]

                # log the series if necessary
                if log:
                    df_country_series_log = np.log(df_country_series)
                else:
                    df_country_series_log = df_country_series
                var_name = f'{country}_{timeframe}_{series}_act_log'
                df_country_series_log.to_csv(f'{data_path_cor}{var_name}.csv')
                plot_series(i=i, series=df_country_series_log, timeframe=timeframe,
                            target_country=country, var_name=var_name)
                i += 1

                # difference the series
                j = 1
                df_country_series_diff = df_country_series_log.copy()
                if diff_level != 0:
                    while j <= diff_order:
                        df_country_series_diff = df_country_series_diff.diff(periods=diff_level)
                        var_name = f'{country}_{timeframe}_{series}_act_log_diff_{j}'
                        df_country_series_diff.to_csv(f'{data_path_cor}{var_name}.csv')
                        plot_series(i=i, series=df_country_series_diff, timeframe=timeframe,
                                    target_country=country, var_name=var_name)
                        i += 1
                        j += 1

                # if series is non-stationary input fake number -99999
                if stat == 'stat':
                    if stat_test(x=df_country_series_diff.dropna(), sign_level=sign_level) == 'stationary':
                        globals()[f"{series}_list"] += list(df_country_series_diff)
                    elif stat_test(x=df_country_series_diff.dropna(), sign_level=sign_level) == 'non_stationary':
                        globals()[f"{series}_list"] += [fake_num] * len(df_country_series_diff)

                elif stat == 'non_stat':
                    if sum(df_country_series_diff.dropna() == 0) == len(df_country_series_diff.dropna()):
                        globals()[f"{series}_list"] += [fake_num] * len(df_country_series_diff)
                    else:
                        globals()[f"{series}_list"] += list(df_country_series_diff)
                else:
                    raise ValueError('Define stat as being "stat" or "non_stat"')

        # put together in dataframe
        total_stat = pd.DataFrame(list(zip(country_list, date_list, year_list, period_list)),
                                  columns=[country_col, date_col, year_col, period_col])
        total_stat['co2'] = co2_list
        total_stat['gdp'] = gdp_list
        total_stat['ind_prod'] = ind_prod_list
        total_stat['infl'] = infl_list
        total_stat['pop'] = pop_list
        total_stat['brent'] = brent_list

        if 'co2_cap' in trans:
            total_stat['co2_cap'] = co2_cap_list
        if 'gdp_cap' in trans:
            total_stat['gdp_cap'] = gdp_cap_list

        total_stat = total_stat.dropna(axis=0, how='any').reset_index(drop=True)
        total_stat.to_csv(f'{get_data_path(timeframe=timeframe)}/total_{timeframe}_{stat}.csv', header=True, index=False)
        if show_results:
            print(f'Timeframe: {timeframe}; Stat: {stat}')
            print(total_stat)

    return total_stat


def preprocess():
    co2_m, co2_q = preprocess_co2_m(source_file='co2_m_2000_2021',
                                    source_country_col='Name',
                                    source_year_col='Year',
                                    var_name='co2'
                                    )

    infl_m, infl_q = preprocess_infl_m(source_file='infl_m_2000_2023',
                                       source_country_col='Country',
                                       var_name='infl'
                                       )

    ind_prod_m, ind_prod_q = preprocess_ind_prod_m(source_file='ind_prod_m_1953_2023',
                                                   source_country_col='Country',
                                                   var_name='ind_prod'
                                                   )

    gdp_m, gdp_q = preprocess_WB_q(source_file='gdp_q_1990_2022',
                                   source_country_col='Country',
                                   source_time_col='TIME',
                                   source_measure_col='MEASURE',
                                   source_incl_measure=['CPCARSA'],
                                   var_name='gdp',
                                   var_scale=1e6
                                   )

    pop_m, pop_q = preprocess_WB_q(source_file='pop_q_1995_2022',
                                   source_country_col='Country',
                                   source_time_col='TIME',
                                   source_measure_col='MEASURE',
                                   source_incl_measure=['PERSA'],
                                   var_name='pop',
                                   var_scale=1e3
                                   )

    brent_m, brent_q = preprocess_brent_m(source_file='brent_m_1990_2023',
                                          source_date_col='DATE',
                                          source_measure_col='BRENT',
                                          var_name='brent'
                                          )

    # total monthly
    timeframe = 'm'
    total_m = total_join(co2=co2_m, pop=pop_m, gdp=gdp_m, brent=brent_m, infl=infl_m, ind_prod=ind_prod_m,
                         key_cols=[country_col, date_col, year_col, month_col], timeframe=timeframe)
    make_stat(df=total_m, timeframe=timeframe)

    # total quarterly
    timeframe = 'q'
    total_q = total_join(co2=co2_q, pop=pop_q, gdp=gdp_q, brent=brent_q, infl=infl_q, ind_prod=ind_prod_q,
                         key_cols=[country_col, date_col, year_col, quarter_col], timeframe=timeframe)
    make_stat(df=total_q, timeframe=timeframe)


if __name__ == "__main__":
    preprocess()
