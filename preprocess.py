################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from definitions import data_source_path, corr_country_names, sign_level, fake_num, save_figs, show_output, \
    country_col, year_col, quarter_col, month_col, date_col, incl_countries
from util_general import read_data, month_name_to_num, quarter_to_month, get_timeframe_col, get_trans, \
    get_data_path, validate_input
from util_preprocess import select_country_year_measure, rename_order_scale, downsample_month_to_quarter, \
    upsample_quarter_to_month, interpolate_series, print_preprocess

from statistical_tests import stat_test
from plot_functions import plot_series


# preprocess monthly CO2 data
def preprocess_co2_m(source_file: str, source_country_col: str, source_year_col: str, var_name: str,
                     var_scale: str, downsample_method: str):
    print_preprocess(var_name=var_name)
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
                               var_name=var_name, var_scale=var_scale, timeframe='m')
    # downsample monthly to quarterly
    co2_q = downsample_month_to_quarter(df_m=co2_m, var_name=var_name, agg=downsample_method)

    # export to csv
    co2_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    co2_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return co2_m, co2_q


# preprocess brent oil data
def preprocess_brent_m(source_file: str, source_date_col: str, source_measure_col: str, var_name: str,
                       downsample_method: str):
    print_preprocess(var_name=var_name)
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
    brent_q = downsample_month_to_quarter(df_m=brent_m, var_name=var_name, agg=downsample_method)

    # export to csv
    brent_m.to_csv(f'{get_data_path(timeframe="m")}/{var_name}_m.csv')
    brent_q.to_csv(f'{get_data_path(timeframe="q")}/{var_name}_q.csv')

    return brent_m, brent_q


# preprocess world bank data
def preprocess_WB(source_file: str, source_country_col: str, source_time_col: str, source_measure_col: str,
                  source_incl_measure: list, source_timeframe: str, var_name: str, var_scale: float,
                  downsample_method: str = None):
    print_preprocess(var_name=var_name)

    # validate input
    validate_input(timeframe=source_timeframe)
    alt_timeframe = ['m' if source_timeframe == 'q' else 'q'][0]

    # read data
    df_raw = read_data(source_path=data_source_path, file_name=source_file)
    df = df_raw.copy()

    # lowercase, replace country names
    df[source_country_col] = df[source_country_col].str.lower()
    df = df.replace({source_country_col: corr_country_names})

    # transform
    df[year_col] = df[source_time_col].str[:4].astype(int)
    df[quarter_col] = df[source_time_col].str[6:].astype(int)
    df[month_col] = df.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    df[date_col] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=1))
    df[var_name] = df['Value']

    # select countries and year
    df = select_country_year_measure(df=df, country_col=source_country_col, year_col=year_col,
                                     measure_col=source_measure_col, incl_measure=source_incl_measure)

    # rename, order and scale: output = [index, country, date, value]
    df = rename_order_scale(df=df, source_country_col=source_country_col, source_year_col=year_col,
                            timeframe=source_timeframe, var_name=var_name, var_scale=var_scale)

    if source_timeframe == 'm':
        df_alt = downsample_month_to_quarter(df_m=df, var_name=var_name, agg=downsample_method)
    else:
        df_alt = upsample_quarter_to_month(df_q=df, var_name=var_name)

    df[date_col] = pd.to_datetime(dict(year=df[year_col], month=df[quarter_col].apply(quarter_to_month), day=1))

    # export to csv
    df.to_csv(f'{get_data_path(timeframe=source_timeframe)}/{var_name}_{source_timeframe}.csv')
    df_alt.to_csv(f'{get_data_path(timeframe=alt_timeframe)}/{var_name}_{alt_timeframe}.csv')

    return df, df_alt


# preprocess eurostat data
def preprocess_EUstat(source_file: str, source_country_col: str, var_name: str, var_scale: float,
                      source_timeframe: str, interpolate_method: str, downsample_method: str = None):
    print_preprocess(var_name=var_name)
    # validate input
    validate_input(timeframe=source_timeframe)
    alt_timeframe = ['m' if source_timeframe == 'q' else 'q'][0]

    # read data
    df_raw = read_data(source_path=data_source_path, file_name=source_file)
    df = df_raw.copy()

    # lowercase, replace country names
    df[source_country_col] = df[source_country_col].str.lower()
    df = df.replace({source_country_col: corr_country_names})

    df = df.melt(id_vars=[source_country_col],
                 value_vars=df.drop([source_country_col], axis=1),
                 value_name=var_name)

    df[date_col] = pd.to_datetime(df['variable'])
    df[year_col] = df[date_col].dt.year
    df[quarter_col] = df[date_col].dt.quarter
    df[month_col] = df[date_col].dt.month
    df = df.drop('variable', axis=1)

    df = select_country_year_measure(df=df, year_col=year_col, country_col=source_country_col)
    df = df.replace({':': np.nan})

    df = rename_order_scale(df=df, source_country_col=source_country_col,
                            var_name=var_name, var_scale=var_scale, timeframe=source_timeframe)
    df[var_name] = interpolate_series(series=df[var_name], method=interpolate_method)

    if source_timeframe == 'm':
        df_alt = downsample_month_to_quarter(df_m=df, var_name=var_name, agg=downsample_method)
    else:
        df_alt = upsample_quarter_to_month(df_q=df, var_name=var_name)

    df.to_csv(f'{get_data_path(timeframe=source_timeframe)}/{var_name}_{source_timeframe}.csv')
    df_alt.to_csv(f'{get_data_path(timeframe=alt_timeframe)}/{var_name}_{alt_timeframe}.csv')

    return df, df_alt


# join all preprocessed series together into one big dataframe
def total_join(key_cols: list, timeframe: str,
               co2: object = None, brent: object = None,
               infl: object = None, unempl: object = None, ind_prod: object = None,
               pop: object = None, gdp: object = None):
    total = co2.copy()
    if gdp is not None:
        total = total.merge(gdp, how='outer', on=key_cols)
    if ind_prod is not None:
        total = total.merge(ind_prod, how='outer', on=key_cols)
    if infl is not None:
        total = total.merge(infl, how='outer', on=key_cols)
    if unempl is not None:
        total = total.merge(unempl, how='outer', on=key_cols)
    if pop is not None:
        total = total.merge(pop, how='outer', on=key_cols)
    if brent is not None:
        total = total.merge(brent, how='outer', on=key_cols.remove(country_col))

    total = total.dropna(axis=0, how='any', subset=total.columns.drop(['ind_prod', 'infl']))
    total = total.fillna(fake_num).reset_index(drop=True)
    total = total.sort_values([country_col, date_col])
    total.to_csv(f'{get_data_path(timeframe=timeframe)}/total_{timeframe}.csv', header=True, index=False)

    return total


# Make all country x variable series stationary based on (log, difference) input
def make_stat(df: object, timeframe: str):
    i = 1
    for stat in ['stat', 'non_stat']:

        country_list = []
        date_list = []
        year_list = []
        period_list = []
        period_col = get_timeframe_col(timeframe=timeframe)
        trans = get_trans(timeframe=timeframe)

        vars = trans.keys()
        for series in vars:
            globals()[f"{series}_list"] = []

        for country in incl_countries:
            data_path_country = get_data_path(timeframe=timeframe, country=country)

            df_country = df[df[country_col] == country]
            country_list += list(df_country[country_col])
            date_list += list(df_country[date_col])
            year_list += list(df_country[year_col])
            period_list += list(df_country[period_col])

            for series in vars:
                # print(timeframe, stat, country, series)
                df_country_series = df_country.set_index(date_col)[series]
                var_name = f'{country}_{timeframe}_{series}_act'
                df_country_series.to_csv(f'{data_path_country}/{var_name}.csv')
                if save_figs and stat == 'stat':
                    plot_series(i=i, series=df_country_series, timeframe=timeframe,
                                treatment_country=country, var_name=var_name)
                i += 1

                log, diff_level = trans[series]

                # log the series if necessary
                if log:
                    df_country_series_log = np.log(df_country_series)
                else:
                    df_country_series_log = df_country_series
                var_name = f'{country}_{timeframe}_{series}_act_log'
                df_country_series_log.to_csv(f'{data_path_country}/{var_name}.csv')
                if save_figs and stat == 'stat':
                    plot_series(i=i, series=df_country_series_log, timeframe=timeframe,
                                treatment_country=country, var_name=var_name)
                i += 1

                # difference the series
                if diff_level != 0:
                    df_country_series_log_diff = df_country_series_log.diff(periods=diff_level)
                else:
                    df_country_series_log_diff = df_country_series_log
                var_name = f'{country}_{timeframe}_{series}_act_log_diff'
                df_country_series_log_diff.to_csv(f'{data_path_country}/{var_name}.csv')
                if save_figs and stat == 'stat':
                    plot_series(i=i, series=df_country_series_log_diff, timeframe=timeframe,
                                treatment_country=country, var_name=var_name)
                i += 1

                # if series is non-stationary input fake number -99999
                if stat == 'stat':
                    if stat_test(x=df_country_series_log_diff.dropna(), sign_level=sign_level) == 'stationary':
                        globals()[f"{series}_list"] += list(df_country_series_log_diff)
                    elif stat_test(x=df_country_series_log_diff.dropna(), sign_level=sign_level) == 'non_stationary':
                        globals()[f"{series}_list"] += [fake_num] * len(df_country_series_log_diff)

                elif stat == 'non_stat':
                    if sum(df_country_series_log_diff.dropna() == 0) == len(df_country_series_log_diff.dropna()):
                        globals()[f"{series}_list"] += [fake_num] * len(df_country_series_log_diff)
                    else:
                        globals()[f"{series}_list"] += list(df_country_series_log_diff)
                else:
                    raise ValueError('Define stat as being "stat" or "non_stat"')

        # put together in dataframe
        total_stat = pd.DataFrame(list(zip(country_list, date_list, year_list, period_list)),
                                  columns=[country_col, date_col, year_col, period_col])
        if 'co2' in trans:
            total_stat['co2'] = co2_list
        if 'gdp' in trans:
            total_stat['gdp'] = gdp_list
        if 'ind_prod' in trans:
            total_stat['ind_prod'] = ind_prod_list
        if 'infl' in trans:
            total_stat['infl'] = infl_list
        if 'unempl' in trans:
            total_stat['unempl'] = unempl_list
        if 'pop' in trans:
            total_stat['pop'] = pop_list
        if 'brent' in trans:
            total_stat['brent'] = brent_list

        total_stat = total_stat.dropna(axis=0, how='any')
        total_stat = total_stat.reset_index(drop=True)
        total_stat.to_csv(f'{get_data_path(timeframe=timeframe)}/total_{timeframe}_{stat}.csv', header=True, index=False)
        if show_output:
            print(f'Timeframe: {timeframe}; Stat: {stat}')
            print(total_stat)


# main function to preprocess all series
def preprocess():
    print('Starting preprocessing...')
    co2_m, co2_q = preprocess_co2_m(source_file='edgar_co2_m_2000_2021',
                                    source_country_col='Name',
                                    source_year_col='Year',
                                    var_name='co2',
                                    var_scale=1e6,
                                    downsample_method='sum'
                                    )

    brent_m, brent_q = preprocess_brent_m(source_file='fred_brent_m_1990_2023',
                                          source_date_col='DATE',
                                          source_measure_col='BRENT',
                                          var_name='brent',
                                          downsample_method='sum'
                                          )

    ind_prod_m, ind_prod_q = preprocess_EUstat(source_file='eurostat_ind_prod_m_1953_2023',
                                               source_country_col='Country',
                                               var_name='ind_prod',
                                               var_scale=1e-2,
                                               source_timeframe='m',
                                               interpolate_method='median',
                                               downsample_method='mean'
                                               )

    infl_m, infl_q = preprocess_EUstat(source_file='eurostat_infl_m_1963_2023',
                                       source_country_col='Country',
                                       var_name='infl',
                                       var_scale=1e-2,
                                       source_timeframe='m',
                                       interpolate_method='median',
                                       downsample_method='mean'
                                       )

    unempl_m, unempl_q = preprocess_EUstat(source_file='eurostat_unempl_m_1980_2023',
                                           source_country_col='Country',
                                           var_name='unempl',
                                           var_scale=1e-2,
                                           source_timeframe='m',
                                           interpolate_method='median',
                                           downsample_method='mean'
                                           )

    gdp_q, gdp_m = preprocess_WB(source_file='wb_gdp_q_1996_2022',
                                 source_country_col='Country',
                                 source_time_col='TIME',
                                 source_measure_col='MEASURE',
                                 source_incl_measure=['GPSA'],  # GPSA=growth rate, CPCARSA = absolute
                                 source_timeframe='q',
                                 var_name='gdp',
                                 var_scale=1e-2  # 1e-2, 1e6
                                 )

    pop_q, pop_m = preprocess_WB(source_file='wb_pop_q_1995_2022',
                                 source_country_col='Country',
                                 source_time_col='TIME',
                                 source_measure_col='MEASURE',
                                 source_incl_measure=['PERSA'],
                                 source_timeframe='q',
                                 var_name='pop',
                                 var_scale=1e3
                                 )

    var_name = 'total'

    # total monthly
    timeframe = 'm'
    print_preprocess(var_name=var_name, timeframe=timeframe)
    total_m = total_join(key_cols=[country_col, date_col, year_col, month_col], timeframe=timeframe, co2=co2_m,
                         brent=brent_m, infl=infl_m, unempl=unempl_m, ind_prod=ind_prod_m, pop=pop_m, gdp=gdp_m)
    make_stat(df=total_m, timeframe=timeframe)

    # total quarterly
    timeframe = 'q'
    print_preprocess(var_name=var_name, timeframe=timeframe)
    total_q = total_join(key_cols=[country_col, date_col, year_col, quarter_col], timeframe=timeframe, co2=co2_q,
                         brent=brent_q, infl=infl_q, unempl=unempl_q, ind_prod=ind_prod_q, pop=pop_q, gdp=gdp_q)
    make_stat(df=total_q, timeframe=timeframe)

    print('Done!')


if __name__ == "__main__":
    preprocess()
