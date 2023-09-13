################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from definitions import target_var, data_path, incl_countries, incl_years, donor_countries, target_countries, \
    country_col, year_col, month_col, quarter_col, date_col, model_val, timeframe_val, tables_path_res, save_results


def get_impl_date(target_country: str = None):
    target_countries_impl_dates = {'switzerland': '2008-01-01',
                                   'ireland': '2010-01-01',
                                   'united kingdom': '2013-01-01',
                                   'france': '2014-01-01',
                                   'portugal': '2015-01-01'
                                   }
    if target_country is None:
        return target_countries_impl_dates
    else:
        return target_countries_impl_dates[target_country]


def get_timescale(timeframe: str = None):
    timeframe_scale = {'q': 4,
                       'm': 12
                       }
    if timeframe is None:
        return timeframe_scale
    else:
        return timeframe_scale[timeframe]


def get_timeframe_col(timeframe: str = None):
    timeframe_col = {'q': 'quarter',
                     'm': 'month'
                     }
    if timeframe is None:
        return timeframe_col
    else:
        return timeframe_col[timeframe]


def get_trans(timeframe: str = None):
    # trans: 'var': (log, diff_level)
    if timeframe == 'm':
        trans = {
            'co2': (True, 12, 2)
            , 'gdp': (True, 12, 2)
            , 'pop': (True, 12, 2)
            # , 'co2_cap': (True, 12, 2)
            # , 'gdp_cap': (True, 12, 2)
        }
    elif timeframe == 'q':
        trans = {
            'co2': (True, 4, 2)
            , 'gdp': (True, 4, 2)
            , 'pop': (True, 4, 2)
            # , 'co2_cap': (True, 12, 2)
            # , 'gdp_cap': (True, 12, 2)
        }
    else:
        trans = ['co2', 'gdp', 'pop']

    return trans


def month_name_to_num(month_name: str = None):
    month_num = {'Jan': 1,
                 'Feb': 2,
                 'Mar': 3,
                 'Apr': 4,
                 'May': 5,
                 'Jun': 6,
                 'Jul': 7,
                 'Aug': 8,
                 'Sep': 9,
                 'Oct': 10,
                 'Nov': 11,
                 'Dec': 12}
    if month_name is None:
        return month_num
    else:
        return month_num[month_name]


def quarter_to_month(quarter: int):
    month = {1: 1,
             2: 4,
             3: 7,
             4: 10}
    if quarter is None:
        return month
    else:
        return month[quarter]


def month_to_quarter(month: int):
    quarter = {1: 1,
               2: 1,
               3: 1,
               4: 2,
               5: 2,
               6: 2,
               7: 3,
               8: 3,
               9: 3,
               10: 4,
               11: 4,
               12: 4}
    if month is None:
        return quarter
    else:
        return quarter[month]


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def first_value(target_country: str, timeframe: str):
    df = read_data(source_path=data_path, file_name=f'total_{timeframe}')
    orig_value = df[df['country'] == target_country].set_index('date')[target_var.replace('_stat', '')].iloc[0]
    return orig_value


def read_data(source_path: str, file_name: str):
    df = pd.read_csv(f'{source_path}{file_name}.csv', delimiter=',', header=0, encoding='latin-1')
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    return df


def validate_input(model: str, timeframe: str, target_country: str):
    if model not in model_val:
        raise ValueError(f'Input a valid model argument: {model_val}')

    elif timeframe not in timeframe_val:
        raise ValueError(f'Input a valid timeframe argument: {timeframe_val}')

    elif target_country not in target_countries:
        raise ValueError(f'Input a valid target_country argument: {target_countries}')

    else:
        return True


def select_country_year_measure(df: object, country_col: str = None, year_col: str = None,
                                measure_col: str = None, incl_measure: list = None):
    if country_col is not None:
        df = df[df[country_col].isin(incl_countries)]

    if year_col is not None:
        df = df[df[year_col].isin(incl_years)]

    if measure_col is not None:
        df = df[df[measure_col].isin(incl_measure)]

    return df


def rename_order_scale(df: object, source_country_col: str, source_year_col: str, timeframe: str,
                       var_name: str, var_scale: float):
    df = df.rename(columns={source_country_col: country_col, source_year_col: year_col})

    period_col = get_timeframe_col(timeframe)
    df = df[[country_col, date_col, year_col, period_col, var_name]]
    df = df.sort_values(by=[country_col, year_col, period_col])

    df[var_name] = df[var_name].astype(float) * var_scale
    df = df.reset_index(drop=True)

    return df


def downsample_month_to_quarter(df_m: object, var_name: str):
    df_q = pd.DataFrame({var_name: [],
                         country_col: []}
                        )

    for country in df_m[country_col].unique():
        df_country = df_m.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country = df_country.set_index(date_col)[var_name]
        df_country = df_country.resample('Q', convention='start').sum().to_frame()
        df_country[country_col] = [country] * len(df_country)

        df_q = pd.concat([df_q, df_country], axis=0)

    df_q = df_q.reset_index()
    df_q = df_q.rename(columns={'index': date_col})

    df_q[date_col] = [df_m[date_col][3 * i].to_pydatetime() for i in range(0, int(len(df_m) / 3))]
    df_q[year_col] = df_q[date_col].dt.year
    df_q[quarter_col] = df_q[date_col].dt.quarter

    df_q = df_q[[country_col, date_col, year_col, quarter_col, var_name]]
    df_q = df_q.sort_values(by=[country_col, year_col, quarter_col])

    return df_q


def upsample_quarter_to_month(df_q: object, var_name: str):
    df_m = pd.DataFrame({var_name: [],
                         country_col: []}
                        )

    for country in df_q[country_col].unique():
        df_country = df_q.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country = df_country.set_index(date_col)[var_name]
        df_country = df_country.resample('M', convention='start').interpolate().to_frame()
        df_country[country_col] = [country] * len(df_country)

        df_m = pd.concat([df_m, df_country], axis=0)

    df_m = df_m.reset_index()
    df_m = df_m.rename(columns={'index': date_col})
    df_m[date_col] = pd.to_datetime(df_m[date_col].astype(str))

    df_m[year_col] = df_m[date_col].dt.year
    df_m[month_col] = df_m[date_col].dt.month

    df_m = df_m[[country_col, date_col, year_col, month_col, var_name]]
    df_m = df_m.sort_values(by=[country_col, year_col, month_col])

    return df_m


def interpolate_series(series: object, method='linear'):
    num_na = sum(series.isna())
    series_no_na = series[~series.isna()]

    if method == 'linear':
        factor = np.mean(np.array(series_no_na.iloc[1:11]) / np.array(series_no_na.iloc[0:10]))

        lst = [series_no_na.iloc[0]]
        for i in range(num_na):
            lst.append(lst[i] * (1 / factor))
        lst = lst[1:][::-1]
        new_series = np.array(lst + list(series_no_na))
        return new_series

    elif method == 'median':
        factor = np.median(series_no_na)
        new_series = series.fillna(factor)
        return new_series

    else:
        raise ValueError('Please specify the interpolation method (median / linear)')


def arco_pivot(df: object, target_country: str):
    target = df[df[country_col] == target_country].set_index(date_col)[target_var]

    donors = df.copy()
    donors = donors[donors[country_col].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index=date_col, columns=[country_col], values=get_trans())
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)
    donors = donors.dropna(axis=0)

    if save_results:
        target.to_csv(f'{tables_path_res}{target_country}/{target_country}_target.csv')
        donors.to_csv(f'{tables_path_res}{target_country}/{target_country}_donors.csv')

    return target, donors


def sc_pivot(df: object, target_country: str):

    df = df[df[country_col].isin(donor_countries + [target_country])]
    df_pivot = df.copy()
    df_pivot = df_pivot.pivot(index=country_col, columns=date_col, values=target_var)
    df_pivot = df_pivot.dropna(axis=1, how='any')

    pre_treat = df_pivot.iloc[:, df_pivot.columns <= get_impl_date(target_country)].values
    post_treat = df_pivot.iloc[:, df_pivot.columns > get_impl_date(target_country)].values
    treat_unit = [idx for idx, val in enumerate(df_pivot.index.values) if val == target_country]

    series = [df_pivot, pre_treat, post_treat, treat_unit]
    for i in range(0, len(series)):
        pd.DataFrame(series[i]).to_csv(f'series_{i}.csv')

    return df_pivot, pre_treat, post_treat, treat_unit


def did_pivot():
    pass

