################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd
import datetime as dt

from definitions import target_var, data_path, incl_countries, incl_years, treatment_countries, \
    country_col, year_col, month_col, quarter_col, date_col, model_val, timeframe_val, \
    output_path, agg_val, interpolation_val, folder_val, stat_val, donor_countries_all


# get data path given the timeframe and the target country
def get_data_path(timeframe: str,  country: str = None):
    if country is not None:
        if country not in incl_countries:
            raise ValueError(f'Input a valid country argument: {incl_countries}')

    if country is not None:
        path_cor = f'{data_path}/{timeframe}/{country}'
    else:
        path_cor = f'{data_path}/{timeframe}'

    if not os.path.exists(path_cor):
        os.makedirs(path_cor)
    return path_cor


# get the figure path given the timeframe and the target country
def get_fig_path(timeframe: str, folder: str, country: str = None, model: str = None):
    if timeframe not in timeframe_val:
        raise ValueError(f'Input a valid timeframe argument: {timeframe_val}')
    if folder not in folder_val:
        raise ValueError(f'Input a valid folder argument: {folder_val}')
    if country is not None:
        if country not in incl_countries + ['EDA']:
            raise ValueError(f'Input a valid country argument: {incl_countries}')

    if country is not None and model is not None:
        path_cor = f'{output_path}/{timeframe}/figures/{folder}/{country}/{model}'
    elif country is not None:
        path_cor = f'{output_path}/{timeframe}/figures/{folder}/{country}'
    else:
        path_cor = f'{output_path}/{timeframe}/figures/{folder}'

    if not os.path.exists(path_cor):
        os.makedirs(path_cor)
    return path_cor


# get the table path given the timeframe and the target country
def get_table_path(timeframe: str, folder: str, country: str = None, model: str = None):
    if timeframe not in timeframe_val:
        raise ValueError(f'Input a valid timeframe argument: {timeframe_val}')
    if folder not in folder_val:
        raise ValueError(f'Input a valid folder argument: {folder_val}')
    if country is not None:
        if country not in incl_countries + ['EDA']:
            raise ValueError(f'Input a valid country argument: {incl_countries}')

    if country is not None and model is not None:
        path_cor = f'{output_path}/{timeframe}/tables/{folder}/{country}/{model}'
    elif country is not None:
        path_cor = f'{output_path}/{timeframe}/tables/{folder}/{country}'
    else:
        path_cor = f'{output_path}/{timeframe}/tables/{folder}'

    if not os.path.exists(path_cor):
        os.makedirs(path_cor)
    return path_cor


# get required transformation for every variable for a given timeframe
def get_trans(timeframe: str = None):
    # trans: 'var': (log, diff_level)
    if timeframe == 'm':
        trans = {
            'co2': (True, 12, 1)
            , 'gdp': (False, 0, 0)
            , 'ind_prod': (False, 0, 0)
            , 'infl': (False, 1, 1)
            , 'unempl': (False, 1, 1)
            , 'pop': (True, 1, 1)
            , 'brent': (True, 1, 1)
        }
    elif timeframe == 'q':
        trans = {
            'co2': (True, 4, 1)
            , 'gdp': (False, 0, 0)
            , 'ind_prod': (False, 0, 0)
            , 'infl': (False, 1, 1)
            , 'unempl': (False, 1, 1)
            , 'pop': (True, 1, 1)
            , 'brent': (True, 1, 1)
        }
    else:
        trans = ['co2'
                 , 'gdp'
                 , 'ind_prod'
                 , 'infl'
                 , 'unempl'
                 , 'pop'
                 , 'brent'
                 ]

    return trans


# get the implementation date of the carbon tax for the treatment countries, in specific data format
def get_impl_date(treatment_country: str = None, input: str = None):
    if input == 'dt':
        if treatment_country not in treatment_countries:
            return dt.date(2015, 1, 1)
        else:
            treatment_countries_impl_dates = {'switzerland': dt.date(2008, 1, 1),
                                              'ireland': dt.date(2010, 5, 1),
                                              'united_kingdom': dt.date(2013, 4, 1),
                                              'france': dt.date(2014, 4, 1),
                                              'portugal': dt.date(2015, 1, 1)
                                              }
    else:
        if treatment_country not in treatment_countries:
            return '2015-01-01'
        else:
            treatment_countries_impl_dates = {'switzerland': '2008-01-01',
                                              'ireland': '2010-05-01',
                                              'united_kingdom': '2013-04-01',
                                              'france': '2014-04-01',
                                              'portugal': '2015-01-01'
                                              }

    return treatment_countries_impl_dates[treatment_country]


# get corrections for implementation months
def get_months_cors(timeframe, treatment_country):
    if treatment_country not in treatment_countries:
        return 0
    else:
        if timeframe == 'm':
            months_cors = {'switzerland': 15,
                           'ireland': 15,
                           'united_kingdom': 0,
                           'france': -3,
                           'portugal': 15,
                           'other': 0
                           }
        elif timeframe == 'q':
            months_cors = {'switzerland': 5,
                           'ireland': 5,
                           'united_kingdom': 0,
                           'france': -1,
                           'portugal': 5,
                           'other': 0
                           }
        else:
            raise ValueError(f'Input valid timeframe argument: {timeframe_val}')

        return months_cors[treatment_country]


# function to get the donor countries given the prox argument and the treatment country
def get_donor_countries(prox: bool = None, treatment_country: str = None):
    if treatment_country not in treatment_countries:
        donor_countries = [i for i in donor_countries_all if i != str(treatment_country)]
        return donor_countries
    else:
        if prox and treatment_country is not None:
            donor_countries_prox = {'switzerland': ['austria', 'germany', 'italy'],
                                    'ireland': donor_countries_all + ['united_kingdom'],
                                    'united_kingdom': ['netherlands', 'belgium', 'spain'],
                                    'france': ['belgium', 'germany', 'italy', 'netherlands', 'spain'],
                                    'portugal': ['spain']
                                    }
            return donor_countries_prox[treatment_country]
        else:
            return donor_countries_all


# define static colors for plotting the series from different models
def get_model_color(type: str):
    color = {'act': '#1f77b4',
             'error': '#1f77b4',
             'impl': 'black',
             'arco': 'darkorange',  # orange
             'sc': 'hotpink',  # green
             }
    return color[type]


# get formal title for saving the plots
def get_formal_title(var_name: str):
    if 'act_pred_log_diff_check' in var_name:
        return 'Log-differenced'

    elif 'act_pred_log_diff' in var_name:
        return 'Log-differenced'

    elif 'act_pred_log' in var_name:
        return 'Log'

    elif 'act_pred' in var_name:
        return ''

    else:
        return var_name


# get timescale for transforming quarter/month into year
def get_timescale(timeframe: str = None):
    timeframe_scale = {'q': 4,
                       'm': 12
                       }
    if timeframe is None:
        return timeframe_scale
    else:
        return timeframe_scale[timeframe]


# get the timeframe column given the timeframe
def get_timeframe_col(timeframe: str = None):
    timeframe_col = {'q': 'quarter',
                     'm': 'month'
                     }
    if timeframe is None:
        return timeframe_col
    else:
        return timeframe_col[timeframe]


# function to get the month number from the month name
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


# function to adapt the quarter to month period
def quarter_to_month(quarter: int):
    month = {1: 1,
             2: 4,
             3: 7,
             4: 10}
    if quarter is None:
        return month
    else:
        return month[quarter]


# function to adapt the month to quarter period
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


# flatten sublists in list to one long list
def flatten(lst):
    return [item for sublist in lst for item in sublist]


# get the first value of a series, for transforming back from log-differencing
def first_value(treatment_country: str, timeframe: str):
    df = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}')
    _, diff_level, diff_order = get_trans(timeframe=timeframe)[target_var]
    i = diff_level * diff_order
    orig_value = df[df[country_col] == treatment_country].set_index(date_col)[target_var].iloc[i]
    return orig_value


# function to easily read the data given the source path and desired file.
def read_data(source_path: str, file_name: str):
    df = pd.read_csv(f'{source_path}/{file_name}.csv', delimiter=',', header=0, encoding='latin-1')
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    return df


# function to validate the input of the main script
def validate_input(model: str = None, stat: str = None, timeframe: str = None, treatment_country: str = None):
    if model is not None and model not in model_val:
        raise ValueError(f'Input a valid model argument: {model_val}')

    if stat is not None and stat not in stat_val:
        raise ValueError(f'Input a valid model argument: {stat_val}')

    if timeframe is not None and timeframe not in timeframe_val:
        raise ValueError(f'Input a valid timeframe argument: {timeframe_val}')

    if treatment_country is not None and treatment_country not in incl_countries:
        raise ValueError(f'Input a valid treatment_country argument: {incl_countries}')

    else:
        return True


# function to select the needed countries, years, measures from the full dataframe
def select_country_year_measure(df: object, country_col: str = None, year_col: str = None,
                                measure_col: str = None, incl_measure: list = None):
    if country_col is not None:
        df = df[df[country_col].isin(incl_countries)]

    if year_col is not None:
        df = df[df[year_col].isin(incl_years)]

    if measure_col is not None:
        df = df[df[measure_col].isin(incl_measure)]

    return df


# function to reoder the dataframe, rename the column names and scale the variables to absolute units.
def rename_order_scale(df: object, source_country_col: str = None, source_year_col: str = None, source_date_col: str = None,
                       timeframe: str = None, var_name: str = None, var_scale: float = None):
    if source_country_col is not None:
        df = df.rename(columns={source_country_col: country_col})
    if source_year_col is not None:
        df = df.rename(columns={source_year_col: year_col})
    if source_date_col is not None:
        df = df.rename(columns={source_date_col: date_col})

    period_col = get_timeframe_col(timeframe)
    if country_col in df.columns:
        df = df[[country_col, date_col, year_col, period_col, var_name]]
        df = df.sort_values(by=[country_col, year_col, period_col])
    else:
        df = df[[date_col, year_col, quarter_col, var_name]]
        df = df.sort_values(by=[year_col, quarter_col])

    df[var_name] = df[var_name].astype(float) * var_scale
    df = df.reset_index(drop=True)

    return df


# function to resample monthly data to quarterly data via aggregation argument
def downsample_month_to_quarter(df_m: object, var_name: str, agg: str):
    if country_col in df_m.columns:
        df_q = pd.DataFrame({var_name: [],
                             country_col: []}
                            )

        for country in df_m[country_col].unique():
            df_country = df_m.copy()
            df_country = df_country[df_country[country_col] == country]
            df_country = df_country.set_index(date_col)[var_name]
            if agg == 'sum':
                df_country = df_country.resample('Q', convention='start').sum().to_frame()
            elif agg == 'mean':
                df_country = df_country.resample('Q', convention='start').mean().to_frame()
            else:
                raise ValueError(f'Input a valid agg argument: {agg_val}')
            df_country[country_col] = [country] * len(df_country)

            df_q = pd.concat([df_q, df_country], axis=0)

    else:
        df_q = df_m.copy()
        df_q = df_q.set_index(date_col)[var_name]
        if agg == 'sum':
            df_q = df_q.resample('Q', convention='start').sum().to_frame()
        elif agg == 'mean':
            df_q = df_q.resample('Q', convention='start').mean().to_frame()
        else:
            raise ValueError(f'Input a valid agg argument: {agg_val}')

    df_q = df_q.reset_index()
    df_q = df_q.rename(columns={'index': date_col})

    df_q[date_col] = [df_m[date_col][3 * i].to_pydatetime() for i in range(0, int(len(df_m) / 3))]
    df_q[year_col] = df_q[date_col].dt.year
    df_q[quarter_col] = df_q[date_col].dt.quarter

    if country_col in df_m.columns:
        df_q = df_q[[country_col, date_col, year_col, quarter_col, var_name]]
        df_q = df_q.sort_values(by=[country_col, year_col, quarter_col])
    else:
        df_q = df_q[[date_col, year_col, quarter_col, var_name]]
        df_q = df_q.sort_values(by=[year_col, quarter_col])

    return df_q


# function to resample quarterly data to monthly data via interpolation
def upsample_quarter_to_month(df_q: object, var_name: str):
    df_m = pd.DataFrame({var_name: [],
                         country_col: []}
                        )

    for country in df_q[country_col].unique():
        df_country = df_q.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country[date_col] = df_country[date_col].dt.to_period('M')
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


# function on how to interpolate series given method argument
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
        raise ValueError(f'Input a valid method argument: {interpolation_val}')