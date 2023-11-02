# import relevant packages
import os
import pandas as pd
import datetime as dt

from definitions import data_path, output_path, target_var, incl_countries, treatment_countries, \
    country_col, date_col, model_val, timeframe_val, folder_val, stat_val, donor_countries_all


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
            'co2': (True, 12)
            , 'gdp': (False, 0)
            , 'ind_prod': (False, 0)
            , 'infl': (False, 1)
            , 'unempl': (True, 1)
            , 'pop': (True, 1)
            , 'brent': (True, 1)
        }
    elif timeframe == 'q':
        trans = {
            'co2': (True, 4)
            , 'gdp': (False, 0)
            , 'ind_prod': (False, 0)
            , 'infl': (False, 1)
            , 'unempl': (True, 1)
            , 'pop': (True, 1)
            , 'brent': (True, 1)
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
                                              'ireland': dt.date(2010, 4, 1),
                                              'united_kingdom': dt.date(2013, 4, 1),
                                              'france': dt.date(2014, 4, 1),
                                              'portugal': dt.date(2015, 1, 1)
                                              }
    else:
        if treatment_country not in treatment_countries:
            return '2015-01-01'
        else:
            treatment_countries_impl_dates = {'switzerland': '2008-01-01',
                                              'ireland': '2010-04-01',
                                              'united_kingdom': '2013-04-01',
                                              'france': '2014-04-01',
                                              'portugal': '2015-01-01'
                                              }

    return treatment_countries_impl_dates[treatment_country]


# get corrections for implementation months
def get_months_cors(model: str, timeframe: str, treatment_country: str):
    if model == 'did':
        return 0
    else:
        months_cors = {'switzerland': 15,
                       'ireland': 0,
                       'united_kingdom': -3,
                       'france': -3,
                       'portugal': 15,
                       'other': 0
                       }
    if timeframe == 'm':
        return int(months_cors[treatment_country])
    elif timeframe == 'q':
        return int(months_cors[treatment_country] / 3)
    else:
        raise ValueError(f'Input valid timeframe argument: {timeframe_val}')


# function to get the donor countries given the prox argument and the treatment country
def get_donor_countries(model: str = None, prox: bool = None, treatment_country: str = None):
    if treatment_country not in treatment_countries:
        donor_countries = [i for i in donor_countries_all if i != str(treatment_country)]
        return donor_countries
    else:
        if prox and treatment_country is not None:
            if model == 'did':
                donor_countries_prox = {'switzerland': ['hungary', 'slovakia'],
                                        'ireland': ['romania', 'spain'],
                                        'united_kingdom': ['czech_republic', 'greece', 'romania'],
                                        'france': ['belgium'],
                                        'portugal': ['czech_republic', 'greece', 'hungary', 'italy', 'romania', 'spain']
                                        }
            else:
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
             'lasso': 'darkorange',  # orange
             'sc': 'hotpink',  # green
             'ols': 'yellowgreen',
             'rf': 'yellowgreen'
             }
    if type in color.keys():
        return color[type]
    else:
        return 'black'


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


# get formal title for saving the plots
def get_formal_title(var_name: str):
    if 'act_pred_log_diff_check' in var_name:
        return 'log-differenced CHECK'

    elif 'act_pred_log_diff' in var_name:
        return 'log-differenced'

    elif 'act_pred_log' in var_name:
        return 'log'

    elif 'act_pred' in var_name:
        return ''
    else:
        return var_name


# get formal var name for saving the plots
def get_formal_var_name(var_name: str = None):
    var_name_formal = {'co2': 'CO2 Emissions',
                       'gdp': 'GDP',
                       'ind_prod': 'Industrial Production',
                       'pop': 'Population',
                       'infl': 'Inflation',
                       'unempl': 'Unemployment',
                       'brent': 'Brent Oil'
                       }
    if var_name is not None:
        return var_name_formal[var_name]
    else:
        return var_name_formal.keys()


# get formal var name for saving the plots
def get_formal_country_name(country: str = None):
    country_name_formal = {'switzerland': 'Switzerland',
                           'ireland': 'Ireland',
                           'united_kingdom': 'United Kingdom',
                           'france': 'France',
                           'portugal': 'Portugal',
                           'other': 'Other'
                           }
    if country is not None and country in treatment_countries:
        return country_name_formal[country]
    else:
        return country_name_formal.keys()