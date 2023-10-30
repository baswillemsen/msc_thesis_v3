################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from definitions import show_output, incl_countries, incl_years, \
    country_col, year_col, month_col, quarter_col, date_col, agg_val, interpolation_val
from util_general import get_timeframe_col


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


# function to re-order the dataframe, rename the column names and scale the variables to absolute units.
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


def print_preprocess(var_name: str, timeframe: str = None):
    if show_output:
        if timeframe is None:
            print(f'Preprocessing {var_name} data...')
        else:
            print(f'Preprocessing {var_name} data for timeframe {timeframe}...')