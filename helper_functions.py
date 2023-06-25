################################
### import relevant packages ###
################################
import pandas as pd

from definitions import *


def month_name_to_num(month_name: str):
    return {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}[month_name]


def quarter_to_month(quarter: int):
    return {1: 1, 2: 4, 3: 7, 4: 12}[quarter]


def month_to_quarter(month: int):
    return {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
            7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}[month]


def read_data(source_path: str, file_name: str):
    df_raw = pd.read_csv(f'{source_path}{file_name}.csv')
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='Unnamed')))]
    return df_raw


def select_country_year_measure(df: object, country_col: str, year_col: str,
                                measure_col: str = None, incl_measure: list = None):
    df = df[
        (df[country_col].isin(incl_countries)) &
        (df[year_col].isin(incl_years))
    ]
    if incl_measure is not None:
        df = df[
            (df[measure_col].isin(incl_measure))
            ]

    return df


def rename_order_scale(df: object, country_col: str, date_col: str, var_col: str, var_name: str, var_scale: float):
    df = df[[country_col, date_col, var_col]]
    df.columns = ['country', 'date', var_name]
    df = df.sort_values(by=['country', 'date'])
    df[var_name] = df[var_name] * var_scale
    df = df.reset_index(drop=True)
    return df


def downsample_month_to_quarter(df_monthly: object, country_col: str, date_col: str,
                                var_monthly: str, var_quarterly: str):

    df_quarterly = pd.DataFrame({var_quarterly: [],
                                 country_col: []}
                                )

    for country in df_monthly[country_col].unique():
        df_country = df_monthly.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country = df_country.rename(columns={var_monthly: var_quarterly})
        df_country = df_country.set_index(date_col)[var_quarterly]
        df_country = df_country.resample('Q', convention='start').sum().to_frame()
        df_country[country_col] = [country] * len(df_country)

        df_quarterly = pd.concat([df_quarterly, df_country], axis=0)

    df_quarterly = df_quarterly.reset_index()
    df_quarterly = df_quarterly.rename(columns={'index': date_col})
    df_quarterly[date_col] = [df_monthly[date_col][3 * i].to_pydatetime() for i in range(0, int(len(df_monthly)/3))]
    df_quarterly[date_col] = pd.to_datetime(
        df_quarterly[date_col].astype(str).replace({'-10': '-12'}, regex=True)).dt.to_period('M')
    df_quarterly = df_quarterly[[country_col, date_col, var_quarterly]]

    return df_quarterly


def upsample_quarter_to_month(df_quarterly: object, country_col: str, date_col: str,
                              var_quarterly: str, var_monthly: str):

    df_monthly = pd.DataFrame({var_monthly: [],
                               country_col: []}
                              )

    for country in df_quarterly[country_col].unique():
        df_country = df_quarterly.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country = df_country.rename(columns={var_quarterly: var_monthly})
        df_country = df_country.set_index(date_col)[var_monthly]
        df_country = df_country.resample('M').interpolate().to_frame()
        df_country[country_col] = [country] * len(df_country)

        df_monthly = pd.concat([df_monthly, df_country], axis=0)

    df_monthly = df_monthly.reset_index()
    df_monthly = df_monthly.rename(columns={'index': date_col})
    df_monthly[date_col] = pd.to_datetime(df_monthly[date_col].astype(str))
    df_monthly = df_monthly[[country_col, date_col, var_monthly]]

    return df_monthly


def pivot_target(df: object, target_country: str, target_var: str):
    return df[df['country'] == target_country][target_var]


def pivot_donors(df: object, donor_countries: list):
    donors = df.copy()
    donors = donors[donors['country'].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index='date', columns=['country'], values=donors.columns[2:])
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)
    donors = donors.dropna(axis=1)
    return donors