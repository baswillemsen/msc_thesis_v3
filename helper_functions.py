################################
### import relevant packages ###
################################
import pandas as pd

from definitions import target_var, data_path, incl_countries, incl_years, \
    country_col, year_col, month_col, quarter_col, date_col


def get_impl_year(target_country: str = None):
    target_countries_impl_years = {'switzerland': 2008,
                                   'ireland': 2010,
                                   'united kingdom': 2013,
                                   'france': 2014,
                                   'portugal': 2015
                                   }
    if target_country is None:
        return target_countries_impl_years
    else:
        return target_countries_impl_years[target_country]


def month_name_to_num(month_name: str):
    return {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}[month_name]


def quarter_to_month(quarter: int):
    return {1: 1, 2: 4, 3: 7, 4: 10}[quarter]


def month_to_quarter(month: int):
    return {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
            7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}[month]


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def first_value(target_country: str):
    df = read_data(source_path=data_path, file_name='total')
    orig_value = df[df['country'] == target_country].set_index('year')[target_var.replace('_stat', '')].iloc[0]
    return orig_value


def read_data(source_path: str, file_name: str):
    df = pd.read_csv(f'{source_path}{file_name}.csv', delimiter=',', header=0, encoding='latin-1')
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    return df


def select_country_year_measure(df: object, country_col: str = None, year_col: str = None,
                                measure_col: str = None, incl_measure: list = None):
    if country_col is not None:
        df = df[df[country_col].isin(incl_countries)]

    if year_col is not None:
        df = df[df[year_col].isin(incl_years)]

    if measure_col is not None:
        df = df[df[measure_col].isin(incl_measure)]

    return df


def rename_order_scale(df: object, source_country_col: str, source_year_col: str, period: str,
                       var_name: str, var_scale: float):
    df = df.rename(columns={source_country_col: country_col, source_year_col: year_col})

    if period == 'monthly':
        df = df[[country_col, date_col, year_col, month_col, var_name]]
        df = df.sort_values(by=[country_col, year_col, month_col])
    elif period == 'quarterly':
        df = df[[country_col, date_col, year_col, quarter_col, var_name]]
        df = df.sort_values(by=[country_col, year_col, quarter_col])
    else:
        ValueError('Select the period ("monthly" or "quarterly")')

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

    df_q[date_col] = [df_m[date_col][3 * i].to_pydatetime() for i in range(0, int(len(df_m)/3))]
    # df_q[date_col] = pd.to_datetime(df_q[date_col].astype(str).replace({'-10': '-12'}, regex=True)).dt.to_period('M')
    # df_q[month_col] = df_q.apply(lambda row: month_name_to_num(row.variable), axis=1)
    # co2_m['quarter'] = co2_m.apply(lambda row: month_to_quarter(row.month), axis=1)

    df_q[year_col] = df_q[date_col].dt.year
    df_q[quarter_col] = df_q[date_col].dt.quarter

    df_q = df_q[[country_col, date_col, year_col, quarter_col, var_name]]
    df_q = df_q.sort_values(by=[country_col, year_col, quarter_col])
    print(df_q)

    return df_q


def upsample_quarter_to_month(df_q: object, country_col: str, time_col: str,
                              var_quarterly: str, var_monthly: str):

    df_m = pd.DataFrame({var_monthly: [],
                               country_col: []}
                              )

    for country in df_q[country_col].unique():
        df_country = df_q.copy()
        df_country = df_country[df_country[country_col] == country]
        df_country = df_country.rename(columns={var_quarterly: var_monthly})
        df_country = df_country.set_index(time_col)[var_monthly]
        df_country = df_country.resample('M').interpolate().to_frame()
        df_country[country_col] = [country] * len(df_country)

        df_m = pd.concat([df_m, df_country], axis=0)

    df_m = df_m.reset_index()
    df_m = df_m.rename(columns={'index': time_col})
    df_m[time_col] = pd.to_datetime(df_m[time_col].astype(str))
    df_m = df_m[[country_col, time_col, var_monthly]]

    return df_m


# def pivot_target(df: object, target_country: str, target_var: str):
#     return df[df['country'] == target_country][target_var]
#
#
# def pivot_donors(df: object, donor_countries: list):
#     donors = df.copy()
#     donors = donors[donors['country'].isin(donor_countries)].reset_index(drop=True)
#     donors = donors.pivot(index='date', columns=['country'], values=donors.columns[2:])
#     donors.columns = donors.columns.to_flat_index()
#     donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
#     donors = donors.reindex(sorted(donors.columns), axis=1)
#     donors = donors.dropna(axis=1)
#     return donors


def arco_pivot(df: object, country_col: str, time_col: str, target_country: str, target_var: str, donor_countries: list):
    target = df[df[country_col] == target_country].set_index(time_col)[target_var]

    donors = df.copy()
    donors = donors[donors[country_col].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index=time_col, columns=[country_col], values=donors.columns[2:])
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)
    donors = donors.dropna(axis=1)

    return target, donors