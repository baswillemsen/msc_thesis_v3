################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
from definitions import *
from helper_functions import *


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
    co2_m = co2_m.melt(id_vars=[source_country_col, source_year_col], value_vars=co2_m.drop([source_country_col, source_year_col], axis=1),
                       value_name=var_name)
    co2_m[month_col] = co2_m.apply(lambda row: month_name_to_num(row.variable), axis=1)
    co2_m[date_col] = pd.to_datetime(dict(year=co2_m[source_year_col], month=co2_m[month_col], day=1))
    co2_m = co2_m.drop('variable', axis=1)

    # rename, order and scale: output = [index, country, date, value]
    co2_m = rename_order_scale(df=co2_m, source_country_col=source_country_col, source_year_col=source_year_col,
                               var_name=var_name, var_scale=1e6, period='monthly')
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
                              period='quarterly', var_name=var_name, var_scale=1e6)

    # upsample monthly to quarterly
    df_m = upsample_quarter_to_month(df_q=df_q, var_name=var_name)
    df_q[date_col] = pd.to_datetime(dict(year=df_q[year_col], month=df_q[quarter_col].apply(quarter_to_month), day=1))

    # export to csv
    df_q.to_csv(f'{data_path}{var_name}_q.csv')
    df_m.to_csv(f'{data_path}{var_name}_m.csv')

    return df_m, df_q


def total_join(co2: object, pop: object, gdp: object, key_cols: list, time: str):
    total = co2.copy()
    total = total.merge(gdp, how='left', on=key_cols)

    total = total.merge(pop, how='left', on=key_cols)

    total[f'co2_cap'] = total[f'co2'] / total[f'pop']
    total[f'gdp_cap'] = total[f'gdp'] / total[f'pop']

    total.to_csv(f'{data_path}total_{time}.csv', header=True, index=False)

    return total


if __name__ == "__main__":
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

    total_m = total_join(co2=co2_m, pop=pop_m, gdp=gdp_m, key_cols=[country_col, date_col, year_col, month_col], time='m')
    total_q = total_join(co2=co2_q, pop=pop_q, gdp=gdp_q, key_cols=[country_col, date_col, year_col, quarter_col], time='q')