################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
from definitions import *
from helper_functions import *


# Monthly CO2 data
def preprocess_co2_m(source_file: str, country_col: str, time_col: str):
    # read data
    co2_m_raw = read_data(source_path=data_source_path, file_name=source_file)
    co2_m = co2_m_raw.copy()
    
    # lowercase, replace country names
    co2_m[country_col] = co2_m[country_col].str.lower()
    co2_m = co2_m.replace({country_col: corr_country_names})

    # select countries and year
    co2_m = select_country_year_measure(df=co2_m, country_col=country_col, time_col=time_col)

    # pivot
    co2_m = co2_m.melt(id_vars=[country_col, time_col], value_vars=co2_m.drop([country_col, time_col], axis=1))
    co2_m['month'] = co2_m.apply(lambda row: month_name_to_num(row.variable), axis=1)
    co2_m['quarter'] = co2_m.apply(lambda row: month_to_quarter(row.month), axis=1)
    co2_m['date'] = pd.to_datetime(dict(year=co2_m[time_col], month=co2_m['month'], day=1))
    co2_m = co2_m.drop('variable', axis=1)

    # rename, order and scale: output = [index, country, date, value]
    co2_m = rename_order_scale(df=co2_m, country_col=country_col, date_col='date', var_col='value',
                               var_name='co2_m', var_scale=1000)
    # export to csv
    co2_m.to_csv(f'{data_path}co2_m_processed.csv')

    # downsample monthly to quarterly
    co2_q = downsample_month_to_quarter(df_monthly=co2_m, country_col='country', date_col='date',
                                        var_monthly='co2_m', var_quarterly='co2_q')
    # export to csv
    co2_q.to_csv(f'{data_path}co2_q_processed.csv')

    return co2_m, co2_q


# Quarterly population data
def preprocess_pop_q(source_file: str, country_col: str, measure_col: str, incl_measure: list):
    # read data
    pop_q_raw = read_data(source_path=data_source_path, file_name=source_file)
    pop_q = pop_q_raw.copy()

    # lowercase, replace country names
    pop_q[country_col] = pop_q[country_col].str.lower()
    pop_q = pop_q.replace({country_col: corr_country_names})

    # transform
    pop_q['year'] = pop_q['TIME'].str[:4].astype(int)
    pop_q['quarter'] = pop_q['TIME'].str[6:].astype(int)
    pop_q['month'] = pop_q.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    pop_q['date'] = pd.to_datetime(dict(year=pop_q.year, month=pop_q.month, day=1)).dt.to_period('M')

    # select countries and year
    pop_q = select_country_year_measure(df=pop_q, country_col=country_col, time_col='year',
                                        measure_col=measure_col, incl_measure=incl_measure)
    # rename, order and scale: output = [index, country, date, value]
    pop_q = rename_order_scale(df=pop_q, country_col=country_col, date_col='date', var_col='Value',
                               var_name='pop_q', var_scale=1e3)
    # export to csv
    pop_q.to_csv(f'{data_path}pop_q_processed.csv')

    # upsample monthly to quarterly
    pop_m = upsample_quarter_to_month(df_quarterly=pop_q, country_col='country', date_col='date',
                                      var_quarterly='pop_q', var_monthly='pop_m')
    # export to csv
    pop_m.to_csv(f'{data_path}pop_m_processed.csv')

    return pop_m, pop_q


# Quarterly GDP data
def preprocess_gdp_q(source_file: str, country_col: str, measure_col: str, incl_measure: list):
    # read data
    gdp_q_raw = read_data(source_path=data_source_path, file_name=source_file)
    gdp_q = gdp_q_raw.copy()

    # lowercase, replace country names
    gdp_q[country_col] = gdp_q[country_col].str.lower()
    gdp_q = gdp_q.replace({country_col: corr_country_names})

    # transform
    gdp_q['year'] = gdp_q['TIME'].str[:4].astype(int)
    gdp_q['quarter'] = gdp_q['TIME'].str[6:].astype(int)
    gdp_q['month'] = gdp_q.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    gdp_q['date'] = pd.to_datetime(
        dict(year=gdp_q.year, month=gdp_q.month, day=1)).dt.to_period('M')

    # select countries and year
    gdp_q = select_country_year_measure(df=gdp_q, country_col=country_col, time_col='year',
                                        measure_col=measure_col, incl_measure=incl_measure)
    # rename, order and scale: output = [index, country, date, value]
    gdp_q = rename_order_scale(df=gdp_q, country_col=country_col, date_col='date', var_col='Value',
                               var_name='gdp_q', var_scale=1e6)
    # export to csv
    gdp_q.to_csv(f'{data_path}gdp_q_processed.csv')

    # upsample monthly to quarterly
    gdp_m = upsample_quarter_to_month(df_quarterly=gdp_q, country_col='country', date_col='date',
                                      var_quarterly='gdp_q', var_monthly='gdp_m')
    # export to csv
    gdp_m.to_csv(f'{data_path}gdp_m_processed.csv')

    return gdp_m, gdp_q


def total_join(time: str, co2: object, pop: object, gdp: object, date_col: str, country_col: str):
    total = co2.copy()
    total = total.merge(pop, how='left', on=[date_col, country_col])
    total = total.merge(gdp, how='left', on=[date_col, country_col])

    total[f'co2_{time}_cap'] = total[f'co2_{time}'] / total[f'pop_{time}']
    total[f'gdp_{time}_cap'] = total[f'gdp_{time}'] / total[f'pop_{time}']

    total.to_csv(f'{data_path}total_{time}.csv', header=True, index=False)

    return total


if __name__ == "__main__":
    co2_m, co2_q = preprocess_co2_m(source_file='co2_m_2000_2021',
                                    country_col='Name',
                                    time_col='Year'
                                    )

    gdp_m, gdp_q = preprocess_gdp_q(source_file='gdp_q_1990_2022',
                                    country_col='Country',
                                    measure_col='MEASURE',
                                    incl_measure=['CPCARSA']
                                    )

    pop_m, pop_q = preprocess_pop_q(source_file='pop_q_1995_2022',
                                    country_col='Country',
                                    measure_col='MEASURE',
                                    incl_measure=['PERSA']
                                    )

    total_monthly = total_join(time='m', co2=co2_m, pop=pop_m, gdp=gdp_m,
                               date_col='date', country_col='country')

    total_quarterly = total_join(time='q', co2=co2_q, pop=pop_q, gdp=gdp_q,
                                 date_col='date', country_col='country')
