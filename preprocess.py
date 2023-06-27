################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
from definitions import *
from helper_functions import *


# Monthly CO2 data
def preprocesss_co2_monthly(source_file: str, country_col: str, year_col: str):
    # read data
    co2_monthly_raw = read_data(source_path=data_source_path, file_name=source_file)
    co2_monthly = co2_monthly_raw.copy()
    # lowercase, replace country names
    co2_monthly[country_col] = co2_monthly[country_col].str.lower()
    co2_monthly = co2_monthly.replace({country_col: corr_country_names})
    # select countries and year
    co2_monthly = select_country_year_measure(df=co2_monthly, country_col=country_col, year_col=year_col)
    # pivot
    co2_monthly = co2_monthly.melt(id_vars=[country_col, year_col], value_vars=co2_monthly.drop([country_col, year_col], axis=1))
    co2_monthly['month'] = co2_monthly.apply(lambda row: month_name_to_num(row.variable), axis=1)
    co2_monthly['quarter'] = co2_monthly.apply(lambda row: month_to_quarter(row.month), axis=1)
    co2_monthly['date'] = pd.to_datetime(dict(year=co2_monthly[year_col], month=co2_monthly['month'], day=1))
    co2_monthly = co2_monthly.drop('variable', axis=1)

    # rename, order and scale: output = [index, country, date, value]
    co2_monthly = rename_order_scale(df=co2_monthly, country_col=country_col, date_col='date', var_col='value',
                                     var_name='co2_monthly', var_scale=1000)
    # export to csv
    co2_monthly.to_csv(f'{data_path}co2_monthly_processed.csv')

    # downsample monthly to quarterly
    co2_quarterly = downsample_month_to_quarter(df_monthly=co2_monthly, country_col='country', date_col='date',
                                               var_monthly='co2_monthly', var_quarterly='co2_quarterly')
    # export to csv
    co2_quarterly.to_csv(f'{data_path}co2_quarterly_processed.csv')

    return co2_monthly, co2_quarterly


# Quarterly population data
def preprocesss_pop_quarterly(source_file: str, country_col: str, measure_col: str, incl_measure: list):
    # read data
    pop_quarterly_raw = read_data(source_path=data_source_path, file_name=source_file)
    pop_quarterly = pop_quarterly_raw.copy()
    # lowercase, replace country names
    pop_quarterly[country_col] = pop_quarterly[country_col].str.lower()
    pop_quarterly = pop_quarterly.replace({country_col: corr_country_names})
    # transform
    pop_quarterly['year'] = pop_quarterly['TIME'].str[:4].astype(int)
    pop_quarterly['quarter'] = pop_quarterly['TIME'].str[6:].astype(int)
    pop_quarterly['month'] = pop_quarterly.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    pop_quarterly['date'] = pd.to_datetime(
        dict(year=pop_quarterly.year, month=pop_quarterly.month, day=1)).dt.to_period('M')

    # select countries and year
    pop_quarterly = select_country_year_measure(df=pop_quarterly, country_col=country_col, year_col='year',
                                                measure_col=measure_col, incl_measure=incl_measure)
    # rename, order and scale: output = [index, country, date, value]
    pop_quarterly = rename_order_scale(df=pop_quarterly, country_col=country_col, date_col='date', var_col='Value',
                                       var_name='pop_quarterly', var_scale=1e3)
    # export to csv
    pop_quarterly.to_csv(f'{data_path}pop_quarterly_processed.csv')

    # upsample monthly to quarterly
    pop_monthly = upsample_quarter_to_month(df_quarterly=pop_quarterly, country_col='country', date_col='date',
                                            var_quarterly='pop_quarterly', var_monthly='pop_monthly')
    # export to csv
    pop_monthly.to_csv(f'{data_path}pop_monthly_processed.csv')

    return pop_monthly, pop_quarterly


# Quarterly GDP data
def preprocesss_gdp_quarterly(source_file: str, country_col: str, measure_col: str, incl_measure: list):
    # read data
    gdp_quarterly_raw = read_data(source_path=data_source_path, file_name=source_file)
    gdp_quarterly = gdp_quarterly_raw.copy()
    # lowercase, replace country names
    gdp_quarterly[country_col] = gdp_quarterly[country_col].str.lower()
    gdp_quarterly = gdp_quarterly.replace({country_col: corr_country_names})
    # transform
    gdp_quarterly['year'] = gdp_quarterly['TIME'].str[:4].astype(int)
    gdp_quarterly['quarter'] = gdp_quarterly['TIME'].str[6:].astype(int)
    gdp_quarterly['month'] = gdp_quarterly.apply(lambda row: quarter_to_month(row.quarter), axis=1)
    gdp_quarterly['date'] = pd.to_datetime(
        dict(year=gdp_quarterly.year, month=gdp_quarterly.month, day=1)).dt.to_period('M')

    # select countries and year
    gdp_quarterly = select_country_year_measure(df=gdp_quarterly, country_col=country_col, year_col='year',
                                                measure_col=measure_col, incl_measure=incl_measure)
    # rename, order and scale: output = [index, country, date, value]
    gdp_quarterly = rename_order_scale(df=gdp_quarterly, country_col=country_col, date_col='date', var_col='Value',
                                       var_name='gdp_quarterly', var_scale=1e6)
    # export to csv
    gdp_quarterly.to_csv(f'{data_path}gdp_quarterly_processed.csv')

    # upsample monthly to quarterly
    gdp_monthly = upsample_quarter_to_month(df_quarterly=gdp_quarterly, country_col='country', date_col='date',
                                            var_quarterly='gdp_quarterly', var_monthly='gdp_monthly')
    # export to csv
    gdp_monthly.to_csv(f'{data_path}gdp_monthly_processed.csv')

    return gdp_monthly, gdp_quarterly


def total_join(time: str, co2: object, pop: object, gdp: object, date_col: str, country_col: str):
    total = co2.copy()
    total = total.merge(pop, how='left', on=[date_col, country_col])
    total = total.merge(gdp, how='left', on=[date_col, country_col])

    total[f'co2_{time}_cap'] = total[f'co2_{time}'] / total[f'pop_{time}']
    total[f'gdp_{time}_cap'] = total[f'gdp_{time}'] / total[f'pop_{time}']

    total.to_csv(f'{data_path}total_{time}.csv', header=True, index=False)

    return total


if __name__ == "__main__":
    co2_monthly, co2_quarterly = preprocesss_co2_monthly(source_file='co2_monthly', country_col='Name', year_col='Year')
    pop_monthly, pop_quarterly = preprocesss_pop_quarterly(source_file='pop_quarterly', country_col='Country',
                                                           measure_col='MEASURE', incl_measure=['PERSA'])
    gdp_monthly, gdp_quarterly = preprocesss_gdp_quarterly(source_file='gdp_quarterly', country_col='Country',
                                                           measure_col='MEASURE', incl_measure=['CPCARSA'])

    total_monthly = total_join(time='monthly', co2=co2_monthly, pop=pop_monthly, gdp=gdp_monthly,
                               date_col='date', country_col='country')
    total_quarterly = total_join(time='quarterly', co2=co2_quarterly, pop=pop_quarterly, gdp=gdp_quarterly,
                                 date_col='date', country_col='country')
