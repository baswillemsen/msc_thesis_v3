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
    # select countries and year
    co2_monthly = select_country_year_measure(df_raw=co2_monthly_raw, country_col=country_col, year_col=year_col)
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


# Monthly CO2 data
# def preprocesss_population_quarterly(source_file: str, country_col: str, year_col: str):
#     # read data
#     pop_quarterly_raw = read_data(source_path=data_source_path, file_name=source_file)
#     # select countries and year
#     pop_quarterly = select_country_year_measure(df_raw=co2_monthly_raw, country_col=country_col, year_col=year_col)
#     # pivot
#     co2_monthly = co2_monthly.melt(id_vars=[country_col, year_col], value_vars=co2_monthly.drop([country_col, year_col], axis=1))
#     co2_monthly['month'] = co2_monthly.apply(lambda row: month_name_to_num(row.variable), axis=1)
#     co2_monthly['quarter'] = co2_monthly.apply(lambda row: month_to_quarter(row.month), axis=1)
#     co2_monthly['date'] = pd.to_datetime(dict(year=co2_monthly[year_col], month=co2_monthly['month'], day=1))
#     co2_monthly = co2_monthly.drop('variable', axis=1)
#
#     # rename, order and scale: output = [index, country, date, value]
#     co2_monthly = rename_order_scale(df=co2_monthly, country_col=country_col, date_col='date', var_col='value',
#                                      var_name='co2_monthly', var_scale=1000)
#     # export to csv
#     co2_monthly.to_csv(f'{data_path}co2_monthly_processed.csv')
#
#     # downsample monthly to quarterly
#     co2_quarterly = downsample_month_to_quarter(df_monthly=co2_monthly, country_col='country', date_col='date',
#                                                var_monthly='co2_monthly', var_quarterly='co2_quarterly')
#     # export to csv
#     co2_quarterly.to_csv(f'{data_path}co2_quarterly_processed.csv')


if __name__ == "__main__":
    preprocesss_co2_monthly(source_file='co2_monthly', country_col='Name', year_col='Year')
    # preprocesss_population_quarterly(source_file='pop_quarterly', country_col='Country', year_col='Year')
