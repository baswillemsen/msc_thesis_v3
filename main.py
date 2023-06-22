################################
### import relevant packages ###
################################
import os
import pandas as pd
from definitions import *

# custom functions
from helper_functions import pivot_target, pivot_donors
from helper_functions import adf_test
from estimators import arco
from plot_functions import plot_predictions, plot_diff

###################################
### define paths & static defs  ###
###################################
for path in [data_path, figures_path, output_path]:
    if not os.path.exists(path):
        os.makedirs(path)


################################
### define variables         ###
################################
data_file = 'total_monthly.csv'
target_countries_impl_years = {'Switzerland': 2008, 'Iceland': 2010, 'Ireland': 2010, 'France': 2014, 'Portugal': 2015}
target_countries = list(target_countries_impl_years.keys())
donor_countries = ['Austria', 'Belgium', 'Bulgaria',
                   #                    'Cyprus',
                   'Croatia', 'Czech Republic',
                   'Germany', 'Greece', 'Hungary', 'Italy', 'Lithuania', 'Netherlands',
                   'Romania', 'Slovakia', 'Spain']

target_country = 'Iceland'
target_var = 'co2_monthly'
target_impl_year = target_countries_impl_years[target_country]

incl_countries = list(target_country) + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)
print(f'Countries included ({len(incl_countries)}x): {incl_countries}')
print(f'Years included ({len(incl_years)}x): {list(incl_years)}')


################################
### main script              ###
################################
def main():
    # read data
    df = pd.read_csv(os.path.join(data_path, data_file), delimiter=',', header=0, encoding='latin-1')
    # pivot donors
    target = pivot_target(df, target_country, target_var)
    donors = pivot_donors(df, donor_countries)
    # adfuller test
    adf_test(df=donors)
    # arco
    model, act, pred = arco(target=target, donors=donors, target_impl_year=target_impl_year)

    plot_predictions(act=act, pred=pred, target_impl_year=target_impl_year)
    plot_diff(act=act, pred=pred, target_impl_year=target_impl_year)


if __name__ == "__main__":
    main()
