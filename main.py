################################
### import relevant packages ###
################################
import os
import pandas as pd
from definitions import *

# custom functions
from helper_functions import pivot_target, pivot_donors
from statistical_tests import adf_test
from estimators import arco
from plot_functions import plot_predictions, plot_diff

###################################
### define paths & static defs  ###
###################################
for path in [data_path, figures_path, output_path]:
    if not os.path.exists(path):
        os.makedirs(path)


################################
### main script              ###
################################
def main():
    # read data
    df = pd.read_csv(os.path.join(data_path, data_file), delimiter=',', header=0, encoding='latin-1')
    # See which countries are included
    print(f'Countries included ({len(incl_countries)}x): {incl_countries}')
    print(f'Years included ({len(incl_years)}x): {list(incl_years)}')
    # pivot donors
    target = pivot_target(df, target_country, target_var)
    donors = pivot_donors(df, donor_countries)
    # stationarity test
    adf_test(df=donors)
    # arco
    model, act, pred = arco(target=target, donors=donors, target_impl_year=target_impl_year)
    # plot predictions versus actual
    plot_predictions(act=act, pred=pred, target_impl_year=target_impl_year)
    plot_diff(act=act, pred=pred, target_impl_year=target_impl_year)


if __name__ == "__main__":
    main()
