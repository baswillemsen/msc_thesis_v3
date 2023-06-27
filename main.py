################################
### import relevant packages ###
################################
import os
import pandas as pd
from definitions import *

# custom functions
from helper_functions import pivot_target, pivot_donors
from estimators import arco, sc, did
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
    # arco
    model, act, pred = arco(target=target, donors=donors,
                            alpha_min=0.01, alpha_max=1.0, alpha_step=0.001, lasso_iters=100000)
    # plot predictions versus actual
    plot_predictions(act=act, pred=pred)
    plot_diff(act=act, pred=pred)


if __name__ == "__main__":
    main()
