################################
### import relevant packages ###
################################
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom functions
from definitions import data_path, figures_path, tables_path, data_file, show_plots
from helper_functions import read_data
from plot_functions import plot_predictions, plot_diff
from estimators import arco, sc

###################################
### define paths & static defs  ###
###################################
for path in [data_path, figures_path, tables_path]:
    if not os.path.exists(path):
        os.makedirs(path)


# ################################
# ### main script              ###
# ################################
# def main():
#
#     # read data
#     df = pd.read_csv(os.path.join(data_path, data_file), delimiter=',', header=0, encoding='latin-1')
#     # See which countries are included
#     print(f'Countries included ({len(incl_countries)}x): {incl_countries}')
#     print(f'Years included ({len(incl_years)}x): {list(incl_years)}')
#     # pivot donors
#     target = pivot_target(df, target_country, target_var)
#     donors = pivot_donors(df, donor_countries)
#     # arco
#     model, act, pred = arco(target=target, donors=donors,
#                             alpha_min=0.01, alpha_max=1.0, alpha_step=0.001, lasso_iters=100000)
#     # plot predictions versus actual
#     plot_predictions(act=act, pred=pred)
#     plot_diff(act=act, pred=pred)


def main(target_country: str):

    # read data
    df = read_data(source_path=data_path, file_name=data_file)
    # See which countries are included
    print(f'Target country: {target_country}')
    print(f'Countries included ({len(df["country"].unique())}x): {df["country"].unique()}')
    print(f'Years included ({len(df["year"].unique())}x): {df["year"].unique()}')

    # run the model, get back actual and predicted values
    model, act_pred = arco(df=df, target_country=target_country, year_start=year_start,
                           alpha_min=0.01, alpha_max=1.0, alpha_step=0.001, lasso_iters=100000)
    # model, act_pred = sc(df=df, target_country=target_country, year_start=year_start)

    if model is None or act_pred is None:
        print("The GHG emissions series of the target country is non-stationary, ArCo method is not possible")
    else:
        # plot predictions versus actual
        if show_plots:
            plot_predictions(act_pred, target_country=target_country, year_start=year_start)
            plot_diff(act_pred, target_country=target_country, year_start=year_start)


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    main(target_country=sys.argv[2])