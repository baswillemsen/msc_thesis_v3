################################
### import relevant packages ###
################################
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom functions
from definitions import data_path, figures_path, tables_path, show_plots
from helper_functions import read_data
from plot_functions import plot_predictions, plot_diff
from estimators import arco, sc

### define paths & static defs
for path in [data_path, figures_path, tables_path]:
    if not os.path.exists(path):
        os.makedirs(path)


################################
### main script              ###
################################
def main(timeframe: str, target_country: str):

    # read data
    df = read_data(source_path=data_path, file_name=f'total_{timeframe}_stat')

    # See which countries are included
    print(f'Target country: {target_country}')
    print(f'Countries included ({len(df["country"].unique())}x): {df["country"].unique()}')
    print(f'Years included ({len(df["year"].unique())}x): {df["year"].unique()}')

    # run the model, get back actual and predicted values
    # model, act_pred = arco(df=df, target_country=target_country,
    #                        alpha_min=0.01, alpha_max=1.0, alpha_step=0.001, lasso_iters=100000)
    model, act_pred = sc(df=df, target_country=target_country)

    if model is None or act_pred is None:
        print("The GHG emissions series of the target country is non-stationary, ArCo method is not possible")
    else:
        # plot predictions versus actual
        if show_plots:
            plot_predictions(act_pred, target_country=target_country, timeframe=timeframe)
            plot_diff(act_pred, target_country=target_country, timeframe=timeframe)


if __name__ == "__main__":
    main(timeframe=sys.argv[1], target_country=sys.argv[2])