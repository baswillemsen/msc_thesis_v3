################################
### import relevant packages ###
################################
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom functions
from definitions import data_path, figures_path_meth, figures_path_res, tables_path_meth, tables_path_res, \
    country_col, year_col, stat, show_plots
from helper_functions import read_data, validate_input, get_trans, get_data_path
from plot_functions import plot_predictions, plot_diff, plot_cumsum
from estimators import arco, sc

### define paths & static defs
for path in [data_path, figures_path_meth, figures_path_res, tables_path_meth, tables_path_res]:
    if not os.path.exists(path):
        os.makedirs(path)


################################
### main script              ###
################################
def main(model: str, timeframe: str, target_country: str):

    if validate_input(model, timeframe, target_country):

        # read data
        df = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}')
        df_log_diff = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}_{stat}')

        # See which countries are included
        print(f'Target country: {target_country}')
        print(f'Parameters included: {get_trans()}')
        print(f'Countries included ({len(df[country_col].unique())}x): {df[country_col].unique()}')
        print(f'Years included ({len(df[country_col].unique())}x): {df[year_col].unique()}')

        # run the model, get back actual and predicted values
        if model == 'arco':
            model, act_pred_diff, act_pred = arco(df=df, df_stat=df_log_diff,
                                                  target_country=target_country, timeframe=timeframe,
                                                  alpha_min=0.01, alpha_max=1.0, alpha_step=0.001, lasso_iters=100000)
        elif model == 'sc':
            model, act_pred_diff, act_pred = sc(df=df, target_country=target_country)
        else:
            raise ValueError('Select a valid model: "arco" or "sc"')

        if model is None or act_pred_diff is None or act_pred is None:
            print("The GHG emissions series of the target country is non-stationary, ArCo method is not possible")
        else:
            if show_plots:
                plot_predictions(df=act_pred, target_country=target_country)
                # plot_diff(df=act_pred, target_country=target_country)
                # plot_cumsum(df=act_pred, target_country=target_country)


if __name__ == "__main__":
    main(model=sys.argv[1], timeframe=sys.argv[2], target_country=sys.argv[3])
