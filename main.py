################################
### import relevant packages ###
################################
import os
import sys
import openpyxl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# custom functions
from definitions import all_paths, country_col, year_col, stat, incl_countries, incl_years
from helper_functions_general import read_data, validate_input, get_trans, get_data_path, get_impl_date
from estimators import arco, sc, did

### define paths & static defs
for path in all_paths:
    if not os.path.exists(path):
        os.makedirs(path)


################################
### main script              ###
################################
def main(model: str, timeframe: str, treatment_country: str):
    if validate_input(model, stat, timeframe, treatment_country):

        # read data
        df = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}')
        df_log_diff = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}_{stat}')

        df = df[(df[country_col].isin(incl_countries)) & (df[year_col].isin(incl_years))]
        df_log_diff = df_log_diff[(df_log_diff[country_col].isin(incl_countries)) & (df_log_diff[year_col].isin(incl_years))]

        # See which countries are included
        print(f'Treatment country: {treatment_country} ({get_impl_date(treatment_country=treatment_country)[:4]})')
        print(f'Variables included ({len(get_trans())}x): {get_trans()}')
        print(f'Countries included ({len(df[country_col].unique())}x): {df[country_col].unique()}')
        print(f'Years included ({len(df[year_col].unique())}x): {df[year_col].unique()}')

        # run the model, get back actual and predicted values
        if model == 'arco':
            act_pred_log_diff = arco(df=df, df_stat=df_log_diff, treatment_country=treatment_country, timeframe=timeframe,
                                     alpha_min=0.001, alpha_max=1.0, alpha_step=0.001, ts_splits=5,
                                     lasso_iters=100000000, tol=0.00000001, model=model, prox=False)
        elif model == 'sc':
            act_pred_log_diff = sc(df=df, df_stat=df_log_diff, treatment_country=treatment_country, timeframe=timeframe,
                                   model=model)
        elif model == 'did':
            act_pred_log_diff = did(df=df, df_stat=df_log_diff, treatment_country=treatment_country, timeframe=timeframe,
                                    model=model, x_years=5)
        else:
            raise ValueError('Select a valid model: "arco", "sc" or "did"')

        if act_pred_log_diff is None:
            print("The GHG emissions series of the treatment country is non-stationary, ArCo method is not possible")


if __name__ == "__main__":
    main(model=sys.argv[1], timeframe=sys.argv[2], treatment_country=sys.argv[3])
