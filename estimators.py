################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC

# custom functions
from definitions import fake_num, save_results, show_plots, sign_level, save_figs
from helper_functions import flatten, arco_pivot, sc_pivot, get_impl_date, transform_back, get_table_path
from plot_functions import plot_lasso_path, plot_predictions, plot_diff, plot_cumsum
from statistical_tests import shapiro_wilk_test


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, target_country: str, timeframe: str, ts_splits: int,
         alpha_min: float, alpha_max: float, alpha_step: float, tol: float, lasso_iters: int,
         model: str):
    # pivot target and donors
    target_log_diff, donors_log_diff = arco_pivot(df=df_stat, target_country=target_country,
                                                  timeframe=timeframe, model=model)
    print(f'Nr of parameters included ({len(donors_log_diff.columns)}x): {donors_log_diff.columns}')

    if fake_num in list(target_log_diff):
        return None, None
    else:
        tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=target_country)

        y_log_diff = np.array(target_log_diff).reshape(-1, 1)
        X_log_diff = np.array(donors_log_diff)

        y_log_diff_pre = np.array(
            target_log_diff[target_log_diff.index < get_impl_date(target_country=target_country)]).reshape(-1, 1)
        X_log_diff_pre = np.array(
            donors_log_diff[donors_log_diff.index < get_impl_date(target_country=target_country)])
        print(f'Nr of timeframes pre-intervention (t < T_0): {len(X_log_diff_pre)}')
        print(f'Nr of timeframes post-intervention (t >= T_0): {len(X_log_diff) - len(X_log_diff_pre)}')

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_targetfit = SS.fit(y_log_diff_pre)
        X_log_diff_stand = SS.fit_transform(X_log_diff)

        # Generating the standardized values of X and y
        X_log_diff_pre_stand = SS.fit_transform(X_log_diff_pre)
        y_log_diff_pre_stand = SS.fit_transform(y_log_diff_pre)

        # # Split the data into training and testing set
        X_log_diff_pre_stand_train, \
            X_log_diff_pre_stand_test, \
            y_log_diff_pre_stand_train, \
            y_log_diff_pre_stand_test = train_test_split(X_log_diff_pre_stand, y_log_diff_pre_stand,
                                                         test_size=0.25, random_state=42,
                                                         shuffle=False)

        # len_data = len(y_log_diff_pre_stand)
        # X_log_diff_pre_stand_train = X_log_diff_pre_stand[:int(0.75 * len_data)]
        # y_log_diff_pre_stand_train = y_log_diff_pre_stand[:int(0.75 * len_data)]

        if show_plots or save_figs:
            plot_lasso_path(X=X_log_diff_pre_stand_train, y=y_log_diff_pre_stand_train, target_country=target_country,
                            alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters,
                            model=model, timeframe=timeframe)

        # define model
        ts_split = TimeSeriesSplit(n_splits=ts_splits)
        # print(ts_split)
        # for i, (train_index, test_index) in enumerate(ts_split.split(X_log_diff_pre_stand)):
        #     print(f"Fold {i}:")
        #     print(f"  Train: index={train_index}")
        #     print(f"  Test:  index={test_index}")

        lasso = LassoCV(
            alphas=np.arange(alpha_min, alpha_max, alpha_step),
            fit_intercept=True,
            cv=ts_split,
            max_iter=lasso_iters,
            tol=tol,
            n_jobs=-1,
            random_state=0,
            selection='random'
        )

        # fit model
        # lasso.fit(X_log_diff_pre_stand, y_log_diff_pre_stand.ravel())  # very good results
        lasso.fit(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train.ravel())  # very good results

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_targetfit.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(target_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

        shapiro_wilk_test(df=act_pred_log_diff, target_country=target_country, alpha=sign_level)

        # save act_pred_log_diff
        var_name = f'{model}_{target_country}_{timeframe}_act_pred_log_diff'
        if save_results:
            act_pred_log_diff.to_csv(f'{tables_path_res}/{var_name}.csv')
        if show_plots or save_figs:
            plot_predictions(df=act_pred_log_diff, target_country=target_country, timeframe=timeframe,
                             var_name=var_name)
            plot_diff(df=act_pred_log_diff, target_country=target_country, timeframe=timeframe, var_name=var_name)
            plot_cumsum(df=act_pred_log_diff, target_country=target_country, timeframe=timeframe, var_name=var_name)

        act_pred_log_diff_check, \
            act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, target_country=target_country,
                                                    timeframe=timeframe, pred_log_diff=pred_log_diff, model=model)

        # save act_pred_log_diff_check
        var_name = f'{model}_{target_country}_{timeframe}_act_pred_log_diff_check'
        if save_results:
            act_pred_log_diff_check.to_csv(f'{tables_path_res}/{var_name}.csv')
        if show_plots or save_figs:
            plot_predictions(df=act_pred_log_diff_check, target_country=target_country, timeframe=timeframe,
                             var_name=var_name)

        # save act_pred_log
        var_name = f'{model}_{target_country}_{timeframe}_act_pred_log'
        if save_results:
            act_pred_log.to_csv(f'{tables_path_res}/{var_name}.csv')
        if show_plots or save_figs:
            plot_predictions(df=act_pred_log, target_country=target_country, timeframe=timeframe, var_name=var_name)

        # save act_pred
        var_name = f'{model}_{target_country}_{timeframe}_act_pred'
        if save_results:
            act_pred.to_csv(f'{tables_path_res}/{var_name}.csv')
        if show_plots or save_figs:
            plot_predictions(df=act_pred, target_country=target_country, timeframe=timeframe, var_name=var_name)

        print(f'R2 pre-stand: {lasso.score(X_log_diff_pre_stand, y_log_diff_pre_stand)}')
        print(f'R2 pre-stand-train: {lasso.score(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train)}')
        print(f'alpha: {lasso.alpha_}')
        # print(f'mse path: {lasso.mse_path_}')

        coefs = list(lasso.coef_)
        coef_index = [i for i, val in enumerate(coefs) if val != 0]
        print(f'Parameters estimated ({len(donors_log_diff.columns[coef_index])}x): '
              f'{list(donors_log_diff.columns[coef_index])}')

        # coeffs = lasso.coef_
        # print(coeffs[coeffs != 0])

        return act_pred_log_diff, act_pred_log


def sc(df: object, df_stat: object, target_country: str, timeframe: str, model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=target_country)

    # pivot target and donors
    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df_stat, target_country=target_country,
                                                           timeframe=timeframe, model=model)

    sc = SparseSC.fit(
        features=np.array(pre_treat),
        targets=np.array(post_treat),
        treated_units=treat_unit
    )

    act_pred_log_diff = df_pivot.loc[df_pivot.index == target_country].T
    act_pred_log_diff.columns = ['act']
    pred_log_diff = sc.predict(df_pivot.values)[treat_unit, :][0]
    act_pred_log_diff['pred'] = pred_log_diff
    act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

    shapiro_wilk_test(df=act_pred_log_diff, target_country=target_country, alpha=sign_level)

    if save_results:
        act_pred_log_diff.to_csv(f'{tables_path_res}/{model}_{target_country}_{timeframe}_act_pred_log_diff.csv')
    if show_plots or save_figs:
        plot_predictions_exp(df=act_pred_log_diff, target_country=target_country, timeframe=timeframe, model=model)

    act_pred_log = transform_back(df=df, df_stat=df_stat, target_country=target_country,
                                  timeframe=timeframe, pred_log_diff=pred_log_diff, model=model)

    return act_pred_log_diff, act_pred_log


def did():
    pass
