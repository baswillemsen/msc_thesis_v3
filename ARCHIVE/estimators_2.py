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
from definitions import fake_num, show_plots, sign_level, save_figs
from helper_functions_general import flatten, get_impl_date
from helper_functions_estimation import arco_pivot, sc_pivot, transform_back, save_dataframe
from plot_functions import plot_lasso_path
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

    # check the target series is stationary
    if fake_num in list(target_log_diff):
        return None
    else:

        ################################
        ### FIRST STAND, SECOND PRE  ###
        ################################

        y_log_diff = np.array(target_log_diff).reshape(-1, 1)
        X_log_diff = np.array(donors_log_diff)

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_targetfit = SS.fit(y_log_diff)

        # Generating the standardized values of X and y
        X_log_diff_stand = SS.fit_transform(X_log_diff)
        y_log_diff_stand = SS.fit_transform(np.array(y_log_diff).reshape(-1, 1))

        # Generating the standardized values of X and y
        X_log_diff_stand_pre = X_log_diff_stand[:get_impl_date(target_country=target_country, input='index')]
        y_log_diff_stand_pre = y_log_diff_stand[:get_impl_date(target_country=target_country, input='index')]

        country_weight = {'switzerland': 0.85,
                          'ireland': 0.8,
                          'united_kingdom': 0.7,
                          'france': 0.8,
                          'portugal': 0.67
                          }

        train_weight = int(country_weight[target_country] * len(y_log_diff_stand_pre))
        y_log_diff_pre_stand_train = y_log_diff_stand_pre[:train_weight]
        X_log_diff_pre_stand_train = X_log_diff_stand_pre[:train_weight]

        if show_plots or save_figs:
            plot_lasso_path(X=X_log_diff_pre_stand_train, y=y_log_diff_pre_stand_train, target_country=target_country,
                            alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters,
                            model=model, timeframe=timeframe)

        # define model
        ts_split = TimeSeriesSplit(n_splits=ts_splits)
        # print(ts_split)
        # for i, (train_index, test_index) in enumerate(ts_split.split(X_log_diff_stand_pre)):
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
        lasso.fit(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train.ravel())  # very good results

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_targetfit.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))

        # act_pred_log_diff
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(target_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

        act_pred_log_diff_check, \
            act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, target_country=target_country,
                                                    timeframe=timeframe, pred_log_diff=pred_log_diff)

        shapiro_wilk_test(df=act_pred_log_diff, target_country=target_country, alpha=sign_level)

        # save dataframes
        save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                       model=model, target_country=target_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

        save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                       model=model, target_country=target_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

        save_dataframe(df=act_pred_log, var_title='act_pred_log',
                       model=model, target_country=target_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

        save_dataframe(df=act_pred, var_title='act_pred',
                       model=model, target_country=target_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

        print(f'R2 pre-stand: {lasso.score(X_log_diff_stand_pre, y_log_diff_stand_pre)}')
        print(f'R2 pre-stand-train: {lasso.score(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train)}')
        print(f'alpha: {lasso.alpha_}')

        coefs = list(lasso.coef_)
        coef_index = [i for i, val in enumerate(coefs) if val != 0]
        print(f'Parameters estimated ({len(donors_log_diff.columns[coef_index])}x): '
              f'{list(donors_log_diff.columns[coef_index])}')

        return act_pred_log_diff


def sc(df: object, df_stat: object, target_country: str, timeframe: str, model: str):
    # pivot target and donors
    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df_stat, target_country=target_country,
                                                           timeframe=timeframe, model=model)

    # define the SC estimator
    sc = SparseSC.fit(
        features=np.array(pre_treat),
        targets=np.array(post_treat),
        treated_units=treat_unit
    )

    # Predict the series, make act_pred dataframe
    act_pred_log_diff = df_pivot.loc[df_pivot.index == target_country].T
    act_pred_log_diff.columns = ['act']
    pred_log_diff = sc.predict(df_pivot.values)[treat_unit, :][0]
    act_pred_log_diff['pred'] = pred_log_diff
    act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

    # transform back
    act_pred_log_diff_check, \
        act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, target_country=target_country,
                                                timeframe=timeframe, pred_log_diff=pred_log_diff)

    shapiro_wilk_test(df=act_pred_log_diff, target_country=target_country, alpha=sign_level)

    # save dataframes
    save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                   model=model, target_country=target_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

    save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                   model=model, target_country=target_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    save_dataframe(df=act_pred_log, var_title='act_pred_log',
                   model=model, target_country=target_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    save_dataframe(df=act_pred, var_title='act_pred',
                   model=model, target_country=target_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    return act_pred_log_diff


def did():
    pass
