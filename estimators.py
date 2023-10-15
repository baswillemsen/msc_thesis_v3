################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# custom functions
from definitions import fake_num, show_plots, sign_level, save_figs, target_var, incl_years, incl_countries, stat, \
    save_output
from helper_functions_general import flatten, get_impl_date, get_table_path
from helper_functions_estimation import arco_pivot, sc_pivot, transform_back, save_dataframe, did_pivot, save_results
from plot_functions import plot_lasso_path
from statistical_tests import shapiro_wilk_test, t_test_result


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, treatment_country: str, timeframe: str, ts_splits: int,
         alpha_min: float, alpha_max: float, alpha_step: float, tol: float, lasso_iters: int,
         model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country)
    # pivot treatment and donors
    treatment_log_diff, donors_log_diff = arco_pivot(df=df_stat, treatment_country=treatment_country,
                                                     timeframe=timeframe, model=model)
    # print(f'Nr of parameters included ({len(donors_log_diff.columns)}x): {donors_log_diff.columns}')
    print(f'Nr of parameters included: {len(donors_log_diff.columns)}x')

    # check the treatment series is stationary
    if fake_num in list(treatment_log_diff):
        return None
    else:

        y_log_diff = np.array(treatment_log_diff).reshape(-1, 1)
        X_log_diff = np.array(donors_log_diff)

        impl_date = get_impl_date(treatment_country=treatment_country)
        impl_date_index = list(treatment_log_diff.index).index(impl_date)
        months_cors = {'switzerland': 6,
                       'ireland': 12,
                       'united_kingdom': -6,
                       'france': -3,
                       'portugal': 15
                       }
        months_cor = months_cors[treatment_country]

        # for months_cor in np.arange(-24, 24, 3):
        # for months_cor in [0]:
        split_index = impl_date_index + months_cor
        split_date = treatment_log_diff.index[split_index]

        y_log_diff_pre = np.array(treatment_log_diff[treatment_log_diff.index < split_date]).reshape(-1, 1)
        X_log_diff_pre = np.array(donors_log_diff[donors_log_diff.index < split_date])
        print(f'Treatment implementation date (T_0):        {impl_date}')
        print(f'Treatment split date (T_0 cor):             {split_date}')
        print(f'Nr of timeframes pre-treatment (t < T_0):   {len(X_log_diff_pre)}')
        print(f'Nr of timeframes post-treatment (t >= T_0): {len(donors_log_diff) - len(X_log_diff_pre)}')
        print("\n")

        # Storing the fit object for later reference
        SS = StandardScaler()
        # SS_treatmentfit = SS.fit(y_log_diff)
        SS_treatmentfit_pre = SS.fit(y_log_diff_pre)
        X_log_diff_stand = SS.fit_transform(X_log_diff)

        # Generating the standardized values of X and y
        X_log_diff_pre_stand = SS.fit_transform(X_log_diff_pre)
        y_log_diff_pre_stand = SS.fit_transform(y_log_diff_pre)

        # define model
        ts_split = TimeSeriesSplit(n_splits=ts_splits)

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
        lasso.fit(X_log_diff_pre_stand, y_log_diff_pre_stand.ravel())
        # lasso results
        r2_pre_log_diff_stand = round(lasso.score(X_log_diff_pre_stand, y_log_diff_pre_stand), 3)
        lasso_alpha = round(lasso.alpha_, 3)
        print(f'R2 r2_pre_log_diff_stand: {r2_pre_log_diff_stand}')
        print(f'alpha: {lasso_alpha}')

        coefs = list(lasso.coef_)
        coefs_index = [i for i, val in enumerate(coefs) if val != 0]
        n_pars = len(coefs_index)
        lasso_pars = list(donors_log_diff.columns[coefs_index])
        lasso_coefs = [round(coef,3) for coef in coefs if coef != 0]
        print(f'Parameters estimated ({n_pars}x): '
              f'{lasso_pars}')
        print(f'Coefficients estimated ({n_pars}x): '
              f'{lasso_coefs}')
        print("\n")

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_treatmentfit_pre.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))

        # act_pred_log_diff
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(treatment_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']
        # other configurations, transform back to act_pred
        act_pred_log_diff_check, \
            act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                    timeframe=timeframe, pred_log_diff=pred_log_diff)

        # perform hypothesis tests
        timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        normal_errors = shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)
        att_mean, att_std, significant = t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)

        # if save_output:
        #     incl_vars = get_trans()
        #     n_train = len(y_log_diff_pre_stand)
        #     n_test = len(y_log_diff) - len(y_log_diff_pre_stand)
        #     r2_pre_log_diff = round(r2_score(act_pred_log_diff['act'][:impl_date_index], act_pred_log_diff['pred'][:impl_date_index]), 3)
        #     r2_pre_log = round(r2_score(act_pred_log['act'][:impl_date_index], act_pred_log['pred'][:impl_date_index]), 3)
        #     r2_pre = round(r2_score(act_pred['act'][:impl_date_index], act_pred['pred'][:impl_date_index]), 3)
        #     colmns = ['model', 'timeframe', 'timestamp', 'treatment_country', 'incl_vars', 'incl_countries', 'incl_years', 'stat', 'impl_date', 'months_cor', 'split_date', 'n_train', 'n_test', 'r2_pre_log_diff_stand', 'r2_pre_log_diff', 'r2_pre_log', 'r2_pre', 'lasso_alpha', 'n_pars', 'lasso_pars', 'lasso_coefs', 'normal_errors', 'att_mean', 'att_std', 'significant']
        #     result = [model,    timeframe,   timestamp,   treatment_country,   incl_vars,   incl_countries,   incl_years,   stat,   impl_date,   months_cor,   split_date,   n_train,   n_test,   r2_pre_log_diff_stand,   r2_pre_log_diff,   r2_pre_log,   r2_pre,   lasso_alpha,   n_pars,   lasso_pars,   lasso_coefs,   normal_errors,   att_mean,   att_std,   significant]
        #
        #     if len(result) != len(colmns):
        #         raise ValueError('Length column names in file is different from length of output')
        #
        #     file_path = f'{tables_path_res}/{model}_{treatment_country}_results.csv'
        #     if not os.path.isfile(file_path):
        #         with open(file_path, 'w', newline='') as file:
        #             writer = csv.writer(file)
        #             writer.writerow(colmns)
        #             file.close()
        #     # else:
        #     #     with open(file_path, 'w') as file:
        #     #         reader = csv.reader(file)
        #     #         if len(next(reader)) != len(result):
        #     #             raise ValueError('Length column names in file is different from length of output')
        #     #         file.close()
        #
        #     # Create a file object for this file
        #     with open(file_path, 'a', newline='') as file:
        #         print('saving results')
        #         writer = csv.writer(file)
        #         writer.writerow(result)
        #         file.close()

        # save dataframes and plots
        if save_output or show_plots or save_figs:
            plot_lasso_path(X=X_log_diff_pre_stand, y=y_log_diff_pre_stand, treatment_country=treatment_country,
                            alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters,
                            model=model, timeframe=timeframe, alpha_cv=lasso.alpha_)

            save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

            save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

            save_dataframe(df=act_pred_log, var_title='act_pred_log',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

            save_dataframe(df=act_pred, var_title='act_pred',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

            save_results(y_log_diff, y_log_diff_pre_stand, act_pred_log_diff, act_pred_log, act_pred,
                         impl_date_index, model, timeframe, timestamp, treatment_country, incl_countries, incl_years,
                         stat, impl_date, months_cor, split_date, r2_pre_log_diff_stand,
                         lasso_alpha, n_pars, lasso_pars, lasso_coefs, normal_errors, att_mean, att_std, significant)

        return act_pred_log_diff


def sc(df: object, df_stat: object, treatment_country: str, timeframe: str, model: str):
    # stationary input
    # pivot treatment and donors
    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df_stat, treatment_country=treatment_country,
                                                           timeframe=timeframe, model=model)

    # define the SC estimator
    sc = SparseSC.fit(
        features=np.array(pre_treat),
        targets=np.array(post_treat),
        treated_units=treat_unit
    )

    # Predict the series, make act_pred dataframe
    act_pred_log_diff = df_pivot.loc[df_pivot.index == treatment_country].T
    act_pred_log_diff.columns = ['act']
    pred_log_diff = sc.predict(df_pivot.values)[treat_unit, :][0]
    act_pred_log_diff['pred'] = pred_log_diff
    act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

    # transform back
    act_pred_log_diff_check, \
        act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                timeframe=timeframe, pred_log_diff=pred_log_diff)

    shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)
    t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)

    # save dataframes
    save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                   model=model, treatment_country=treatment_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

    save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                   model=model, treatment_country=treatment_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    save_dataframe(df=act_pred_log, var_title='act_pred_log',
                   model=model, treatment_country=treatment_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    save_dataframe(df=act_pred, var_title='act_pred',
                   model=model, treatment_country=treatment_country, timeframe=timeframe,
                   save_csv=True, save_predictions=True, save_diff=False, save_cumsum=False)

    # # normal input = same results
    # df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df, treatment_country=treatment_country,
    #                                                        timeframe=timeframe, model=model)
    #
    # # define the SC estimator
    # sc = SparseSC.fit(
    #     features=np.array(pre_treat),
    #     targets=np.array(post_treat),
    #     treated_units=treat_unit
    # )
    #
    # # Predict the series, make act_pred dataframe
    # act_pred = df_pivot.loc[df_pivot.index == treatment_country].T
    # act_pred.columns = ['act']
    # pred = sc.predict(df_pivot.values)[treat_unit, :][0]
    # act_pred['pred'] = pred
    # act_pred['error'] = act_pred['pred'] - act_pred['act']
    #
    # shapiro_wilk_test(df=act_pred, treatment_country=treatment_country, alpha=sign_level)
    #
    # save_dataframe(df=act_pred, var_title='act_pred',
    #                model=model, treatment_country=treatment_country, timeframe=timeframe,
    #                save_csv=True, save_predictions=True, save_diff=True, save_cumsum=True)

    return act_pred


def did(df: object, df_stat: object, treatment_country: str, timeframe: str, model: str, x_years: int):
    # get treatment and donors pre- and post-treatment
    df_sel, treatment_pre, treatment_post, donors_pre, donors_post = did_pivot(df=df, treatment_country=treatment_country,
                                                                         timeframe=timeframe, model=model, x_years=x_years)
    # easy diff-in-diff
    treatment_pre_mean = np.mean(treatment_pre)
    treatment_post_mean = np.mean(treatment_post)
    treatment_diff = treatment_post_mean - treatment_pre_mean

    donors_pre_mean = np.mean(donors_pre)
    donors_post_mean = np.mean(donors_post)
    donors_diff = donors_post_mean - donors_pre_mean

    diff_in_diff = treatment_diff - donors_diff
    print(diff_in_diff)

    # linear regression
    lr = LinearRegression()

    X = df_sel[['treatment_dummy', 'post_dummy', 'treatment_post_dummy']]
    y = df_sel[target_var]

    lr.fit(X, y)
    print(lr.coef_)  # the coefficient for gt is the DID, which is 2.75

    ols = smf.ols('co2 ~ treatment_dummy + post_dummy + treatment_post_dummy', data=df_sel).fit()
    print(ols.summary())

    return diff_in_diff


