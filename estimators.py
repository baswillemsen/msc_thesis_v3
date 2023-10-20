################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

# import os
# import csv
# from datetime import datetime
# from sklearn.metrics import r2_score

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
    save_output, date_col
from helper_functions_general import flatten, get_impl_date, get_table_path, get_months_cors
from helper_functions_estimation import arco_pivot, sc_pivot, transform_back, save_dataframe, did_pivot, save_results
from plot_functions import plot_lasso_path


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, treatment_country: str, timeframe: str, ts_splits: int,
         alpha_min: float, alpha_max: float, alpha_step: float, tol: float, lasso_iters: int,
         model: str, prox: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    # pivot treatment and donors
    treatment_log_diff, donors_log_diff = arco_pivot(df=df_stat, treatment_country=treatment_country,
                                                     timeframe=timeframe, model=model, prox=prox)
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

        months_cor = get_months_cors(timeframe=timeframe, treatment_country=treatment_country)

        # for months_cor in np.arange(-24, 24, 3):
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
        r2_pre_log_diff_stand = lasso.score(X_log_diff_pre_stand, y_log_diff_pre_stand)
        lasso_alpha = lasso.alpha_
        lasso_alpha = lasso.alpha_
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

        ind = list(range(1, n_pars+1))
        df_results = pd.DataFrame(list(zip(lasso_pars, lasso_coefs)), columns=['Regressor', 'Coefficient'])
        df_results = df_results.sort_values('Coefficient', ascending=False)
        df_results['Index'] = ind
        df_results = df_results[['Index', 'Regressor', 'Coefficient']]
        df_results.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_lasso_pars.csv', index=False)

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_treatmentfit_pre.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(treatment_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

        act_pred_log_diff_check, \
            act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                    timeframe=timeframe, pred_log_diff=pred_log_diff)

        n_train = len(y_log_diff_pre_stand)
        n_test = len(y_log_diff) - len(y_log_diff_pre_stand)

        # # perform hypothesis tests
        # timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M")
        # normal_errors, shapiro_p = shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)
        # t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)
        # t_test_result(df=act_pred, treatment_country=treatment_country)

        # if save_output:
        #
        #     save_results(y_log_diff=y_log_diff, y_log_diff_pre_stand=y_log_diff_pre_stand, prox=prox,
        #                  act_pred_log_diff=act_pred_log_diff, act_pred_log=act_pred_log, act_pred=act_pred,
        #                  impl_date_index=impl_date_index, model=model, timeframe=timeframe, sign_level=sign_level,
        #                  treatment_country=treatment_country, incl_countries=incl_countries, incl_years=incl_years,
        #                  stat=stat, impl_date=impl_date, months_cor=months_cor, split_date=split_date,
        #                  r2_pre_log_diff_stand=r2_pre_log_diff_stand, var_title=f'{model}_results_optim',
        #                  lasso_alpha=lasso_alpha, n_pars=n_pars, lasso_pars=lasso_pars, lasso_coefs=lasso_coefs)

            # incl_vars = get_trans()
            # n_train = len(y_log_diff_pre_stand)
            # n_test = len(y_log_diff) - len(y_log_diff_pre_stand)
            # r2_pre_log_diff = round(r2_score(act_pred_log_diff['act'][:impl_date_index], act_pred_log_diff['pred'][:impl_date_index]), 3)
            # r2_pre_log = round(r2_score(act_pred_log['act'][:impl_date_index], act_pred_log['pred'][:impl_date_index]), 3)
            # r2_pre = round(r2_score(act_pred['act'][:impl_date_index], act_pred['pred'][:impl_date_index]), 3)
            # att_mean, att_std, \
            #     att_sign, att_p = t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)
            # colmns = ['model', 'timeframe', 'timestamp', 'treatment_country', 'incl_vars', 'incl_countries', 'incl_years', 'stat', 'impl_date', 'months_cor', 'split_date', 'n_train', 'n_test', 'r2_pre_log_diff_stand', 'r2_pre_log_diff', 'r2_pre_log', 'r2_pre', 'lasso_alpha', 'n_pars', 'lasso_pars', 'lasso_coefs', 'normal_errors', 'shapiro_p', 'att_mean', 'att_std', 'att_sign', 'att_sign']
            # result = [model,    timeframe,   timestamp,   treatment_country,   incl_vars,   incl_countries,   incl_years,   stat,   impl_date,   months_cor,   split_date,   n_train,   n_test,   r2_pre_log_diff_stand,   r2_pre_log_diff,   r2_pre_log,   r2_pre,   lasso_alpha,   n_pars,   lasso_pars,   lasso_coefs,   normal_errors,   shapiro_p,   att_mean,   att_std,   att_sign,   att_p]
            #
            # if len(result) != len(colmns):
            #     raise ValueError('Length column names in file is different from length of output')
            #
            # file_path = f'{tables_path_res}/{model}_{treatment_country}_results_optim.csv'
            # if not os.path.isfile(file_path):
            #     with open(file_path, 'w', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerow(colmns)
            #         file.close()
            # # else:
            # #     with open(file_path, 'w') as file:
            # #         reader = csv.reader(file)
            # #         if len(next(reader)) != len(result):
            # #             raise ValueError('Length column names in file is different from length of output')
            # #         file.close()
            #
            # # Create a file object for this file
            # with open(file_path, 'a', newline='') as file:
            #     print('saving results')
            #     writer = csv.writer(file)
            #     writer.writerow(result)
            #     file.close()

        # save dataframes and plots
        if save_output or show_plots or save_figs:
            plot_lasso_path(X=X_log_diff_pre_stand, y=y_log_diff_pre_stand, treatment_country=treatment_country,
                            alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters,
                            model=model, timeframe=timeframe, alpha_cv=lasso.alpha_)

            save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True,
                           save_cumsum=True, save_cumsum_impl=True, save_qq=True)

            save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=False, save_diff=False,
                           save_cumsum=False, save_cumsum_impl=False, save_qq=False)

            save_dataframe(df=act_pred_log, var_title='act_pred_log',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=False, save_diff=False,
                           save_cumsum=False, save_cumsum_impl=False, save_qq=False)

            save_dataframe(df=act_pred, var_title='act_pred',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=False,
                           save_cumsum=False, save_cumsum_impl=False, save_qq=False)

            save_results(act_pred_log_diff=act_pred_log_diff, act_pred_log=act_pred_log, act_pred=act_pred,
                         model=model, stat=stat, timeframe=timeframe, sign_level=sign_level,
                         incl_countries=incl_countries, incl_years=incl_years,
                         treatment_country=treatment_country, impl_date=impl_date, impl_date_index=impl_date_index,
                         n_train=n_train, n_test=n_test, var_title=f'{model}_results',
                         # model specific
                         prox=prox, months_cor=months_cor, split_date=split_date, r2_pre_log_diff_stand=r2_pre_log_diff_stand,
                         lasso_alpha=lasso_alpha, n_pars=n_pars, lasso_pars=lasso_pars, lasso_coefs=lasso_coefs)

        return act_pred_log_diff


def sc(df: object, df_stat: object, treatment_country: str, timeframe: str, model: str, prox: bool):
    # pivot treatment and donors
    impl_date = get_impl_date(treatment_country=treatment_country)
    impl_date_index = list(df[date_col]).index(impl_date)

    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df_stat, treatment_country=treatment_country,
                                                           timeframe=timeframe, model=model, impl_date=impl_date,
                                                           prox=prox)

    # # define the SC estimator
    # sc = SparseSC.fit_fast(
    #     features=np.array(pre_treat.T),
    #     targets=np.array(post_treat.T),
    #     treated_units=treat_unit,
    #     model_type='retrospective',
    # )
    #
    # # Predict the series, make act_pred dataframe
    # act_pred_log_diff = df_pivot[treatment_country].to_frame()
    # act_pred_log_diff.rename(columns={treatment_country: 'act'}, inplace=True)
    # pred_log_diff = sc.predict(df_pivot.T.values)[0]
    # act_pred_log_diff['pred'] = pred_log_diff
    # act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

    # standardize
    SS = StandardScaler()
    df_pivot_stand = pd.DataFrame(SS.fit_transform(df_pivot), columns=df_pivot.columns).set_index(df_pivot.index)
    pre_treat_stand = pd.DataFrame(SS.fit_transform(pre_treat), columns=pre_treat.columns).set_index(pre_treat.index)
    post_treat_stand = pd.DataFrame(SS.fit_transform(post_treat), columns=df_pivot.columns).set_index(post_treat.index)

    # define the SC estimator
    sc = SparseSC.fit_fast(
        features=np.array(pre_treat_stand.T),
        targets=np.array(post_treat_stand.T),
        treated_units=treat_unit,
        model_type='retrospective',
    )

    # Predict the series, make act_pred dataframe
    SS_treatmentfit = SS.fit(np.array(df_pivot).reshape(-1, 1))
    act_pred_log_diff = df_pivot[treatment_country].to_frame()
    act_pred_log_diff.rename(columns={treatment_country: 'act'}, inplace=True)
    pred_log_diff = SS_treatmentfit.inverse_transform(sc.predict(df_pivot_stand.T.values)[0].reshape(-1, 1))
    act_pred_log_diff['pred'] = pred_log_diff
    act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

    # transform back
    act_pred_log_diff_check, \
        act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                timeframe=timeframe, pred_log_diff=pred_log_diff)

    r2_pre_log_diff_stand = sc.score_R2
    n_train = len(pre_treat.columns)
    n_test = len(post_treat.columns)

    # save dataframes and plots
    if save_output or show_plots or save_figs:
        save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                       model=model, treatment_country=treatment_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=True,
                       save_cumsum=True, save_cumsum_impl=True, save_qq=True)

        save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                       model=model, treatment_country=treatment_country, timeframe=timeframe,
                       save_csv=True, save_predictions=False, save_diff=False,
                       save_cumsum=False, save_cumsum_impl=False, save_qq=False)

        save_dataframe(df=act_pred_log, var_title='act_pred_log',
                       model=model, treatment_country=treatment_country, timeframe=timeframe,
                       save_csv=True, save_predictions=False, save_diff=False,
                       save_cumsum=False, save_cumsum_impl=False, save_qq=False)

        save_dataframe(df=act_pred, var_title='act_pred',
                       model=model, treatment_country=treatment_country, timeframe=timeframe,
                       save_csv=True, save_predictions=True, save_diff=False,
                       save_cumsum=False, save_cumsum_impl=False, save_qq=False)

        save_results(act_pred_log_diff=act_pred_log_diff, act_pred_log=act_pred_log, act_pred=act_pred,
                     model=model, stat=stat, timeframe=timeframe, sign_level=sign_level,
                     incl_countries=incl_countries, incl_years=incl_years,
                     treatment_country=treatment_country, impl_date=impl_date, impl_date_index=impl_date_index,
                     n_train=n_train, n_test=n_test, var_title=f'{model}_results',
                     # model specific
                     prox=prox, r2_pre_log_diff_stand=r2_pre_log_diff_stand)

    return act_pred


def did(df: object, treatment_country: str, timeframe: str, model: str, prox: bool, x_years: int):

    # get treatment and donors pre- and post-treatment
    df_sel, treatment_pre, treatment_post, \
        donors_pre, donors_post = did_pivot(df=df, treatment_country=treatment_country, prox=prox,
                                            timeframe=timeframe, model=model, x_years=x_years)

    # easy diff-in-diff
    treatment_pre_mean = np.mean(treatment_pre[target_var])
    treatment_post_mean = np.mean(treatment_post[target_var])
    treatment_diff = treatment_post_mean - treatment_pre_mean

    donors_pre_mean = np.mean(donors_pre[target_var])
    donors_post_mean = np.mean(donors_post[target_var])
    donors_diff = donors_post_mean - donors_pre_mean

    diff_in_diff = treatment_diff - donors_diff
    print(f'diff-in-diff {diff_in_diff}')

    # linear regression
    lr = LinearRegression()

    X = df_sel[['treatment_dummy', 'post_dummy', 'treatment_post_dummy']]
    y = df_sel[target_var]

    lr.fit(X, y)
    print(lr.coef_)

    # extended OLS
    ols = smf.ols('co2 ~ treatment_dummy + post_dummy + treatment_post_dummy', data=df_sel).fit()
    print(ols.summary())

    return diff_in_diff


