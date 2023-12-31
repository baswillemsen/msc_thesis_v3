# import relevant packages
import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn.metrics import r2_score

from definitions import target_var, country_col, date_col, save_output, fake_num, show_plots, save_figs
from util_general import get_table_path, get_impl_date, get_trans, get_donor_countries
from plot_functions import plot_predictions, plot_diff, plot_cumsum, plot_cumsum_impl, plot_qq
from statistical_tests import shapiro_wilk_test, sign_test_result, durbin_watson_test


# pivot the standard dataframe into needed series for the arco method
def arco_pivot(df: object, treatment_country: str, timeframe: str, model: str, prox: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    # get donor countries
    donor_countries = get_donor_countries(model=model, prox=prox, treatment_country=treatment_country)

    # get data for treatment country
    treatment = df.copy()
    treatment = treatment[treatment[country_col] == treatment_country].set_index(date_col)[target_var].to_frame()

    # get data for donor countries
    donors = df.copy()
    donors = donors[donors[country_col].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index=date_col, columns=[country_col], values=get_trans())
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)
    donors = donors.T.drop_duplicates().T
    donors = donors.dropna(axis=0)
    donors = donors.drop(columns=donors.columns[(donors == fake_num).any()])

    # save treatment and donor data
    if save_output:
        treatment.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment.csv')
        donors.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors.csv')

    return treatment, donors


# pivot the standard dataframe into needed series for the synthetic control method
def sc_pivot(df: object, treatment_country: str, timeframe: str, model: str, impl_date: str, prox: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)
    # get donor countries
    donor_countries = get_donor_countries(model=model, prox=prox, treatment_country=treatment_country)

    # get table containing treatment and donor countries
    df_pivot = df.copy()
    df_pivot = df_pivot[df_pivot[country_col].isin(donor_countries + [treatment_country])]
    df_pivot = df_pivot.pivot(index=date_col, columns=country_col, values=target_var)
    df_pivot = df_pivot.replace({fake_num: np.nan})
    df_pivot = df_pivot.dropna(axis=1, how='all')
    df_pivot = df_pivot.dropna(axis=0, how='any')

    # define pretreatment and posttreatment data
    pre_treat = df_pivot[df_pivot.index < impl_date]
    post_treat = df_pivot[df_pivot.index >= impl_date]
    treat_unit = [idx for idx, val in enumerate(df_pivot.columns) if val == treatment_country]

    # save data
    if save_output:
        df_pivot.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_full_pivot.csv')
        pre_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_pre_treat.csv')
        post_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_post_treat.csv')

    return df_pivot, pre_treat, post_treat, treat_unit


# pivot the standard dataframe into needed series for the did method
def did_pivot(df: object, treatment_country: str, timeframe: str, model: str, prox: bool, x_years: int):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)
    # select donor countries
    donor_countries = get_donor_countries(model=model, prox=prox, treatment_country=treatment_country)

    # get implementation date and select x_years before and after for evaluation
    impl_date = get_impl_date(treatment_country=treatment_country)
    impl_date_index = list(df[date_col]).index(impl_date)
    all_periods = df[date_col][impl_date_index - int(12*x_years):impl_date_index + int(12*x_years)]

    # select right countries and time periods
    df = df[(df[country_col].isin(donor_countries + [treatment_country])) &
            (df[date_col].isin(all_periods))].set_index(date_col)[[country_col, target_var]]
    df = df.replace({fake_num: np.nan})
    df = df.dropna(axis=0, how='any')

    # define dummies for DiD OLS Regression
    df_sel = df.copy()
    df_sel['treatment_dummy'] = np.where(df_sel[country_col] == treatment_country, 1, 0)
    df_sel['post_dummy'] = np.where(df_sel.index >= impl_date, 1, 0)
    df_sel['treatment_post_dummy'] = df_sel['treatment_dummy'] * df_sel['post_dummy']
    df_sel = df_sel[[country_col, target_var, 'treatment_dummy', 'post_dummy', 'treatment_post_dummy']]

    treatment_pre = df_sel[(df_sel['treatment_dummy'] == 1) & (df_sel['post_dummy'] == 0)]
    treatment_post = df_sel[(df_sel['treatment_dummy'] == 1) & (df_sel['post_dummy'] == 1)]

    donors_pre = df_sel[(df_sel['treatment_dummy'] == 0) & (df_sel['post_dummy'] == 0)]
    donors_post = df_sel[(df_sel['treatment_dummy'] == 0) & (df_sel['post_dummy'] == 1)]

    # save data
    if save_output:
        df_sel.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_df_selection.csv')
        treatment_pre.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment_pre.csv')
        treatment_post.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment_post.csv')
        donors_pre.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors_pre.csv')
        donors_post.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors_post.csv')

    return df_sel, treatment_pre, treatment_post, donors_pre, donors_post


# function to transform back the log-differenced co2 series to absolute CO2 emissions
def transform_back(df: object, act_pred_log_diff: object, timeframe: str, treatment_country: str):

    # obtain log-diff actuals and predictions
    act_log_diff = act_pred_log_diff['act']
    pred_log_diff = act_pred_log_diff['pred']

    # summarize chosen configuration
    date_start = df['date'].iloc[0]
    date_end = df['date'].iloc[-1]
    log, diff_level, = get_trans(timeframe=timeframe)[target_var]

    # get the original data (untransformed)
    orig = df.copy()
    orig = orig[(orig[country_col] == treatment_country) &
                (orig[date_col] >= date_start) &
                (orig[date_col] <= date_end)].set_index(date_col)[target_var]

    # log if necessary
    if log:
        orig_log = np.log(orig)
    else:
        orig_log = orig

    # diff if necessary
    if diff_level != 0:
        orig_log_diff = orig_log.diff(diff_level).dropna()
    else:
        orig_log_diff = orig_log.dropna()

    if sum(orig_log_diff - act_log_diff) > 1e-5:
        raise ValueError('Fault in conversion')

    # save act_pred_log_diff_check
    act_pred_log_diff_check = pd.DataFrame(list(zip(act_log_diff, orig_log_diff, pred_log_diff)),
                                           columns=['act', 'check', 'pred']).set_index(pred_log_diff.index)
    act_pred_log_diff_check['error'] = act_pred_log_diff_check['act'] - act_pred_log_diff_check['pred']

    # calculate log from log-diff by inverting difference
    act_log = np.zeros(len(orig_log))
    pred_log = np.zeros(len(orig_log))
    pred_log[:diff_level] = orig_log[:diff_level]
    act_log[:diff_level] = orig_log[:diff_level]
    for i in range(diff_level, len(orig_log)):
        if diff_level != 0:
            pred_log[i] = pred_log[i - diff_level] + pred_log_diff[i - diff_level]
            act_log[i] = act_log[i - diff_level] + act_log_diff[i - diff_level]
    # validate calculations
    if sum(orig_log - act_log) > 1e-5:
        raise ValueError('Fault in conversion')

    # save act_pred_log
    act_pred_log = pd.DataFrame(list(zip(act_log, pred_log)), columns=['act', 'pred']).set_index(orig_log.index)
    act_pred_log['error'] = act_pred_log['act'] - act_pred_log['pred']
    act_pred_log = act_pred_log.iloc[diff_level:]

    # calculate absolute from log by taking exponentials
    act = np.exp(act_log)
    pred = np.exp(pred_log)
    # validate calculations
    if sum(orig_log - act_log) > 1e-5:
        raise ValueError('Fault in conversion')

    # save act_pred
    act_pred = pd.DataFrame(list(zip(act, pred)), columns=['act', 'pred']).set_index(orig.index)
    act_pred['error'] = act_pred['act'] - act_pred['pred']
    act_pred = act_pred.iloc[diff_level:]

    # return transformed data
    return act_pred_log_diff_check, act_pred_log, act_pred


# function to save the intermediate dataframes, including plots on predictions, errors, cumsum, qq
def save_dataframe(df: object, var_title: str, model: str, treatment_country: str, timeframe: str,
                   save_csv: bool, save_predictions: bool, save_diff: bool,
                   save_cumsum: bool, save_cumsum_impl: bool, save_qq: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    var_name = f'{model}_{treatment_country}_{timeframe}_{var_title}'

    if save_output:
        if save_csv:
            df.to_csv(f'{tables_path_res}/{var_name}.csv')
    if show_plots or save_figs:
        if save_qq:
            plot_qq(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)
        if save_predictions:
            plot_predictions(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)
        if save_diff:
            plot_diff(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)
        if save_cumsum:
            plot_cumsum(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)
        if save_cumsum_impl:
            plot_cumsum_impl(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)


# save results from the arco (lasso, rf, ols) and sc methods into csv
def save_results(act_pred_log_diff, act_pred_log, act_pred, var_title, model, stat, timeframe, sign_level,
                 incl_countries, incl_years, treatment_country, impl_date, impl_date_index,
                 prox=None, r2_pre_log_diff_stand=None, lasso_alpha=None, n_pars=None, pars=None, coefs=None):

    if var_title == f'{model}_results_optim':
        tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)
    else:
        tables_path_res = get_table_path(timeframe=timeframe, folder='results')
    tables_path_res_country = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M")
    normal_errors, shapiro_p = shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country,
                                                 alpha=sign_level)
    serial_corr, db = durbin_watson_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)

    ate_log_diff_mean, ate_log_diff_std, \
        ate_log_diff_sign, ate_log_diff_p = sign_test_result(df=act_pred_log_diff, treatment_country=treatment_country)
    ate_log_mean, ate_log_std, \
        ate_log_sign, ate_log_p = sign_test_result(df=act_pred_log, treatment_country=treatment_country)
    ate_mean, ate_std, \
        ate_sign, ate_p = sign_test_result(df=act_pred, treatment_country=treatment_country)

    # return ate and p-values for log-diff, log and absolute
    if var_title == f'{model}_results':
        series = ['12-m log-diff $CO_2$', 'Log $CO_2$', 'Absolute $CO_2$']
        ate = [round(ate_log_diff_mean, 3), round(ate_log_mean, 3), round(ate_mean, 3)]
        std = [round(ate_log_diff_std, 3), round(ate_log_std, 3), round(ate_std, 3)]
        p_val = [round(ate_log_diff_p, 3), round(ate_log_p, 3), round(ate_p, 3)]
        df_res = pd.DataFrame(list(zip(series, ate, std, p_val)),
                              columns=['Series', '$\hat{\Delta}_T$ (ATE)', '$\hat{\sigma}_{\Delta_T}$ (STD)', 'p-value'])
        df_res.to_csv(f'{tables_path_res_country}/{model}_{treatment_country}_{timeframe}_ate.csv', index=False)

    # vars included in model
    incl_vars = get_trans()
    # r2 scores
    r2_pre_log_diff = r2_score(act_pred_log_diff['act'][:impl_date_index], act_pred_log_diff['pred'][:impl_date_index])
    r2_pre_log = r2_score(act_pred_log['act'][:impl_date_index], act_pred_log['pred'][:impl_date_index])
    r2_pre = r2_score(act_pred['act'][:impl_date_index], act_pred['pred'][:impl_date_index])
    colmns = ['model', 'timestamp', 'timeframe', 'stat', 'sign_level',  'treatment_country', 'incl_vars', 'incl_countries', 'incl_years', 'impl_date', 'r2_pre_log_diff', 'r2_pre_log', 'r2_pre', 'shapiro_p', 'normal_errors', 'db', 'serial_corr',  'ate_mean', 'ate_std', 'ate_p', 'ate_sign', 'ate_log_diff_mean', 'ate_log_diff_std', 'ate_log_diff_p', 'ate_log_diff_sign', 'prox', 'r2_pre_log_diff_stand', 'lasso_alpha', 'n_pars', 'pars', 'coefs']
    result = [model,    timestamp,   timeframe,   stat,   sign_level,    treatment_country,   incl_vars,   incl_countries,   incl_years,   impl_date,   r2_pre_log_diff,   r2_pre_log,   r2_pre,   shapiro_p,   normal_errors,  db,    serial_corr,    ate_mean,   ate_std,   ate_p,   ate_sign,   ate_log_diff_mean,   ate_log_diff_std,   ate_log_diff_p,   ate_log_diff_sign,   prox,   r2_pre_log_diff_stand,   lasso_alpha,   n_pars,   pars,   coefs]

    if len(result) != len(colmns):
        raise ValueError('Length column names in file is different from length of output')

    # save results to file
    file_path = f'{tables_path_res}/{var_title}.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(colmns)
            file.close()

    # Create a file object for this file
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)
        print(f'Results saved in {file_path}')
        file.close()


# save results from the did method into csv
def save_did(model, stat, timeframe, sign_level, incl_countries, incl_years, treatment_country, impl_date,
             var_title, x_years, prox, diff_in_diff, ols):

    tables_path_res = get_table_path(timeframe=timeframe, folder='results')
    timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M")

    # parameters for did regression
    incl_vars = get_trans()
    ols_n = ols.nobs
    ols_diff = ols.params['treatment_post_dummy']
    ols_t_p = ols.pvalues['treatment_post_dummy']
    ols_f_p = ols.f_pvalue
    ols_r2 = ols.rsquared
    colmns = ['model', 'timestamp', 'timeframe', 'stat', 'sign_level',  'treatment_country', 'incl_vars', 'incl_countries', 'incl_years', 'impl_date', 'x_years', 'prox', 'diff_in_diff', 'ols_n', 'ols_diff', 'ols_t_p', 'ols_f_p', 'ols_r2']
    result = [model,    timestamp,   timeframe,   stat,   sign_level,    treatment_country,   incl_vars,   incl_countries,   incl_years,   impl_date,   x_years,   prox,   diff_in_diff,   ols_n,   ols_diff,   ols_t_p,   ols_f_p,   ols_r2]

    if len(result) != len(colmns):
        raise ValueError('Length column names in file is different from length of output')

    # save results to file
    file_path = f'{tables_path_res}/{var_title}.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(colmns)
            file.close()

    # Create a file object for this file
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)
        print(f'Results saved in {file_path}')
        file.close()