################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn.metrics import r2_score

from definitions import target_var, country_col, date_col, save_output, fake_num, show_plots, \
    save_figs, year_col
from helper_functions_general import get_table_path, get_impl_date, get_trans
from plot_functions import plot_predictions, plot_diff, plot_cumsum, plot_cumsum_impl, plot_qq
from statistical_tests import shapiro_wilk_test, t_test_result


def arco_pivot(df: object, treatment_country: str, donor_countries: list, timeframe: str, model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    treatment = df.copy()
    treatment = treatment[treatment[country_col] == treatment_country].set_index(date_col)[target_var].to_frame()

    donors = df.copy()
    donors = donors[donors[country_col].isin(donor_countries)].reset_index(drop=True)
    donors = donors.pivot(index=date_col, columns=[country_col], values=get_trans())
    donors.columns = donors.columns.to_flat_index()
    donors.columns = [str(col_name[1]) + ' ' + str(col_name[0]) for col_name in donors.columns]
    donors = donors.reindex(sorted(donors.columns), axis=1)

    donors = donors.T.drop_duplicates().T
    donors = donors.dropna(axis=0)

    donors = donors.drop(columns=donors.columns[(donors == fake_num).any()])

    if save_output:
        treatment.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment.csv')
        donors.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors.csv')

    return treatment, donors


def sc_pivot(df: object, treatment_country: str, donor_countries: list, timeframe: str, model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    df_pivot = df.copy()
    df_pivot = df_pivot[df_pivot[country_col].isin(donor_countries + [treatment_country])]
    df_pivot = df_pivot.pivot(index=country_col, columns=date_col, values=target_var)
    df_pivot = df_pivot.replace({fake_num: np.nan})
    df_pivot = df_pivot.dropna(axis=0, how='any')

    pre_treat = df_pivot.iloc[:, df_pivot.columns < get_impl_date(treatment_country)]
    post_treat = df_pivot.iloc[:, df_pivot.columns >= get_impl_date(treatment_country)]
    treat_unit = [idx for idx, val in enumerate(df_pivot.index.values) if val == treatment_country]

    if save_output:
        df_pivot.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_full_pivot.csv')
        pre_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_pre_treat.csv')
        post_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_post_treat.csv')

    return df_pivot, pre_treat, post_treat, treat_unit


def did_pivot(df: object, treatment_country: str, donor_countries: list, timeframe: str, model: str, x_years: int):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)
    impl_year = get_impl_date(treatment_country=treatment_country, input='dt').year

    df = df[df[country_col].isin(donor_countries + [treatment_country])]

    treatment = df[df[country_col] == treatment_country].set_index(date_col)
    treatment_pre = treatment[treatment[year_col] == (impl_year - x_years)][target_var]
    treatment_post = treatment[treatment[year_col] == (impl_year + x_years)][target_var]

    donors = df.copy()
    donors = donors[donors[country_col].isin(donor_countries)].set_index(date_col)
    donors_pre = donors[donors[year_col] == (impl_year - x_years)][target_var]
    donors_post = donors[donors[year_col] == (impl_year + x_years)][target_var]

    df_sel = df.copy()
    # df_sel['treatment'] = [1 if df_sel[country_col] == treatment_country else 0]
    # df_sel['post_treat'] = [1 if df_sel[year_col] == (impl_year + x_years) else 0]
    df_sel['treatment_dummy'] = np.where(df_sel[country_col] == treatment_country, 1, 0)
    df_sel['post_dummy'] = np.where(df_sel[year_col] == (impl_year + x_years), 1, 0)
    df_sel = df_sel[df_sel[year_col].isin([impl_year - x_years, impl_year + x_years])]
    df_sel = df_sel.set_index(date_col)[[target_var, 'treatment_dummy', 'post_dummy']]
    df_sel['treatment_post_dummy'] = df_sel['treatment_dummy'] * df_sel['post_dummy']

    if save_output:
        df_sel.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_df_selection.csv')
        treatment_pre.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment_pre.csv')
        treatment_post.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment_post.csv')
        donors_pre.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors_pre.csv')
        donors_post.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors_post.csv')

    return df_sel, treatment_pre, treatment_post, donors_pre, donors_post


def transform_back(df: object, df_stat: object, pred_log_diff: object, timeframe: str, treatment_country: str):

    # summarize chosen configuration
    date_start = df_stat['date'].iloc[0]
    date_end = df_stat['date'].iloc[-1]
    _, diff_level, diff_order = get_trans(timeframe=timeframe)[target_var]

    orig_data = df.copy()
    orig_data = orig_data[(orig_data[country_col] == treatment_country) &
                          (orig_data[date_col] >= date_start) &
                          (orig_data[date_col] <= date_end)].set_index(date_col)[target_var]
    orig_data_log = np.log(orig_data)

    if diff_order >= 1:
        orig_data_log_diff1 = orig_data_log.diff(diff_level)
        orig_data_act_pred_log_diff_check = orig_data_log_diff1
    if diff_order == 2:
        orig_data_act_pred_log_diff_check = orig_data_log_diff1.diff(diff_level)

    # save act_pred_log_diff_check
    act_pred_log_diff_check = pd.DataFrame(list(zip(orig_data_act_pred_log_diff_check, pred_log_diff)),
                                           columns=['act', 'pred']).set_index(orig_data_log.index)
    act_pred_log_diff_check['error'] = act_pred_log_diff_check['pred'] - act_pred_log_diff_check['act']

    if diff_order == 2:
        pred1 = np.zeros(len(orig_data_log_diff1))
        pred1[diff_level:2 * diff_level] = orig_data_log_diff1[diff_level:2 * diff_level]
        for i in range(2 * diff_level, len(orig_data_log_diff1)):
            pred1[i] = pred1[i - diff_level] + pred_log_diff[i]

    pred2 = np.zeros(len(orig_data_log))
    pred2[:diff_level] = orig_data_log[:diff_level]
    for i in range(diff_level, len(orig_data_log)):
        if diff_order == 1:
            pred2[i] = pred2[i - diff_level] + pred_log_diff[i]
        if diff_order == 2:
            pred2[i] = pred2[i - diff_level] + pred1[i]

    # act_pred_log
    act_pred_log = pd.DataFrame(list(zip(orig_data_log, pred2)),
                                columns=['act', 'pred']).set_index(orig_data_log.index)
    act_pred_log['error'] = act_pred_log['pred'] - act_pred_log['act']

    # act_pred
    act_pred = pd.DataFrame(list(zip(np.exp(orig_data_log), np.exp(pred2))),
                            columns=['act', 'pred']).set_index(orig_data_log.index)
    act_pred['error'] = act_pred['pred'] - act_pred['act']

    return act_pred_log_diff_check, act_pred_log, act_pred


def save_dataframe(df: object, var_title: str, model: str, treatment_country: str, timeframe: str,
                   save_csv: bool, save_predictions: bool, save_diff: bool, save_cumsum: bool, save_qq: bool):
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
            plot_cumsum_impl(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name, model=model)


def save_results(y_log_diff, y_log_diff_pre_stand, act_pred_log_diff, act_pred_log, act_pred,
                 impl_date_index, model, timeframe, sign_level, treatment_country, prox, incl_countries, incl_years,
                 stat, impl_date, months_cor, split_date, r2_pre_log_diff_stand,
                 lasso_alpha=None, n_pars=None, lasso_pars=None, lasso_coefs=None):

    tables_path_res = get_table_path(timeframe=timeframe, folder='results')
    tables_path_res_country = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M")
    normal_errors, shapiro_p = shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country,
                                                 alpha=sign_level)
    att_log_diff_mean, att_log_diff_std, \
        att_log_diff_sign, att_log_diff_p = t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)
    att_mean, att_std, \
        att_sign, att_p = t_test_result(df=act_pred, treatment_country=treatment_country)

    series = ['12-m Log-Diff $CO_2$', 'Absolute $CO_2$']
    att = [round(att_log_diff_mean, 3), round(att_mean, 3)]
    std = [round(att_log_diff_std, 3), round(att_std, 3)]
    p_val = [round(att_log_diff_p, 3), round(att_p, 3)]
    df_arco = pd.DataFrame(list(zip(series, att, std, p_val)),
                           columns=['Series', '$\hat{\Delta}_T$ (ATT)', '$\hat{\sigma}_{\Delta_T}$ (STD)', 'P-value'])
    df_arco.to_csv(f'{tables_path_res_country}/{model}_{treatment_country}_{timeframe}_att.csv', index=False)

    incl_vars = get_trans()
    n_train = len(y_log_diff_pre_stand)
    n_test = len(y_log_diff) - len(y_log_diff_pre_stand)
    r2_pre_log_diff = r2_score(act_pred_log_diff['act'][:impl_date_index], act_pred_log_diff['pred'][:impl_date_index])
    r2_pre_log = r2_score(act_pred_log['act'][:impl_date_index], act_pred_log['pred'][:impl_date_index])
    r2_pre = r2_score(act_pred['act'][:impl_date_index], act_pred['pred'][:impl_date_index])
    colmns = ['model', 'timeframe', 'timestamp', 'treatment_country', 'prox', 'incl_vars', 'incl_countries', 'incl_years', 'stat', 'sign_level',  'impl_date', 'months_cor', 'split_date', 'n_train', 'n_test', 'r2_pre_log_diff_stand', 'r2_pre_log_diff', 'r2_pre_log', 'r2_pre', 'lasso_alpha', 'n_pars', 'lasso_pars', 'lasso_coefs', 'normal_errors', 'shapiro_p', 'att_mean', 'att_std', 'att_sign', 'att_p', 'att_log_diff_mean', 'att_log_diff_std', 'att_log_diff_sign', 'att_log_diff_p']
    result = [model,    timeframe,   timestamp,   treatment_country,   prox,   incl_vars,   incl_countries,   incl_years,   stat,   sign_level,    impl_date,   months_cor,   split_date,   n_train,   n_test,   r2_pre_log_diff_stand,   r2_pre_log_diff,   r2_pre_log,   r2_pre,   lasso_alpha,   n_pars,   lasso_pars,   lasso_coefs,   normal_errors,   shapiro_p,   att_mean,   att_std,   att_sign,   att_p,   att_log_diff_mean,   att_log_diff_std,   att_log_diff_sign,   att_log_diff_p]

    if len(result) != len(colmns):
        raise ValueError('Length column names in file is different from length of output')

    file_path = f'{tables_path_res}/{model}_results.csv'
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