################################
### import relevant packages ###
################################
import os
import numpy as np
import pandas as pd

from definitions import target_var, donor_countries, country_col, date_col, save_results, fake_num, show_plots, \
    save_figs, year_col
from helper_functions_general import get_table_path, get_impl_date, get_trans
from plot_functions import plot_predictions, plot_diff, plot_cumsum


def arco_pivot(df: object, treatment_country: str, timeframe: str, model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country)

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

    if save_results:
        treatment.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_treatment.csv')
        donors.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_donors.csv')

    return treatment, donors


def sc_pivot(df: object, treatment_country: str, timeframe: str, model: str):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country)

    df_pivot = df.copy()
    df_pivot = df_pivot[df_pivot[country_col].isin(donor_countries + [treatment_country])]
    df_pivot = df_pivot.pivot(index=country_col, columns=date_col, values=target_var)
    df_pivot = df_pivot.replace({fake_num: np.nan})
    df_pivot = df_pivot.dropna(axis=0, how='any')

    pre_treat = df_pivot.iloc[:, df_pivot.columns < get_impl_date(treatment_country)]
    post_treat = df_pivot.iloc[:, df_pivot.columns >= get_impl_date(treatment_country)]
    treat_unit = [idx for idx, val in enumerate(df_pivot.index.values) if val == treatment_country]

    if save_results:
        df_pivot.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_full_pivot.csv')
        pre_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_pre_treat.csv')
        post_treat.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_post_treat.csv')

    return df_pivot, pre_treat, post_treat, treat_unit


def did_pivot(df: object, treatment_country: str, timeframe: str, model: str, x_years: int):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country)
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

    if save_results:
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
    act_pred_log_diff_check['error'] = act_pred_log_diff_check['act'] - act_pred_log_diff_check['pred']

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
    act_pred_log['error'] = act_pred_log['act'] - act_pred_log['pred']

    # act_pred
    act_pred = pd.DataFrame(list(zip(np.exp(orig_data_log), np.exp(pred2))),
                            columns=['act', 'pred']).set_index(orig_data_log.index)
    act_pred['error'] = act_pred['act'] - act_pred['pred']

    return act_pred_log_diff_check, act_pred_log, act_pred


def save_dataframe(df: object, var_title: str, model: str, treatment_country: str, timeframe: str,
                   save_csv: bool, save_predictions: bool, save_diff: bool, save_cumsum: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country)

    var_name = f'{model}_{treatment_country}_{timeframe}_{var_title}'

    if save_results:
        if save_csv:
            df.to_csv(f'{tables_path_res}/{var_name}.csv')
    if show_plots or save_figs:
        if save_predictions:
            plot_predictions(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name)
        if save_diff:
            plot_diff(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name)
        if save_cumsum:
            plot_cumsum(df=df, treatment_country=treatment_country, timeframe=timeframe, var_name=var_name)