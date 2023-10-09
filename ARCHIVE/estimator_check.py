################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC

# custom functions
from definitions import show_results, fake_num, country_col, date_col, tables_path_res, save_results, \
    target_var, show_plots, sign_level
from helper_functions_general import flatten, arco_pivot, sc_pivot, get_impl_date, get_trans
from plot_functions import plot_lasso_path, plot_predictions
from statistical_tests import shapiro_wilk_test


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, treatment_country: str, timeframe: str,
         alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):

    # pivot target and donors
    target_log_diff, donors_log_diff = arco_pivot(df=df_stat, treatment_country=treatment_country)

    if fake_num in list(target_log_diff):
        return None, None, None
    else:
        y_log_diff = np.array(target_log_diff).reshape(-1, 1)
        X_log_diff = np.array(donors_log_diff)

        y_log_diff_pre = np.array(target_log_diff[target_log_diff.index <= get_impl_date(treatment_country=treatment_country)]).reshape(-1, 1)
        X_log_diff_pre = np.array(donors_log_diff[donors_log_diff.index <= get_impl_date(treatment_country=treatment_country)])

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_targetfit = SS.fit(y_log_diff_pre)

        # Generating the standardized values of X and y
        X_log_diff_stand = SS.fit_transform(X_log_diff)
        X_log_diff_pre_stand = SS.fit_transform(X_log_diff_pre)
        y_log_diff_pre_stand = SS.fit_transform(y_log_diff_pre)

        # Split the data into training and testing set
        X_log_diff_pre_stand_train, X_log_diff_pre_stand_test, \
            y_log_diff_pre_stand_train, y_log_diff_pre_stand_test = train_test_split(X_log_diff_pre_stand, y_log_diff_pre_stand,
                                                                         test_size=0.25, random_state=42,
                                                                         shuffle=False)

        plot_lasso_path(X=X_log_diff_pre_stand_train, y=y_log_diff_pre_stand_train, treatment_country=treatment_country,
                        alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters)

        # define model
        ts_split = TimeSeriesSplit(n_splits=5)
        model = LassoCV(
            alphas=np.arange(0.001, 1, 0.001),
            fit_intercept=True,
            cv=ts_split,
            max_iter=1000000,
            tol=0.00001,
            n_jobs=-1,
            random_state=0,
            selection='random'
        )

        # fit model
        # model.fit(X_log_diff_pre_stand, y_log_diff_pre_stand.ravel())  # very good results
        model.fit(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train.ravel())  # very wack results

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_targetfit.inverse_transform(model.predict(X_log_diff_stand).reshape(-1, 1)))
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(target_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']

        shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)
        if save_results:
            act_pred_log_diff.to_csv(f'{tables_path_res}{treatment_country}/{treatment_country}_act_pred_log_diff.csv')
        if show_plots:
            print('act_pred_log_diff')
            plot_predictions(df=act_pred_log_diff, treatment_country=treatment_country)
            print('act_pred_errors')
            plt.plot(act_pred_log_diff['error'])

        # summarize chosen configuration
        date_start = df_stat['date'].iloc[0]
        _, diff_level, diff_order = get_trans(timeframe=timeframe)[target_var]

        orig_data = df.copy()
        orig_data = orig_data[(orig_data[country_col] == treatment_country) &
                              (orig_data[date_col] >= date_start)].set_index(date_col)[target_var]
        orig_data_log = np.log(orig_data)

        if diff_order >= 1:
            orig_data_log_diff1 = orig_data_log.diff(diff_level)
            orig_data_act_pred_log_diff_check = orig_data_log_diff1
        if diff_order == 2:
            orig_data_act_pred_log_diff_check = orig_data_log_diff1.diff(diff_level)
        act_pred_log_diff_check = pd.DataFrame(list(zip(orig_data_act_pred_log_diff_check, pred_log_diff)),
                                               columns=['act', 'pred']).set_index(orig_data_log.index)
        act_pred_log_diff_check.to_csv(f'{tables_path_res}{treatment_country}/{treatment_country}_act_pred_log_diff_check.csv')
        if show_plots:
            print('act_pred_log_diff_check')
            plot_predictions(df=act_pred_log_diff_check, treatment_country=treatment_country)

        if diff_order == 2:
            pred1 = np.zeros(len(orig_data_log_diff1))
            pred1[diff_level:2 * diff_level] = orig_data_log_diff1[diff_level:2 * diff_level]
            for i in range(2*diff_level, len(orig_data_log_diff1)):
                pred1[i] = pred1[i - diff_level] + pred_log_diff[i]

        pred2 = np.zeros(len(orig_data_log))
        pred2[:diff_level] = orig_data_log[:diff_level]
        for i in range(diff_level, len(orig_data_log)):
            if diff_order == 1:
                pred2[i] = pred2[i - diff_level] + pred_log_diff[i]
            if diff_order == 2:
                pred2[i] = pred2[i - diff_level] + pred1[i]

        act_pred_log = pd.DataFrame(list(zip(orig_data_log, pred2)),
                                columns=['act', 'pred']).set_index(orig_data_log.index)
        act_pred_log['error'] = act_pred_log['pred'] - act_pred_log['act']
        if save_results:
            act_pred_log.to_csv(f'{tables_path_res}{treatment_country}/{treatment_country}_act_pred_log.csv')
        if show_plots:
            print('act_pred')
            plot_predictions(df=act_pred_log, treatment_country=treatment_country)

        if show_results:
            print('alpha: %f' % model.alpha_)
            # print(model.coef_)
            # print(model.intercept_)
            # print(model.score)
            # print(model.get_params)

            coefs = list(model.coef_)
            coef_index = [i for i, val in enumerate(coefs) if val != 0]

            print(len(donors_log_diff.columns[coef_index]))
            print(donors_log_diff.columns[coef_index])

            coeffs = model.coef_
            print(coeffs[coeffs != 0])

        return model, act_pred_log_diff, act_pred_log


def sc(df: object, treatment_country: str):
    # pivot target and donors
    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df, treatment_country=treatment_country)

    model = SparseSC.fit(
        features=np.array(pre_treat),
        targets=np.array(post_treat),
        treated_units=treat_unit,
    )

    act_pred = df_pivot.loc[df_pivot.index == treatment_country].T
    act_pred.columns = ['act']
    act_pred['pred'] = model.predict(df_pivot.values)[treat_unit, :][0]

    act_pred_log_diff = []

    return model, act_pred_log_diff, act_pred


# def did():
#     pass