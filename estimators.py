################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC

# custom functions
from definitions import show_results, donor_countries, fake_num, country_col, date_col, tables_path, save_results, \
    data_path, target_var, year_col, show_plots, sign_level
from helper_functions import flatten, arco_pivot, sc_pivot, get_impl_date, read_data, get_trans
from plot_functions import plot_lasso_path, plot_predictions
from statistical_tests import shapiro_wilk_test

tables_path_cor = f'{tables_path}results/'


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, target_country: str, timeframe: str,
         alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):

    # pivot target and donors
    target_diff, donors_diff = arco_pivot(df=df_stat, target_country=target_country)

    if fake_num in list(target_diff):
        return None, None
    else:
        y_diff = np.array(target_diff).reshape(-1, 1)
        X_diff = np.array(donors_diff)

        y_pre_diff = np.array(target_diff[target_diff.index <= get_impl_date(target_country=target_country)]).reshape(-1, 1)
        X_pre_diff = np.array(donors_diff[donors_diff.index <= get_impl_date(target_country=target_country)])
        # print(y_pre_diff)
        # print(X_pre_diff)

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_targetfit = SS.fit(y_pre_diff)

        # Generating the standardized values of X and y
        X_diff_stand = SS.fit_transform(X_diff)
        X_pre_diff_stand = SS.fit_transform(X_pre_diff)
        y_pre_diff_stand = SS.fit_transform(y_pre_diff)

        # Split the data into training and testing set
        # print(len(X_pre_diff_stand))
        # print(len(y_pre_diff_stand))
        X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(X_pre_diff_stand, y_pre_diff_stand,
                                                                            test_size=0.25, random_state=42,
                                                                            shuffle=False)

        # Quick sanity check with the shapes of Training and testing datasets
        # print(X_pre_train.shape)
        # print(y_pre_train.shape)

        plot_lasso_path(X=X_pre_train, y=y_pre_train, target_country=target_country,
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
        model.fit(X_pre_train, y_pre_train.ravel())

        # summarize chosen configuration
        act_diff = flatten(y_diff)
        pred_diff = flatten(SS_targetfit.inverse_transform(model.predict(X_diff_stand).reshape(-1, 1)))
        act_pred_diff = pd.DataFrame(list(zip(act_diff, pred_diff)),
                                     columns=['act', 'pred']).set_index(target_diff.index)
        act_pred_diff['error'] = act_pred_diff['pred'] - act_pred_diff['act']
        if show_plots:
            print('act_pred_errors')
            plt.plot(act_pred_diff['error'])

        shapiro_wilk_test(df=act_pred_diff, target_country=target_country, alpha=sign_level)
        if save_results:
            act_pred_diff.to_csv(f'{tables_path_cor}{target_country}/{target_country}_act_pred_diff.csv')
        if show_plots:
            print('act_pred_diff')
            plot_predictions(df=act_pred_diff, target_country=target_country)

        # summarize chosen configuration
        date_start = df_stat['date'].iloc[0]
        _, diff_level, diff_order = get_trans(timeframe=timeframe)[target_var]

        orig_data = df.copy()
        orig_data = orig_data[(orig_data[country_col] == target_country) &
                              (orig_data[date_col] >= date_start)].set_index(date_col)[target_var]
        orig_data_log = np.log(orig_data)

        if diff_order >= 1:
            orig_data_log_diff1 = orig_data_log.diff(diff_level)
            orig_data_log_diff2 = orig_data_log_diff1
        if diff_order == 2:
            orig_data_log_diff2 = orig_data_log_diff1.diff(diff_level)
        log_diff2 = pd.DataFrame(list(zip(orig_data_log_diff2, pred_diff)),
                                 columns=['act', 'pred']).set_index(orig_data_log.index)
        log_diff2.to_csv(f'{tables_path_cor}{target_country}/{target_country}_log_diff2.csv')
        if show_plots:
            print('log_diff2')
            plot_predictions(df=log_diff2, target_country=target_country)

        if diff_order == 2:
            pred1 = np.zeros(len(orig_data_log_diff1))
            pred1[diff_level:2 * diff_level] = orig_data_log_diff1[diff_level:2 * diff_level]
            for i in range(2*diff_level, len(orig_data_log_diff1)):
                pred1[i] = pred1[i - diff_level] + pred_diff[i]

        pred2 = np.zeros(len(orig_data_log))
        pred2[:diff_level] = orig_data_log[:diff_level]
        for i in range(diff_level, len(orig_data_log)):
            if diff_order == 1:
                pred2[i] = pred2[i - diff_level] + pred_diff[i]
            if diff_order == 2:
                pred2[i] = pred2[i - diff_level] + pred1[i]

        act_pred = pd.DataFrame(list(zip(orig_data_log, pred2)),
                                columns=['act', 'pred']).set_index(orig_data_log.index)
        act_pred['error'] = act_pred['pred'] - act_pred['act']
        if save_results:
            act_pred.to_csv(f'{tables_path_cor}{target_country}/{target_country}_act_pred.csv')
        if show_plots:
            print('act_pred')
            plot_predictions(df=act_pred, target_country=target_country)

        if show_results:
            print('alpha: %f' % model.alpha_)
            # print(model.coef_)
            # print(model.intercept_)
            # print(model.score)
            # print(model.get_params)
            #
            # coefs = list(model.coef_)
            # coef_index = [i for i, val in enumerate(coefs) if val != 0]
            #
            # print(len(donors.columns[coef_index]))
            # print(donors.columns[coef_index])

            coeffs = model.coef_
            print(coeffs[coeffs != 0])

        return model, act_pred_diff, act_pred


def sc(df: object, target_country: str):
    # pivot target and donors
    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df, target_country=target_country)

    model = SparseSC.fit(
        features=np.array(pre_treat),
        targets=np.array(post_treat),
        treated_units=treat_unit,
    )

    act_pred = df_pivot.loc[df_pivot.index == target_country].T
    act_pred.columns = ['act']
    act_pred['pred'] = model.predict(df_pivot.values)[treat_unit, :][0]

    act_pred_diff = []

    return model, act_pred_diff, act_pred


# def did():
#     pass