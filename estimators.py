################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

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
from definitions import fake_num, show_plots, sign_level, save_figs, target_var
from helper_functions_general import flatten, get_impl_date
from helper_functions_estimation import arco_pivot, sc_pivot, transform_back, save_dataframe, did_pivot
from plot_functions import plot_lasso_path
from statistical_tests import shapiro_wilk_test, t_test_result


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, treatment_country: str, timeframe: str, ts_splits: int,
         alpha_min: float, alpha_max: float, alpha_step: float, tol: float, lasso_iters: int,
         model: str):
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

        y_log_diff_pre = np.array(
            treatment_log_diff[treatment_log_diff.index < get_impl_date(treatment_country=treatment_country)]).reshape(-1, 1)
        X_log_diff_pre = np.array(
            donors_log_diff[donors_log_diff.index < get_impl_date(treatment_country=treatment_country)])
        print(f'Nr of timeframes pre-intervention (t < T_0): {len(X_log_diff_pre)}')
        print(f'Nr of timeframes post-intervention (t >= T_0): {len(donors_log_diff) - len(X_log_diff_pre)}')
        print("\n")

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_treatmentfit = SS.fit(y_log_diff_pre)
        X_log_diff_stand = SS.fit_transform(X_log_diff)

        # Generating the standardized values of X and y
        X_log_diff_pre_stand = SS.fit_transform(X_log_diff_pre)
        y_log_diff_pre_stand = SS.fit_transform(y_log_diff_pre)

        # country_weights = np.arange(0.5, 1.0, 0.05)
        # country_weights = [0.5]
        # for country_weight in country_weights:
        # country_weight = {'switzerland': 0.9,
        #                   'ireland': 1.0,
        #                   'united_kingdom': 0.95,
        #                   'france': 1.0,
        #                   'portugal': 0.65,
        #                   'belgium': 1.0
        #                   }
        country_weight = {'switzerland': 1,
                          'ireland': 1,
                          'united_kingdom': 1,
                          'france': 1,
                          'portugal': 1,
                          'belgium': 1
                          }
        train_weight = int(country_weight[treatment_country] * len(y_log_diff_pre_stand))
        # print(f'country_weight: {country_weight}')
        # train_weight = int(country_weight * len(y_log_diff_pre_stand))

        y_log_diff_pre_stand_train = y_log_diff_pre_stand[:train_weight]
        X_log_diff_pre_stand_train = X_log_diff_pre_stand[:train_weight]

        # define model
        ts_split = TimeSeriesSplit(n_splits=ts_splits)
        # print(ts_split)
        # for i, (train_index, test_index) in enumerate(ts_split.split(X_log_diff_pre_stand_train)):
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
        # lasso results
        print(f'R2 pre-stand:       {round(lasso.score(X_log_diff_pre_stand, y_log_diff_pre_stand), 3)}')
        print(f'R2 pre-stand-train: {round(lasso.score(X_log_diff_pre_stand_train, y_log_diff_pre_stand_train), 3)}')
        print(f'alpha: {round(lasso.alpha_, 3)}')

        coefs = list(lasso.coef_)
        coef_index = [i for i, val in enumerate(coefs) if val != 0]
        print(f'Parameters estimated ({len(donors_log_diff.columns[coef_index])}x): '
              f'{list(donors_log_diff.columns[coef_index])}')
        # print("\n")

        # summarize chosen configuration
        act_log_diff = flatten(y_log_diff)
        pred_log_diff = flatten(SS_treatmentfit.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))

        # act_pred_log_diff
        act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                         columns=['act', 'pred']).set_index(treatment_log_diff.index)
        act_pred_log_diff['error'] = act_pred_log_diff['pred'] - act_pred_log_diff['act']
        # other configurations, transform back to act_pred
        act_pred_log_diff_check, \
            act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                    timeframe=timeframe, pred_log_diff=pred_log_diff)

        # perform hypothesis tests
        shapiro_wilk_test(df=act_pred_log_diff, treatment_country=treatment_country, alpha=sign_level)
        t_test_result(df=act_pred_log_diff, treatment_country=treatment_country)

        # save dataframes and plots
        if show_plots or save_figs:
            plot_lasso_path(X=X_log_diff_pre_stand_train, y=y_log_diff_pre_stand_train, treatment_country=treatment_country,
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


