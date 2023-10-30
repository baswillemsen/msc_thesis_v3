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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.ensemble import RandomForestRegressor

import SparseSC
import statsmodels.formula.api as smf
import statsmodels.api as sm

# custom functions
from definitions import fake_num, show_plots, sign_level, save_figs, incl_years, incl_countries, stat, save_output, \
    date_col, target_var, model_val
from util_general import flatten, get_impl_date, get_table_path, get_months_cors
from util_estimation import arco_pivot, sc_pivot, transform_back, save_dataframe, did_pivot, save_results, save_did
from plot_functions import plot_lasso_path


################################
### Arco method              ###
################################
def arco(df: object, df_stat: object, treatment_country: str, timeframe: str, ts_splits: int,
         alpha_min: float, alpha_max: float, alpha_step: float, tol: float, lasso_iters: int,
         model: str, prox: bool):
    tables_path_res = get_table_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    treatment_log_diff, donors_log_diff = arco_pivot(df=df_stat, treatment_country=treatment_country,
                                                     timeframe=timeframe, model=model, prox=prox)
    # print(f'Nr of parameters included ({len(donors_log_diff.columns)}x): {donors_log_diff.columns}')
    print(f'Nr of parameters included: {len(donors_log_diff.columns)}x')

    # check the treatment series is stationary
    if fake_num in list(treatment_log_diff):
        return None
    else:

        # for months_cor in [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]:
        months_cor = get_months_cors(model=model, timeframe=timeframe, treatment_country=treatment_country)

        y_log_diff = treatment_log_diff
        X_log_diff = donors_log_diff

        impl_date = get_impl_date(treatment_country=treatment_country)
        impl_date_index = list(treatment_log_diff.index).index(impl_date)
        split_index = impl_date_index + months_cor
        split_date = treatment_log_diff.index[split_index]

        y_log_diff_pre = treatment_log_diff[treatment_log_diff.index < split_date]
        X_log_diff_pre = donors_log_diff[donors_log_diff.index < split_date]
        print(f'Treatment implementation date (T_0):        {impl_date}')
        print(f'Treatment split date (T_0 cor):             {split_date}')
        print(f'Nr of timeframes pre-treatment (t < T_0):   {len(X_log_diff_pre)}')
        print(f'Nr of timeframes post-treatment (t >= T_0): {len(donors_log_diff) - len(X_log_diff_pre)}')
        print("\n")

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_treatmentfit_pre = SS.fit(np.array(y_log_diff_pre).reshape(-1, 1))
        X_log_diff_stand = SS.fit_transform(np.array(X_log_diff))

        # Generating the standardized values of X and y
        X_log_diff_pre_stand = pd.DataFrame(SS.fit_transform(X_log_diff_pre), index=X_log_diff_pre.index,
                                            columns=X_log_diff_pre.columns)
        y_log_diff_pre_stand = pd.DataFrame(SS.fit_transform(y_log_diff_pre), index=y_log_diff_pre.index,
                                            columns=y_log_diff_pre.columns)

        ### LASSO ======================================================================================
        if model == 'lasso':
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
            X = X_log_diff_pre_stand
            y = y_log_diff_pre_stand[target_var]
            lasso.fit(X, y)
            # lasso results
            r2_pre_log_diff_stand = lasso.score(X, y)
            lasso_alpha = lasso.alpha_
            print(f'R2 r2_pre_log_diff_stand: {r2_pre_log_diff_stand}')
            print(f'alpha: {lasso_alpha}')

            feats = {}
            for i, val in enumerate(list(lasso.coef_)):
                if val != 0:
                    feats[donors_log_diff.columns[i]] = val
            feats = dict(sorted(feats.items(), key=lambda item: item[1], reverse=True))
            n_pars = len(feats)
            pars = list(feats.keys())
            coefs = list(feats.values())

            # feats = list(lasso.coef_)
            # coefs_index = [i for i, val in enumerate(feats) if val != 0]
            # n_pars = len(coefs_index)
            # pars = list(donors_log_diff.columns[coefs_index])
            # coefs = [round(coef, 3) for coef in feats if coef != 0]
            print(f'Parameters estimated ({n_pars}x): {pars}')
            print(f'Coefficients estimated ({n_pars}x): {coefs}')
            print("\n")

            ind = list(range(1, n_pars + 1))
            df_results = pd.DataFrame(list(zip(pars, coefs)), columns=['Regressor', 'Coefficient'])
            df_results = df_results.sort_values('Coefficient', ascending=False)
            df_results['Index'] = ind
            df_results = df_results[['Index', 'Regressor', 'Coefficient']]
            df_results.to_csv(f'{tables_path_res}/{model}_{treatment_country}_{timeframe}_lasso_pars.csv', index=False)

            plot_lasso_path(X=X_log_diff_pre_stand, y=y_log_diff_pre_stand, treatment_country=treatment_country,
                            alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step, lasso_iters=lasso_iters,
                            model=model, timeframe=timeframe, alpha_cv=lasso_alpha)

            # summarize chosen configuration
            act_log_diff = flatten(np.array(y_log_diff).reshape(-1, 1))
            pred_log_diff = flatten(
                SS_treatmentfit_pre.inverse_transform(lasso.predict(X_log_diff_stand).reshape(-1, 1)))
            act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                             columns=['act', 'pred']).set_index(treatment_log_diff.index)
            act_pred_log_diff['error'] = act_pred_log_diff['act'] - act_pred_log_diff['pred']

        ### RANDOM FOREST ===================================================================================
        elif model == 'rf':
            rf = RandomForestRegressor(n_estimators=10,
                                       max_depth=10,
                                       min_weight_fraction_leaf=0.25,
                                       bootstrap=True,
                                       random_state=0,
                                       criterion='squared_error'
                                       )

            y = np.array(y_log_diff_pre_stand).ravel()
            X = X_log_diff_pre_stand
            rf.fit(X, y)

            act_log_diff = flatten(np.array(y_log_diff).reshape(-1, 1))
            pred_log_diff = flatten(SS_treatmentfit_pre.inverse_transform(
                np.array(rf.predict(X_log_diff_stand)).reshape(-1, 1)))
            act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)), columns=['act', 'pred']).set_index(
                treatment_log_diff.index)
            act_pred_log_diff['error'] = act_pred_log_diff['act'] - act_pred_log_diff['pred']

            lasso_alpha = None
            r2_pre_log_diff_stand = rf.score(X=X, y=y)

            feats = {}
            for i, val in enumerate(rf.feature_importances_):
                if val != 0:
                    feats[donors_log_diff.columns[i]] = val
            feats = dict(sorted(feats.items(), key=lambda item: item[1], reverse=True))
            n_pars = len(feats)
            pars = list(feats.keys())
            coefs = list(feats.values())

        ### OLS ======================================================================================
        elif model == 'ols':
            # Perform stepwise regression
            # n_pars = get_ols_pars(treatment_country=treatment_country)
            ts_split = TimeSeriesSplit(n_splits=ts_splits)
            sfs = SequentialFeatureSelector(estimator=LinearRegression(),
                                            n_features_to_select='auto',
                                            tol=1e-3,
                                            direction='forward',
                                            scoring='neg_mean_squared_error',
                                            cv=ts_split)
            selected_features = sfs.fit(X_log_diff_pre_stand, y_log_diff_pre_stand)
            pars = selected_features.get_feature_names_out()

            # Storing the fit object for later reference
            X_log_diff = X_log_diff[pars]
            X_log_diff_pre = X_log_diff[X_log_diff.index < split_date]

            SS = StandardScaler()
            SS_treatmentfit_pre = SS.fit(np.array(y_log_diff_pre).reshape(-1, 1))
            X_log_diff_stand = pd.DataFrame(SS.fit_transform(np.array(X_log_diff)), index=X_log_diff.index,
                                            columns=X_log_diff.columns)

            # Generating the standardized values of X and y
            X_log_diff_pre_stand = pd.DataFrame(SS.fit_transform(X_log_diff_pre), index=X_log_diff_pre.index,
                                                columns=X_log_diff_pre.columns)
            y_log_diff_pre_stand = pd.DataFrame(SS.fit_transform(y_log_diff_pre), index=y_log_diff_pre.index,
                                                columns=y_log_diff_pre.columns)

            y = y_log_diff_pre_stand
            X = X_log_diff_pre_stand
            model_res = sm.OLS(y, X).fit()
            print(model_res.summary())

            act_log_diff = flatten(np.array(y_log_diff).reshape(-1, 1))
            pred_log_diff = flatten(SS_treatmentfit_pre.inverse_transform(
                np.array(model_res.predict(X_log_diff_stand)).reshape(-1, 1)))
            act_pred_log_diff = pd.DataFrame(list(zip(act_log_diff, pred_log_diff)),
                                             columns=['act', 'pred']).set_index(treatment_log_diff.index)
            act_pred_log_diff['error'] = act_pred_log_diff['act'] - act_pred_log_diff['pred']

            lasso_alpha = None
            r2_pre_log_diff_stand = model_res.rsquared
            n_pars = len(pars)
            pars = pars
            coefs = list(model_res.params)

        else:
            raise ValueError(f'Input valid model argument: {model_val}')

        # transform back
        act_pred_log_diff_check, \
            act_pred_log, act_pred, = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                     timeframe=timeframe, pred_log_diff=pred_log_diff)

        n_train = len(y_log_diff_pre_stand)
        n_test = len(y_log_diff) - len(y_log_diff_pre_stand)

        # # save dataframes and plots
        if save_output or show_plots or save_figs:
            save_results(act_pred_log_diff=act_pred_log_diff, act_pred_log=act_pred_log, act_pred=act_pred,
                         model=model, stat=stat, timeframe=timeframe,
                         sign_level=sign_level, incl_countries=incl_countries, incl_years=incl_years,
                         treatment_country=treatment_country, impl_date=impl_date, impl_date_index=impl_date_index,
                         n_train=n_train, n_test=n_test, var_title=f'{model}_results',
                         # model specific
                         prox=prox, months_cor=months_cor, split_date=split_date,
                         r2_pre_log_diff_stand=r2_pre_log_diff_stand,
                         lasso_alpha=lasso_alpha, n_pars=n_pars, pars=pars, coefs=coefs)

            save_dataframe(df=act_pred_log_diff, var_title='act_pred_log_diff',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True,
                           save_cumsum=True, save_cumsum_impl=True, save_qq=True)

            save_dataframe(df=act_pred_log_diff_check, var_title='act_pred_log_diff_check',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True,
                           save_cumsum=True, save_cumsum_impl=True, save_qq=True)

            save_dataframe(df=act_pred_log, var_title='act_pred_log',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True,
                           save_cumsum=True, save_cumsum_impl=True, save_qq=True)

            save_dataframe(df=act_pred, var_title='act_pred',
                           model=model, treatment_country=treatment_country, timeframe=timeframe,
                           save_csv=True, save_predictions=True, save_diff=True,
                           save_cumsum=True, save_cumsum_impl=True, save_qq=True)

        return act_pred_log_diff


################################
### Synthetic Control method ###
################################
def sc(df: object, df_stat: object, treatment_country: str, timeframe: str, model: str, prox: bool):
    # pivot treatment and donors
    impl_date = get_impl_date(treatment_country=treatment_country)
    impl_date_index = list(df[date_col]).index(impl_date)

    # for months_cor in [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]:
    months_cor = get_months_cors(model=model, timeframe=timeframe, treatment_country=treatment_country)

    split_index = impl_date_index + months_cor
    split_date = df[date_col][split_index]

    df_pivot, pre_treat, post_treat, treat_unit = sc_pivot(df=df_stat, treatment_country=treatment_country,
                                                           timeframe=timeframe, model=model, impl_date=split_date,
                                                           prox=prox)

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
        verbose=0
    )

    # Predict the series, make act_pred dataframe
    SS_treatmentfit = SS.fit(np.array(df_pivot).reshape(-1, 1))
    act_pred_log_diff = df_pivot[treatment_country].to_frame()
    act_pred_log_diff.rename(columns={treatment_country: 'act'}, inplace=True)
    pred_log_diff = flatten(SS_treatmentfit.inverse_transform(sc.predict(df_pivot_stand.T.values)[0].reshape(-1, 1)))
    act_pred_log_diff['pred'] = pred_log_diff
    act_pred_log_diff['error'] = act_pred_log_diff['act'] - act_pred_log_diff['pred']

    # transform back
    act_pred_log_diff_check, \
        act_pred_log, act_pred = transform_back(df=df, df_stat=df_stat, treatment_country=treatment_country,
                                                timeframe=timeframe, pred_log_diff=pred_log_diff)

    r2_pre_log_diff_stand = sc.score_R2
    n_train = len(pre_treat.columns)
    n_test = len(post_treat.columns)

    # # save dataframes and plots
    if save_output or show_plots or save_figs:
        save_results(act_pred_log_diff=act_pred_log_diff, act_pred_log=act_pred_log, act_pred=act_pred,
                     model=model, stat=stat, timeframe=timeframe,
                     sign_level=sign_level, incl_countries=incl_countries, incl_years=incl_years,
                     treatment_country=treatment_country, impl_date=impl_date, impl_date_index=impl_date_index,
                     n_train=n_train, n_test=n_test, var_title=f'{model}_results',
                     # model specific
                     prox=prox, months_cor=months_cor, split_date=split_date, r2_pre_log_diff_stand=r2_pre_log_diff_stand,
                     lasso_alpha=None, n_pars=None, pars=None, coefs=None)

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

    return act_pred_log_diff


################################
### Diff-in-Diff method      ###
################################
def did(df: object, treatment_country: str, timeframe: str, model: str, prox: bool, x_years: int):

    impl_date = get_impl_date(treatment_country=treatment_country)
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

    # # linear regression
    # lr = LinearRegression()
    # X = df_sel[['treatment_dummy', 'post_dummy', 'treatment_post_dummy']]
    # y = df_sel[target_var]
    # lr.fit(X, y)
    # print(lr.coef_)

    # Extended OLS
    ols = smf.ols('co2 ~ treatment_dummy + post_dummy + treatment_post_dummy', data=df_sel).fit()
    # diff_in_diff = ols.params['treatment_post_dummy']
    print(ols.summary())

    # save dataframes and plots
    if save_output or show_plots or save_figs:
        save_did(model=model, stat=stat, timeframe=timeframe, sign_level=sign_level, incl_countries=incl_countries,
                 incl_years=incl_years, treatment_country=treatment_country, impl_date=impl_date, var_title=f'{model}_results',
                 # model specific
                 x_years=x_years, prox=prox, diff_in_diff=diff_in_diff, ols=ols)

    return diff_in_diff


