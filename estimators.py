################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC

# custom functions
from definitions import show_results, donor_countries, fake_num, country_col, date_col
from helper_functions import flatten, arco_pivot, sc_pivot, get_impl_date
from plot_functions import plot_lasso_path
from statistical_tests import shapiro_wilk_test


################################
### Arco method              ###
################################
def arco(df: object, target_country: str, alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):
    # pivot target and donors
    target, donors = arco_pivot(df=df, target_country=target_country)

    if fake_num in list(target):
        return None, None
    else:
        y = np.array(target).reshape(-1, 1)
        X = np.array(donors)

        # Storing the fit object for later reference
        SS = StandardScaler()
        SS_targetfit = SS.fit(y)

        # Generating the standardized values of X and y
        X = SS.fit_transform(X)
        y = SS.fit_transform(y)

        # Split the data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

        # Quick sanity check with the shapes of Training and testing datasets
        # print(X_train)
        # print(y_train)
        # print(X_test.shape)
        # print(y_test.shape)

        plot_lasso_path(X=X_train, y=y_train, target_country=target_country,
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
        model.fit(X_train, y_train.ravel())
        # summarize chosen configuration
        act = flatten(SS_targetfit.inverse_transform(y))
        pred = flatten(SS_targetfit.inverse_transform(model.predict(X).reshape(-1, 1)))

        act_pred = pd.DataFrame(list(zip(act, pred)), columns=['act', 'pred']).set_index(target.index)
        shapiro_wilk_test(df=act_pred, target_country=target_country, alpha=0.05)

        if show_results:
            print('alpha: %f' % model.alpha_)
            print(model.coef_)
            print(model.intercept_)
            print(model.score)
            print(model.get_params)

            coefs = list(model.coef_)
            coef_index = [i for i, val in enumerate(coefs) if val != 0]

            print(len(donors.columns[coef_index]))
            print(donors.columns[coef_index])

            coeffs = model.coef_
            print(coeffs[coeffs != 0])

        return model, act_pred


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

    return model, act_pred


# def did():
#     pass