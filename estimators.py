################################
### import relevant packages ###
################################
import numpy as np

from definitions import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

import SparseSC

# custom functions
from plot_functions import print_lasso_path


################################
### Arco method              ###
################################
def arco(target: list, donors: list, alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):

    y = np.array(target).reshape(-1, 1)
    X = np.array(donors)

    # standardization of data
    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(y)

    # Generating the standardized values of X and y
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    # Split the data into training and testing set
    ts = (2019 - target_impl_year + 1) * timeframe_scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Quick sanity check with the shapes of Training and testing datasets
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print_lasso_path(X_train, y_train,
                     alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step,
                     lasso_iters=lasso_iters)

    # define model
    model = LassoCV(
        alphas=np.arange(0.01, 1, 0.001),
        cv=TimeSeriesSplit(n_splits=10),
        max_iter=100000,
        tol=0.00001
    )
    # fit model
    model.fit(X_train, y_train.ravel())
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)

    act = TargetVarScalerFit.inverse_transform(y)
    pred = TargetVarScalerFit.inverse_transform(model.predict(X).reshape(-1, 1))

    if show_results:
        # print(model.alpha_)
        # print(model.coef_X)
        # print(model.intercept_)
        # print(model.score)
        # print(model.get_params)

        coefs = list(model.coef_)
        coef_index = [i for i, val in enumerate(coefs) if val != 0]

        print(len(donors.columns[coef_index]))
        print(donors.columns[coef_index])

        coeffs = model.coef_
        print(coeffs[coeffs != 0])

    return model, act, pred


def sc(target: list, donors: list):
    pass
    # y = target
    # X = np.array(donors)
    # sc_model = SparseSC.fit(
    #     features=np.array(donors),
    #     target=np.array(target),
    #     treated_units=
    # )


def did():
    pass
