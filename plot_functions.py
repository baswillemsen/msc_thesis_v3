################################
### import relevant packages ###
################################
import numpy as np
import matplotlib.pyplot as plt
from definitions import *

from sklearn.linear_model import Lasso


# print lasso path for given alphas and LASSO solution
def print_lasso_path(X: list, y: list, alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):
    alphas = np.arange(alpha_min, alpha_max, alpha_step)
    lasso = Lasso(max_iter=lasso_iters)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    if save_figs:
        plt.savefig(f'{figures_path}lasso_path.png')
    if show_plots:
        plt.show()


def plot_predictions(act: list, pred: list):
    plt.figure(figsize=(15, 6))
    plt.plot(act, label='actual')
    plt.plot(pred, label='predicted')
    # plt.axvline(x=target_impl_year/timeframe_scale)
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path}act_vs_pred.png')
    if show_plots:
        plt.show()


def plot_diff(act: list, pred: list):
    diff = act - pred
    print(sum(diff[:round(target_impl_year / timeframe_scale)]))
    print(sum(diff[round(target_impl_year / timeframe_scale):]))

    # act_cum = np.cumsum(act)
    # pred_cum = np.cumsum(pred)
    act_cum = np.cumsum(act[round(target_impl_year / timeframe_scale):])
    pred_cum = np.cumsum(pred[round(target_impl_year / timeframe_scale):])

    plt.figure(figsize=(15, 6))
    plt.plot(act_cum, label='act_cum')
    plt.plot(pred_cum, label='pred_cum')
    # plt.axvline(x = round(target_impl_year/12))
    plt.axvline(x=0)
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path}act_pred_diff.png')
    if show_plots:
        plt.show()