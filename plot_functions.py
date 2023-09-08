################################
### import relevant packages ###
################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso

from definitions import data_path, figures_path, data_file, fig_size, show_plots, save_figs, stat
from helper_functions import read_data, first_value, get_impl_year

figures_path_cor = f'{figures_path}results/{stat}/'


def plot_series(target_country: str):
    df = read_data(source_path=data_path, file_name=data_file)
    df_target = df[df['country'] == target_country].set_index('year')
    print(df_target.columns.drop('country'))
    for series in df_target.columns.drop('country').sort_values():
        df_target[series].plot(figsize=fig_size)
        plt.title(series)
        if save_figs:
            plt.savefig(f'{figures_path_cor}{target_country}/{target_country}_test.png',
                        bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()


# print lasso path for given alphas and LASSO solution
def plot_lasso_path(X: list, y: list, target_country: str, year_start: int,
                    alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int):
    alphas = np.arange(alpha_min, alpha_max, alpha_step)
    lasso = Lasso(max_iter=lasso_iters)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    if save_figs:
        plt.savefig(f'{figures_path_cor}{target_country}/{year_start}_{target_country}_lasso_path.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


# def plot_predictions(act: list, pred: list):
#     plt.figure(figsize=(15, 6))
#     plt.plot(act, label='actual')
#     plt.plot(pred, label='predicted')
#     # plt.axvline(x=target_impl_year/timeframe_scale)
#     plt.legend()
#     if save_figs:
#         plt.savefig(f'{figures_path}act_vs_pred.png')
#     if show_plots:
#         plt.show()


def plot_predictions(act_pred: object, target_country: str, year_start: int):
    orig_value = first_value(target_country=target_country)
    # act = act_pred['act']
    # pred = act_pred['pred']
    act = np.exp(act_pred['act'].cumsum())*orig_value
    pred = np.exp(act_pred['pred'].cumsum())*orig_value

    plt.figure(figsize=fig_size)
    plt.plot(act, label='actual')
    plt.plot(pred, label='predicted')
    plt.axvline(x=get_impl_year(target_country), c='black')
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path_cor}{target_country}/{year_start}_{target_country}_act_pred.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


# def plot_diff(act: list, pred: list):
#     diff = act - pred
#     print(sum(diff[:round(target_impl_year / timeframe_scale)]))
#     print(sum(diff[round(target_impl_year / timeframe_scale):]))
#
#     # act_cum = np.cumsum(act)
#     # pred_cum = np.cumsum(pred)
#     act_cum = np.cumsum(act[round(target_impl_year / timeframe_scale):])
#     pred_cum = np.cumsum(pred[round(target_impl_year / timeframe_scale):])
#
#     plt.figure(figsize=(15, 6))
#     plt.plot(act_cum, label='act_cum')
#     plt.plot(pred_cum, label='pred_cum')
#     # plt.axvline(x = round(target_impl_year/12))
#     plt.axvline(x=0)
#     plt.legend()
#     if save_figs:
#         plt.savefig(f'{figures_path}act_pred_diff.png')
#     if show_plots:
#         plt.show()


def plot_diff(act_pred: object, target_country: str, year_start: int):
    orig_value = first_value(target_country=target_country)
    diff = act_pred['act'] - act_pred['pred']
    # diff = np.cumsum(diff)
    # diff = np.exp(diff.cumsum())*orig_value

    plt.figure(figsize=fig_size)
    plt.plot(diff, label='diff')
    plt.axvline(x=get_impl_year(target_country), c='black')
    plt.tight_layout()
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path_cor}{target_country}/{year_start}_{target_country}_act_pred_diff.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


def plot_corr(matrix: object):
    plt.figure(figsize=fig_size)
    plt.tight_layout()
    sns.heatmap(matrix, annot=True)
    plt.show()


if __name__ == "__main__":
    plot_series(target_country='france')