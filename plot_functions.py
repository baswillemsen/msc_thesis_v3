################################
### import relevant packages ###
################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import Lasso

from definitions import figures_path_meth, figures_path_res, fig_size, show_plots, save_figs, date_col, target_countries
from helper_functions import read_data, get_impl_date, get_data_path


def plot_total(target_country: str, timeframe: str):
    df = read_data(source_path=get_data_path(timeframe=timeframe), file_name=f'total_{timeframe}')
    df_target = df[df['country'] == target_country].set_index(date_col)
    print(df_target.columns.drop('country'))
    for series in df_target.columns.drop('country').sort_values():
        df_target[series].plot(figsize=fig_size)
        plt.title(series)
        if save_figs:
            plt.savefig(f'{figures_path_meth}{target_country}/{target_country}_test.png',
                        bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()


def plot_series(i: int, series: object, country_path: str, target_country: str, var_name: str):
    plt.figure(i)
    series.plot(figsize=fig_size)
    if target_country in target_countries:
        plt.axvline(x=list(series.index).index(get_impl_date(target_country=target_country, input='dt')), c='black')
    plt.title(f'{target_country}')
    plt.xlabel('Date')
    plt.ylabel(f'{var_name}')
    if save_figs:
        plt.savefig(f'{country_path}{var_name}.png', bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    plt.cla()


# print lasso path for given alphas and LASSO solution
def plot_lasso_path(X: list, y: list, target_country: str, model: str,
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
        plt.savefig(f'{figures_path_res}{target_country}/{model}_{target_country}_lasso_path.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


def plot_predictions(df: object, target_country: str, log: str, model: str):
    if log == 'log':
        act = df['act']
        pred = df['pred']
    elif log == 'exp':
        act = np.exp(df['act'])
        pred = np.exp(df['pred'])

    plt.figure(figsize=fig_size)
    plt.plot(act, label='actual')
    plt.plot(pred, label='predicted')
    plt.axvline(x=list(act.index).index(get_impl_date(target_country)), c='black')
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path_res}{target_country}/{model}_{target_country}_act_pred_{log}.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


def plot_diff(df: object, target_country: str, model: str):
    diff = df['error']

    plt.figure(figsize=fig_size)
    plt.plot(diff, label='diff')
    plt.axvline(x=list(diff.index).index(get_impl_date(target_country)), c='black')
    plt.tight_layout()
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path_res}{target_country}/{model}_{target_country}_act_pred_log_diff.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


def plot_cumsum(df: object, target_country: str, model: str):
    act = df['act'].cumsum()
    pred = df['pred'].cumsum()

    plt.figure(figsize=fig_size)
    plt.plot(act, label='actual')
    plt.plot(pred, label='predicted')
    plt.axvline(x=list(act.index).index(get_impl_date(target_country)), c='black')
    plt.legend()
    if save_figs:
        plt.savefig(f'{figures_path_res}{target_country}/{model}_{target_country}_cumsum.png',
                    bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()


def plot_corr(matrix: object):
    plt.figure(figsize=fig_size)
    plt.tight_layout()
    sns.heatmap(matrix, annot=True)
    if show_plots:
        plt.show()


if __name__ == "__main__":
    plot_total(target_country='france', timeframe='m')
