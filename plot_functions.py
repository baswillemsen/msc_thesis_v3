################################
### import relevant packages ###
################################
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import Lasso

from definitions import fig_size, show_plots, save_figs, treatment_countries, country_name_formal
from util_general import get_impl_date, get_fig_path, get_formal_title, get_model_color


# plot intermediate series as transformed by preprocess
def plot_series(i: int, series: object, timeframe: str, treatment_country: str, var_name: str):
    figures_path_data = get_fig_path(timeframe=timeframe, folder='data', country=treatment_country)

    plt.figure(i)
    series.plot(figsize=fig_size, color=get_model_color(type='act'))
    if treatment_country in treatment_countries:
        plt.axvline(x=list(series.index).index(get_impl_date(treatment_country=treatment_country, input='dt')),
                    color=get_model_color('impl'))

    plt.title(f'{treatment_country}')
    plt.xlabel('Date')
    plt.ylabel(f'{var_name}')
    if save_figs:
        plt.savefig(f'{figures_path_data}/{var_name}.png', bbox_inches='tight')


# print lasso path for given alphas and LASSO solution
def plot_lasso_path(X: list, y: list, treatment_country: str, model: str, timeframe: str,
                    alpha_min: float, alpha_max: float, alpha_step: float, lasso_iters: int, alpha_cv: float):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)
    var_name = f'{model}_{treatment_country}_{timeframe}_lasso_path'

    alphas = np.arange(alpha_min, alpha_max, alpha_step)
    lasso = Lasso(max_iter=lasso_iters)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(alphas, coefs)
    ax.axvline(x=alpha_cv, color=get_model_color('impl'))

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} LASSO Path')
    else:
        ax.set_title(f'LASSO Path')
    ax.set_xscale('log')
    ax.axis('tight')
    ax.set_xlabel('alpha')
    ax.set_ylabel('weights')

    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot predictions and actual series of treatment country co2
def plot_predictions(df: object, treatment_country: str, model: str, timeframe: str, var_name: str):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(df['act'], label='actual', color=get_model_color(type='act'))
    ax.plot(df['pred'], label='predicted', color=get_model_color(type=model))
    ax.axvline(x=get_impl_date(treatment_country, input='dt'), color=get_model_color(type='impl'))

    if var_name == 'act_pred':
        df['year'] = df.index.year
        df_group = df.groupby('year').mean()
        df_group.index = pd.to_datetime(df_group.index, format='%Y')
        ax.plot(df_group['act'], linestyle='--', color=get_model_color(type='act'))
        ax.plot(df_group['pred'], linestyle='--', color=get_model_color(type=model))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} {get_formal_title(var_name=var_name)} CO2 series')
    else:
        ax.set_title(f'{get_formal_title(var_name=var_name)} CO2 series')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{get_formal_title(var_name=var_name)} CO2 (tons)')
    ax.legend(loc='best')

    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot prediction errors of treatment country co2
def plot_diff(df: object, treatment_country: str, model: str, timeframe: str, var_name: str):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(df['error'], label='error', color=get_model_color(type='error'))
    ax.axvline(x=get_impl_date(treatment_country, input='dt'), color=get_model_color('impl'))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} {get_formal_title(var_name=var_name)} '
                     f'CO2 series prediction error')
    else:
        ax.set_title(f'{get_formal_title(var_name=var_name)} CO2 series prediction error')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'')
    ax.legend(loc='best')

    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}_error.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot cumulative summation of treatment country co2
def plot_cumsum(df: object, treatment_country: str, model: str, timeframe: str, var_name: str):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(df['act'].cumsum(), label='actual', color=get_model_color(type='act'))
    ax.plot(df['pred'].cumsum(), label='predicted', color=get_model_color(type=model))
    ax.axvline(x=get_impl_date(treatment_country, input='dt'), color=get_model_color('impl'))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} {get_formal_title(var_name=var_name)} CO2 series (cumsum)')
    else:
        ax.set_title(f'{get_formal_title(var_name=var_name)} CO2 series (cumsum)')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{get_formal_title(var_name=var_name)} CO2 (cumsum)')
    ax.legend(loc='best')
    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}_cumsum.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot cumulative summation from the implementation date of treatment country co2
def plot_cumsum_impl(df: object, treatment_country: str, model: str, timeframe: str, var_name: str):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    df = df[df.index >= get_impl_date(treatment_country)]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(df['act'].cumsum(), label='actual', color=get_model_color(type='act'))
    ax.plot(df['pred'].cumsum(), label='predicted', color=get_model_color(type=model))
    ax.axvline(x=get_impl_date(treatment_country, input='dt'), color=get_model_color('impl'))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} {get_formal_title(var_name=var_name)} CO2 series (cumsum)')
    else:
        ax.set_title(f'{get_formal_title(var_name=var_name)} CO2 series (cumsum)')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{get_formal_title(var_name=var_name)} CO2 (cumsum)')
    ax.legend(loc='best')
    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}_cumsum_impl.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot qq plot of the prediction errors of treatment country co2
def plot_qq(df: object, treatment_country: str, model: str, timeframe: str, var_name: str):
    figures_path_res = get_fig_path(timeframe=timeframe, folder='results', country=treatment_country, model=model)

    df.index = pd.to_datetime(df.index)
    df = df[df.index < get_impl_date(treatment_country)]

    fig, ax = plt.subplots(figsize=fig_size)
    # print(df['error'].dropna())
    sm.qqplot(df['error'].dropna(), line='45', fit=True, ax=ax)

    if treatment_country in country_name_formal.keys():
        ax.set_title(f'{country_name_formal[treatment_country]} {get_formal_title(var_name=var_name)} '
                     f'prediction error QQ-plot')
    else:
        ax.set_title(f'{get_formal_title(var_name=var_name)} prediction error QQ-plot')

    if save_figs:
        plt.savefig(f'{figures_path_res}/{var_name}_qq.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()


# plot correlation matrix of all included variables
def plot_corr(matrix: object, timeframe: str):
    figures_path_meth = get_fig_path(timeframe=timeframe, folder='methodology')
    var_name = 'corr_matrix'

    plt.figure(figsize=fig_size)
    sns_plot = sns.heatmap(matrix, annot=True)
    plt.title('Correlation matrix')

    if save_figs:
        sns_plot.figure.savefig(f'{figures_path_meth}/{var_name}.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
