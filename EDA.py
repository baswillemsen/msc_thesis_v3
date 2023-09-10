################################
### import relevant packages ###
################################
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

from sklearn.preprocessing import StandardScaler

from helper_functions import read_data, get_impl_year
from definitions import data_path, show_plots, data_source_path, figures_path, trans, \
    target_countries, fig_size, show_plots, save_figs, show_results
from plot_functions import plot_corr

figures_path_cor = f'{figures_path}methodology/'


def descriptive_stats(df: object, var_name: str):
    if show_results:
        # describe df
        print(df.info())
        print(df.describe())
        print("\n")

        print(f"# missing: {sum(df[var_name].isna())}")
        if var_name != 'brent':
            print(f"# countries: {len(df['country'].unique())}")
            print(f"countries: {df['country'].unique()}")
            print("\n")

            print(df[df[var_name].isna()].groupby('country').count())
            print(df[df[var_name].isna()].groupby('country').max() + 1)
            print("\n")

            print(f"within-country std: \n{df.groupby('country').std().mean()[var_name]}")


def co2_target_countries(df: object):
    print(get_impl_year)

    for country in target_countries:
        plt.figure(figsize=fig_size)
        df_target = df[df['country'] == country].set_index('date')['co2']
        df_target.plot(figsize=fig_size)
        plt.title(country.upper())
        plt.xlabel('date')
        plt.ylabel('GHG (metric tons CO2e)')
        plt.grid()
        plt.tight_layout()
        plt.axvline(get_impl_year(country) / 12, color='black')
        if save_figs:
            plt.savefig(f"{figures_path_cor}co2_{country}.png")
        if show_plots:
            plt.show()


def all_series(df: object, period: str):
    if period == 'm':
        timescale = 12
    elif period == 'q':
        timescale = 4

    for series in trans.keys():

        df_pivot = df.pivot(index='date', columns='country', values=series)
        df_scale = df_pivot

        plt.figure(figsize=fig_size)
        plt.plot(df_pivot.index, df_scale, label=df_pivot.columns)
        plt.title(series.upper())
        plt.xticks([df_pivot.index[timescale * i] for i in range(int(len(df_pivot)/timescale))], rotation='vertical')
        plt.xlabel('dae')
        plt.ylabel(f"{series}")
        if series == 'pop':
            plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        if save_figs:
            plt.savefig(f"{figures_path}methodology/{series}.png")
        if show_plots:
            plt.show()


def all_series_stand(df: object, period: str):
    if period == 'm':
        timescale = 12
    elif period == 'q':
        timescale = 4

    scaler = StandardScaler()
    for series in trans.keys():

        df_pivot = df.pivot(index='date', columns='country', values=series)
        df_scale = scaler.fit_transform(df_pivot)

        plt.figure(figsize=fig_size)
        plt.plot(df_pivot.index, df_scale, label=df_pivot.columns)

        plt.title(series.upper())
        plt.xticks([df_pivot.index[timescale * i] for i in range(int(len(df_pivot) / timescale))], rotation='vertical')
        plt.xlabel('date')
        plt.ylabel(f"{series} (standardized)")
        if series == 'pop':
            plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        if save_figs:
            plt.savefig(f"{figures_path}methodology/{series}_stand.png")
        if show_plots:
            plt.show()


def corr_matrix(df: object, target_country: str):
    df_cor = df.copy()
    df_cor = df_cor[df_cor['country'] == target_country]
    df_cor = df_cor.drop(['country', 'year'], axis=1)
    cor_matrix = df_cor.corr()
    if save_figs:
        plt.savefig(f"{figures_path_cor}corr_matrix.png")
    if show_plots:
        plot_corr(matrix=cor_matrix)


def eda(period: str):
    data_file = f'total_{period}'
    df = read_data(source_path=data_path, file_name=data_file)
    df_stat = read_data(source_path=data_path, file_name=data_file)

    target_country = 'france'
    var_name = 'co2'

    descriptive_stats(df=df, var_name=var_name)
    all_series(df=df, period=period)
    all_series(df=df, period=period)
    corr_matrix(df=df, target_country=target_country)


if __name__ == "__main__":
    eda(period=sys.argv[1])
