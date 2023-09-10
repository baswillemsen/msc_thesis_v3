################################
### import relevant packages ###
################################
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

from sklearn.preprocessing import StandardScaler

from definitions import data_path, figures_path, target_countries, fig_size, show_plots, save_figs, show_results, \
    target_var, country_col, year_col, date_col
from helper_functions import read_data, get_impl_date, get_trans, get_timescale
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
            print(f"# countries: {len(df[country_col].unique())}")
            print(f"countries: {df[country_col].unique()}")
            print("\n")

            print(df[df[var_name].isna()].groupby(country_col).count())
            print(df[df[var_name].isna()].groupby(country_col).max() + 1)
            print("\n")

            print(f"within-country std: \n{df.groupby(country_col).std().mean()[var_name]}")


def co2_target_countries(df: object):

    for country in target_countries:
        plt.figure(figsize=fig_size)
        df_target = df[df[country_col] == country].set_index(date_col)[target_var]
        df_target.plot(figsize=fig_size)
        plt.title(country.upper())
        plt.xlabel('date')
        plt.ylabel('CO2 emissions (metric tons)')
        plt.grid()
        plt.tight_layout()
        plt.axvline(get_impl_date(country), color='black')
        if save_figs:
            plt.savefig(f"{figures_path_cor}co2_{country}.png")
        if show_plots:
            plt.show()


def all_series(df: object, timeframe: str):
    timescale = get_timescale(timeframe=timeframe)
    trans = get_trans(timeframe=timeframe)

    for series in trans.keys():

        df_pivot = df.pivot(index=date_col, columns=country_col, values=series)
        df_scale = df_pivot

        plt.figure(figsize=fig_size)
        plt.plot(df_pivot.index, df_scale, label=df_pivot.columns)
        plt.title(series.upper())
        plt.xticks([df_pivot.index[timescale * i] for i in range(int(len(df_pivot)/timescale))], rotation='vertical')
        plt.xlabel('date')
        plt.ylabel(f"{series}")
        if series == 'pop':
            plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        if save_figs:
            plt.savefig(f"{figures_path}methodology/{series}.png")
        if show_plots:
            plt.show()


def all_series_stand(df: object, timeframe: str):
    timescale = get_timescale(timeframe=timeframe)
    trans = get_trans(timeframe=timeframe)

    scaler = StandardScaler()
    for series in trans.keys():

        df_pivot = df.pivot(index=date_col, columns=country_col, values=series)
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
    df_cor = df_cor[df_cor[country_col] == target_country]
    df_cor = df_cor[get_trans()]
    cor_matrix = df_cor.corr()
    if save_figs:
        plt.savefig(f"{figures_path_cor}corr_matrix.png")
    if show_plots:
        plot_corr(matrix=cor_matrix)


def eda(timeframe: str):
    data_file = f'total_{timeframe}'
    df = read_data(source_path=data_path, file_name=data_file)
    df_stat = read_data(source_path=data_path, file_name=data_file)

    target_country = 'france'
    var_name = 'co2'

    descriptive_stats(df=df, var_name=var_name)
    co2_target_countries(df=df)
    all_series(df=df, timeframe=timeframe)
    all_series_stand(df=df, timeframe=timeframe)
    corr_matrix(df=df, target_country=target_country)


if __name__ == "__main__":
    eda(timeframe=sys.argv[1])
