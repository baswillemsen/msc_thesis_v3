import os


def paths():
    data_path = 'data/'
    figures_path = 'figures/'
    output_path = 'output/'
    for path in [data_path, figures_path, output_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    return data_path, figures_path, output_path


def verbatim():
    pr_results = False
    show_plots = True
    save_figs = True
    return pr_results, save_figs, show_plots
