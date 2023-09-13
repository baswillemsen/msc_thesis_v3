# PATHS
data_source_path = 'data/source/'
data_path = 'data/'
figures_path_meth = 'output/figures/methodology/'
figures_path_res = 'output/figures/results/'
tables_path_meth = 'output/tables/methodology/'
tables_path_res = 'output/tables/results/'

# PRINTING, SHOWING PLOTS
show_results = True
show_plots = True
fig_size = (10, 6)

save_results = True
save_figs = True

# NON-STATIC DEFINITIONS
target_var = 'co2'
stat = 'non_stat'
sign_level = 0.05
fake_num = -99999

country_col = 'country'
year_col = 'year'
month_col = 'month'
quarter_col = 'quarter'
date_col = 'date'

model_val = ['arco', 'sc']
timeframe_val = ['m', 'q']

# COUNTRIES, YEARS INCLUDED
target_countries = ['switzerland', 'ireland', 'france', 'portugal', 'united kingdom']  # 5x
donor_countries = ['austria', 'belgium', 'bulgaria', 'croatia', 'czech republic',
                   'germany', 'greece', 'hungary', 'italy', 'lithuania', 'netherlands',
                   'romania', 'slovakia', 'spain']  # 14x
incl_countries = target_countries + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)

corr_country_names = {'republic of cyprus': 'cyprus',
                      'slovak republic': 'slovakia',
                      'czechia': 'czech republic',
                      'the netherlands': 'netherlands',
                      'holland': 'netherlands',
                      'france and monaco': 'france',
                      'spain and andorra': 'spain',
                      'italy, san marino and the holy see': 'italy',
                      'switzerland and liechtenstein': 'switzerland'
                      }