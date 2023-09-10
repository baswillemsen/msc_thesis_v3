# PATHS
data_source_path = 'data/source/'
data_path = 'data/'
figures_path = 'output/figures/'
tables_path = 'output/tables/'

# PRINTING, SHOWING PLOTS
show_results = True
show_plots = True
fig_size = (10, 6)

save_results = True
save_figs = True

# NON-STATIC DEFINITIONS
stat = 'stat'
target_var = 'co2'

sign_level = 0.10
fake_num = -99999

country_col = 'country'
year_col = 'year'
month_col = 'month'
quarter_col = 'quarter'
date_col = 'date'

# COUNTRIES, YEARS INCLUDED
target_countries = ['switzerland', 'ireland', 'france', 'portugal', 'united kingdom']  # 5x
donor_countries = ['austria', 'belgium', 'bulgaria', 'croatia', 'czech republic',
                   # 'cyprus',
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