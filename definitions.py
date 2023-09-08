# PATHS
data_source_path = 'data/source/'
data_path = 'data/'
figures_path = 'figures/'
output_path = 'output/'

# PRINTING, SHOWING PLOTS
show_results = False
show_plots = False
fig_size = (10, 6)

save_results = True
save_figs = True

# TIMEFRAME DEFINITIONS
timeframe = 'quarterly'
timeframe_scale = 4

# NON-STATIC DEFINITIONS
stat = 'stat'
sign_level = 0.10
fake_num = -99999
target_country = 'france'
data_file = f'total_{timeframe}.csv'
target_var = f'co2_{timeframe}'

country_col = 'country'
year_col = 'year'
month_col = 'month'
quarter_col = 'quarter'
date_col = 'date'

# COUNTRIES, YEARS INCLUDED
target_countries = ['switzerland', 'ireland', 'france', 'portugal', 'united kingdom']  # 5x
donor_countries = ['austria', 'belgium', 'bulgaria', 'croatia', 'czech republic', 'cyprus', 'germany', 'greece',
                   'hungary', 'italy', 'lithuania', 'netherlands', 'romania', 'slovakia', 'spain']  # 15x
incl_countries = target_countries + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)

target_countries_impl_years = {'switzerland': 2008, 'iceland': 2010, 'ireland': 2010, 'france': 2014, 'portugal': 2015}
target_impl_year = target_countries_impl_years[target_country]

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