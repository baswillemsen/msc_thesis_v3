# PATHS
data_source_path = 'data/source/'
data_path = 'data/'
figures_path = 'figures/'
output_path = 'output/'

# PRINTING, SHOWING PLOTS
pr_results = False
show_plots = True
save_figs = True

# NON-STATIC DEFINITIONS
timeframe = 'monthly'
timeframe_scale = 12
target_country = 'iceland'
data_file = f'total_{timeframe}.csv'
target_var = f'co2_{timeframe}'

# COUNTRIES, YEARS INCLUDED
target_countries = ['switzerland', 'iceland', 'ireland', 'france', 'portugal']  # 5x
donor_countries = ['austria', 'belgium', 'bulgaria', 'croatia', 'czech republic', 'germany', 'greece', 'hungary',
                   'italy', 'lithuania', 'netherlands', 'romania', 'slovakia', 'spain']  # 14x
incl_countries = target_countries + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)

target_countries_impl_years = {'switzerland': 2008, 'iceland': 2010, 'ireland': 2010, 'france': 2014, 'portugal': 2015}
target_impl_year = target_countries_impl_years[target_country]

corr_country_names = {'republic of cyprus': 'cyprus',
                      'slovak republic': 'slovakia',
                      'czechia': 'czech republic',
                      'the netherlands': 'netherlands',
                      'holland': 'netherlands'}
