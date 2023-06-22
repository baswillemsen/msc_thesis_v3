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
data_file = 'total_monthly.csv'
target_country = 'iceland'
target_var = 'co2_monthly'

# COUNTRIES, YEARS INCLUDED
target_countries = ['switzerland', 'iceland', 'ireland', 'france', 'portugal']
donor_countries = ['austria', 'belgium', 'bulgaria', 'croatia', 'czech Republic', 'germany', 'greece', 'hungary',
                   'italy', 'lithuania', 'netherlands', 'romania', 'slovakia', 'spain']
incl_countries = list(target_country) + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)

target_countries_impl_years = {'switzerland': 2008, 'iceland': 2010, 'ireland': 2010, 'france': 2014, 'portugal': 2015}
target_impl_year = target_countries_impl_years[target_country]

corr_country_names = {'republic of cyprus': 'cyprus',
                      'slovak republic': 'slovakia',
                      'czechia': 'czech Republic',
                      'the netherlands': 'netherlands',
                      'holland': 'netherlands'}
