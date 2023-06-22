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
target_country = 'Iceland'
target_var = 'co2_monthly'

# COUNTRIES, YEARS INCLUDED
target_countries = ['Switzerland', 'Iceland', 'Ireland', 'France', 'Portugal']
donor_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czech Republic', 'Germany', 'Greece', 'Hungary',
                   'Italy', 'Lithuania', 'Netherlands', 'Romania', 'Slovakia', 'Spain']
incl_countries = list(target_country) + donor_countries
incl_countries.sort()
incl_years = range(2000, 2020)

target_countries_impl_years = {'Switzerland': 2008, 'Iceland': 2010, 'Ireland': 2010, 'France': 2014, 'Portugal': 2015}
target_impl_year = target_countries_impl_years[target_country]
