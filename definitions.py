# PATHS
data_source_path = 'data/source'
data_path = 'data'
output_path = 'output'
all_paths = [data_source_path, data_path, output_path]

# PRINTING, SHOWING PLOTS
show_output = True
show_plots = False
fig_size = (10, 6)

save_output = True
save_figs = True

# NON-STATIC DEFINITIONS
target_var = 'co2'
stat = 'stat'
sign_level = 0.10
fake_num = -99999

country_col = 'country'
year_col = 'year'
month_col = 'month'
quarter_col = 'quarter'
date_col = 'date'

model_val = ['ols', 'lasso', 'nn', 'rf', 'sc', 'did']
stat_val = ['stat', 'non_stat']
timeframe_val = ['m', 'q']
folder_val = ['data', 'methodology', 'results']

interpolation_val = ['median', 'linear']
agg_val = ['sum', 'mean']

# COUNTRIES, YEARS INCLUDED
treatment_countries = ['switzerland',
                       'ireland',
                       'united_kingdom',
                       'france',
                       'portugal'
                       ]  # 5x
donor_countries_all = ['austria',
                       'belgium',
                       'bulgaria',
                       'croatia',
                       'czech_republic',
                       'germany',
                       'greece',
                       'hungary',
                       'italy',
                       'lithuania',
                       'netherlands',
                       'romania',
                       'slovakia',
                       'spain',
                       ]  # 14x
incl_countries = treatment_countries + donor_countries_all
incl_countries = list(set(incl_countries))
incl_countries.sort()
incl_years = range(2000, 2020)

corr_country_names = {'republic of cyprus': 'cyprus',
                      'slovak republic': 'slovakia',
                      'czechia': 'czech_republic',
                      'czech republic': 'czech_republic',
                      'the netherlands': 'netherlands',
                      'holland': 'netherlands',
                      'france and monaco': 'france',
                      'spain and andorra': 'spain',
                      'italy, san marino and the holy see': 'italy',
                      'switzerland and liechtenstein': 'switzerland',
                      'united kingdom': 'united_kingdom'
                      }

country_name_formal = {'switzerland': 'Switzerland',
                       'ireland': 'Ireland',
                       'united_kingdom': 'United Kingdom',
                       'france': 'France',
                       'portugal': 'Portugal'
                       }

var_name_formal = {'co2': 'CO2 Emissions',
                   'co2_cap': 'CO2 Emissions per capita',
                   'gdp': 'GDP',
                   'gdp_cap': 'GDP per capita',
                   'ind_prod': 'Industrial Production',
                   'pop': 'Population',
                   'infl': 'Inflation',
                   'brent': 'Brent Oil'
                   }

