# MSc. Econometrics Thesis Bas Willemsen

This repository contains all code used in the MSc. Thesis _"Artificial Counterfactual analysis on
carbon taxation in Europe"_. This includes code to load, transform, estimate the data and models, 
and save the output. 

In order to run the code, clone the repository using the following code in a Command Prompt (and Git installed):
```
git clone https://github.com/baswillemsen/msc_thesis_v3.git
```

The runnable code can be found below in the sectioned areas. In order to run the code, open a terminal, 
navigate to the path _.../msc_thesis_v3_, and run the code with the necessary arguments.

## Explanation of code

### definitions.py
This file contains some static variables used throughout the code, such as data paths, timeframes, 
countries included. Here also some definitions can be found to toggle on and off the possibility to 
print and save output, and to print and save plots.
```python
show_output = True
show_plots = False
fig_size = (10, 6)
save_output = True
save_figs = False
```

### util_{}.py
The files _util\_general_, _util\_preprocess_ and _util\_estimation_ contain the helper functions for 
general use, preprocessing and estimation respectively. Functions are for instance for getting data paths, 
variable names, resampling and interpolating data, pivot data in preparation for modeling and saving results,
respectively.

### preprocessing.py
This code loads the source data, performs transformations such as selecting countries and years,
interpolate the data, making the data stationary, and saving the total dataframes used for the
counterfactual analysis modeling. No arguments are needed to run this code:
```python
python preprocess.py
```

### estimators.py
This file contains the functions for using the arco method with estimation models LASSO, Random Forest and OLS.
Also contains functions for performing the Synthetic Control and Difference-in-differences methods.


### main.py
This file contains the functions to activate the estimator functions found in _estimators.py_. 
The data is read and based on the arguments (model, timeframe, treatment\_country) estimation is done.
```python
python main.py '{model}' '{timeframe}' '{treatment_country}'
```
For the arguments for model, timeframe and target_country the following are available 
(please note in this thesis only timeframe _'m'_ (monthly) is used, while _q_ (quarterly) is not used):
- model = {'lasso', 'rf', 'ols', 'sc', 'did'}
- timeframe = {'m', 'q'}
- treatment_country = {'switzerland', 'ireland', 'united_kingdom', 'france', 'portugal'}

### gen_results.py
While _main.py_ can run results for 1 model for 1 timeframe for 1 treatment country, this file is able to
loop over the _main.py_ counterfactual method function for all models, timeframes and treatment counties
to efficiently create all the results needed.
```python
python gen_results.py '{timeframe}'
```
For the timeframe arguemen the following are available available 
(please note in this thesis only timeframe _'m'_ (monthly) is used, while _q_ (quarterly) is not used):
- timeframe = {'m', 'q'}

### plot_function.py
This file contains all functions for plotting the predictions, erros, cumulative summations, lasso paths,
and any other plots found in the thesis paper.

### statistical_tests.py
This file contains all functions for applying the statistical tests used in the thesis paper, such as the
Augmented Dickey-Fuller test, the Durbin-Watson test and the two-sided T-test.






## Explanation of folders

### data
The data folder contains the source data, as well as transformed data for all variables, included countries, 
as well as the total dataframes (both stationary and non-stationary):
- data  
&nbsp; > m / q  
&nbsp; &nbsp; > countries


### output
The output folder contains all output from the different models and methods used, 
including both the figures and tables:
- output  
&nbsp; > m / q  
&nbsp; &nbsp; > figures / tables  
&nbsp; &nbsp; &nbsp; > data / methodology / results
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; > all results (csv) and results per country  


### notebooks
The notebooks folder contains Jupyter notebooks used for quick analysis, processing of results,
and exploratory data analysis.

