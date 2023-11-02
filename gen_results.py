# import relevant packages
import sys

from preprocess import preprocess
from main import main

from definitions import treatment_countries


##################################################
### Generate all results given timeframe       ###
##################################################
def gen_results(timeframe: str):

    preprocess()

    for model in ['lasso', 'rf', 'ols', 'sc', 'did']:

        for treatment_country in treatment_countries:
            print('============================================================')
            main(model=model, timeframe=timeframe, treatment_country=treatment_country)
            print("\n")


if __name__ == "__main__":
    gen_results(timeframe=sys.argv[1])
