################################
### import relevant packages ###
################################
import sys

from preprocess import preprocess
from main import main

from definitions import treatment_countries


def gen_results(timeframe: str):

    # preprocess()

    for model in ['sc']:

        for treatment_country in treatment_countries:
            print('============================================================')
            main(model=model, timeframe=timeframe, treatment_country=treatment_country)
            print("\n")


if __name__ == "__main__":
    gen_results(timeframe=sys.argv[1])
