################################
### import relevant packages ###
################################
import sys

from preprocess import preprocess
from main import main

from definitions import target_countries


def gen_results(timeframe: str):

    # preprocess()

    for model in ['arco']:

        for target_country in target_countries:
            print('============================================================')
            main(model=model, timeframe=timeframe, target_country=target_country)
            print("\n")


if __name__ == "__main__":
    gen_results(timeframe=sys.argv[1])
