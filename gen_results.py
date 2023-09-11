################################
### import relevant packages ###
################################
import sys

from preprocess import preprocess
from main import main

from definitions import target_countries


def gen_results(model: str, timeframe: str):
    preprocess()

    for target_country in target_countries:
        main(model=model, timeframe=timeframe, target_country=target_country)
        print("\n")


if __name__ == "__main__":
    gen_results(model=sys.argv[1], timeframe=sys.argv[2])


