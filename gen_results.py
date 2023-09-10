################################
### import relevant packages ###
################################
import pandas as pd

from preprocess import preprocess
from main import main

from definitions import target_countries

if __name__ == "__main__":

    preprocess()
    for target_country in target_countries:
        main(target_country=target_country)
        print("\n")
