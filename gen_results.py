################################
### import relevant packages ###
################################
from preprocess import preprocess
from main import main

from definitions import target_countries


if __name__ == "__main__":

    preprocess()

    for target_country in target_countries:
        main(timeframe='q', target_country=target_country)
        print("\n")


