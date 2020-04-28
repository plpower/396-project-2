import csv
import matplotlib.pyplot as plt
import numpy as np

data = open('bid_data.csv')
csv_data = csv.reader(data)
# given_values = [32.1, 50, 31, 81.4]
given_values = list(range(1, 101, 1))
our_bids = [30, 45, 30, 75]
