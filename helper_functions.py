import math, random, csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def add_win_column(scoreRow):  
    # Make a column that represents if the home team won, tied or loss
    # home win = 1, home loss = -1, draw = 0
    
    if scoreRow[0] > scoreRow[1]:
        return 1
    elif scoreRow[0] == scoreRow[1]:
        return 0
    else:
        return -1