import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split

## loading the file
health_care = pd.read_csv('./healthcare/train_data.csv')

## getting the inputs and labels

X = health_care.drop('Stay', axis=1)
y = health_care['Stay']

