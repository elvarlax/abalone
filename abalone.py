# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Adding header
attribute_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                   'Shell_weight', 'Rings']

# Importing the dataset
dataset = pd.read_csv('abalone.csv', header=None, names=attribute_names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Basic summary statistics
print(dataset.describe())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Printing X_test
print(X_test)

# Printing X_train
print(X_train)
