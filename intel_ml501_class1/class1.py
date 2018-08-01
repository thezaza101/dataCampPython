from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path =  os.path.abspath(os.path.join('intel_ml501_class1','datasets'))
#data_path = ['...........Intel-ML101-Class1\\Intel-ML101_Class1\\data\\']
print (data_path)

## Q1
filepath = data_path + "\\Iris_Data.csv"
print(filepath)
data = pd.read_csv(filepath)
print(data.head())

# Number of rows
print(data.shape[0])

# Column names
print(data.columns.tolist())

# Data types
print(data.dtypes)

## Q2
# The str method maps the following function to each entry as a string
data['species'] = data.species.str.replace('Iris-', '')
# alternatively
# data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))

print(data.head())

## Q3
##The number of each species presen
print(data.groupby('species').nunique())

## The mean, median, and quantiles and ranges (max-min) for each petal and sepal measurement.
desc = data.describe()
desc.loc["range"] = desc.loc['max'] - desc.loc['min']
print(desc)

## Q4
# The mean calculation
data.groupby('species').mean()

# The median calculation
data.groupby('species').median()

# applying multiple functions at once - 2 methods
data.groupby('species').agg(['mean', 'median'])  # passing a list of recognized strings
data.groupby('species').agg([np.mean, np.median])  # passing a list of explicit aggregation functions

# If certain fields need to be aggregated differently, we can do:
from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'
pprint(agg_dict)
data.groupby('species').agg(agg_dict)

## Q5
# A simple scatter plot with Matplotlib
ax = plt.axes()

ax.scatter(data.sepal_length, data.sepal_width)

# Label the axes
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width')

# show the plot
plt.show()

## Q6
ax = plt.hist(data['petal_length'])

plt.xlabel('petal length')
plt.ylabel('feq')

# show the plot
plt.show()

## TBA...