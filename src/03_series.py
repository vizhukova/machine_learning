import numpy as np
import pandas as pd

my_index = ['Ukraine', 'USA', 'Canada', 'Mexico']
my_data = [1991, 1776, 1867, 1821]
my_ser = pd.Series(data=my_data, index=my_index)  # it's kind of dictionary
print('Type is: ', type(my_ser))
print('My series is: ', my_ser)

print('Get first element: ', my_ser[0])
print('Get first element: ', my_ser['Ukraine'])

ages = {'Sam': 5, 'Frank':10, 'Spike': 7}
ages_ser = pd.Series(ages)
print(ages_ser)
print('Get first element: ', ages_ser[0])
print('Get first element: ', ages_ser['Sam'])

q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}
# Convert into Pandas Series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q2)

print('Will summarize together values with identical keys: ', sales_Q1 + sales_Q2)
# but will return NaN if the key does not existing in some series 


print('Will keep value of missing keys: ', sales_Q1.add(sales_Q2, fill_value=0))