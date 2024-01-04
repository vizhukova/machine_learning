import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 
# Dealing with the categorical data
# 

df = pd.read_csv('DATA/Ames_NO_Missing_Data.csv')
print(df.head())

df['MS SubClass'] = df['MS SubClass'].apply(str)
direction = pd.Series(['Up', 'Up', 'Down'])
print(direction)

df_object_dummies = pd.get_dummies(direction, drop_first = True)
# translate values to false and true
print(df_object_dummies)

# shows if the value is a string
my_object_df = df.select_dtypes(include = 'object')
my_numeric_df = df.select_dtypes(exclude = 'object')

df_object_dummies = pd.get_dummies(my_object_df, drop_first = True)
final_df = pd.concat([my_numeric_df, my_numeric_df],axis = 1)

print(final_df)