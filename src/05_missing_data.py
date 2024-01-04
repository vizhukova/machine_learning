import numpy as np
import pandas as pd

print(np.nan, pd.NA, pd.NaT)

myvar = np.nan
print('Will return false: ', myvar == np.nan)
print('Will return true: ', myvar is np.nan)

df = pd.read_csv('movie_scores.csv')
print(df)
print(df.isnull())
print(df.notnull())
print(df[ (df['pre_movie_score'].isnull()) & (df['first_name'].notnull()) ])
print('Drops all rows with empty value, that do not have atleast 2 non empty values: ', df.dropna(thresh=2))
print('Drops all rows that do not have value in last_name', df.dropna(subset=['last_name']))

print('Fill empties with provided data: ', df.fillna('<3'))

print(df['pre_movie_score'].fillna(df['pre_movie_score'].mean()))
# print(df.fillna(df.mean()))

airline_tix = {'first':100,'business':np.nan,'economy-plus':50,'economy':30}
ser = pd.Series(airline_tix)
print(ser.interpolate())