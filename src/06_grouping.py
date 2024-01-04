import numpy as np
import pandas as pd

df = pd.read_csv('mpg.csv')
print(df)
print(df['model_year'].value_counts())
print('sum:', df.groupby('model_year').sum())
print(df.groupby('model_year').sum()['mpg'])
print(df.groupby('model_year'))
print(df.groupby('model_year').describe().transpose())

print(df.keys)
avg_year = df.groupby('model_year')['acceleration'].mean()

data_one = {'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']}
data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}
one = pd.DataFrame(data_one)
two = pd.DataFrame(data_two)

data_test = {'Sex': ['F', 'M', 'F', 'F'], 'Age': [18, 20, 7, 5], 'Survived': [6,7,8,3], 'SexCode': [0,1,0,0], 'Some_Str': ['3', 'fdff', 'jjj', 'o']}
data_test_frame = pd.DataFrame(data_test)
print(data_test_frame)
with_excluded_data = data_test_frame.loc[:, data_test_frame.columns!='Some_Str']
print(with_excluded_data.groupby('Sex').mean())

print('Concat by columns: ', pd.concat([one, two], axis=1))

print('Concat by rows: ', pd.concat([one, two], axis=0))
two.columns = one.columns
print('Concat by rows: ', pd.concat([one, two], axis=0))

#
# Merge
#

registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})
print('registrations: ', registrations)
print('logins: ', logins)

print('inner: ', pd.merge(registrations, logins, how='inner', on='name'))
print('left: ', pd.merge(registrations, logins, how='left', on='name'))
print('right: ', pd.merge(registrations, logins, how='right', on='name'))
print('outer: ', pd.merge(registrations, logins, how='outer', on='name'))

registrations = registrations.set_index("name")
print(registrations)
print(pd.merge(registrations,logins,left_index=True,right_on='name'))
print(pd.merge(logins,registrations,right_index=True,left_on='name'))

registrations = registrations.reset_index()
print(registrations)
#rename columns
registrations.columns = ['reg_name','reg_id']
# ERROR
# pd.merge(registrations,logins)
print(pd.merge(registrations,logins,left_on='reg_name',right_on='name'))
print(pd.merge(registrations,logins,left_on='reg_name',right_on='name').drop('reg_name',axis=1))

# Pandas automatically tags duplicate columns
registrations.columns = ['name','id']
logins.columns = ['id','name']
print('registrations: ', registrations)
print('logins: ', logins)
# _x is for left
# _y is for right
pd.merge(registrations,logins,on='name')
print(pd.merge(registrations,logins,on='name',suffixes=('_reg','_log')))
