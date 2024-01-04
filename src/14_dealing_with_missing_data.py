import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('DATA/Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())

df = pd.read_csv("DATA/Ames_outliers_removed.csv")
print(df.head())
print('columns: ', len(df.columns))
print('info: ', df.info)
df = df.drop('PID',axis=1)

print('shows if there are empty data', df.isnull())

def percent_missing(df):
    percent_nan = 100* df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan>0].sort_values()
    return percent_nan

percent_nan = percent_missing(df)
print('percent_nan', percent_nan)

sns.barplot(x = percent_nan.index, y = percent_nan)
plt.xticks(rotation = 90)
# Set 1% Threshold
plt.ylim(0,1)

print(percent_nan[percent_nan < 1])

def print_persantage_missing_data_plot(df, show = False):
    percent_nan = percent_missing(df)
    plt.figure()
    sns.barplot(x=percent_nan.index,y=percent_nan)
    plt.xticks(rotation=90)
    plt.ylim(0,1)
    if ( show):
        plt.show()

# drop row with empty values of specific features
df = df.dropna(axis=0,subset= ['Electrical','Garage Cars'])
print_persantage_missing_data_plot(df)

# The numerical basement columns:
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

# The string basement columns:
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')
print_persantage_missing_data_plot(df)

df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
print_persantage_missing_data_plot(df)

print(df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']])

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)
print_persantage_missing_data_plot(df)

df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)
print_persantage_missing_data_plot(df)

df['Fireplace Qu'] = df['Fireplace Qu'].fillna("None")
print_persantage_missing_data_plot(df)


# 
# Imputation of missing data
# 

# To impute missing data, we need to decide what other filled in (no NaN values) feature 
# most probably relates and is correlated with the missing feature data. In this particular 
# case we will use:

print('unique neighborhood: ', df['Neighborhood'].unique())
plt.figure(figsize=(8,12))
sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')

print(df.groupby('Neighborhood')['Lot Frontage'].mean())

# fill all NA values with the middle value from the table
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
print('missed filled data: ', df['Lot Frontage'].iloc[21:26])
print_persantage_missing_data_plot(df)

print('quantity of null values: ', df.isnull().sum())

df['Lot Frontage'] = df['Lot Frontage'].fillna(0)
percent_nan = percent_missing(df)
# No empty data anymore 
print(percent_nan)

df.to_csv("DATA/Ames_NO_Missing_Data.csv",index=False)