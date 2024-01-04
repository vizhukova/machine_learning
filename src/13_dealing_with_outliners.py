import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Choose a mean,standard deviation, and number of samples

def create_ages(mu=50,sigma=13,num_samples=100,seed=42):

    # Set a random seed in the same cell as the random call to get the same values as us
    # We set seed to 42 (42 is an arbitrary choice from Hitchhiker's Guide to the Galaxy)
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages,decimals=0)
    
    return sample_ages

sample = create_ages()
print(sample, len(sample))
sns.displot(sample, bins = 20)

plt.figure()
sns.boxplot(sample)

series = pd.Series(sample)
desc = series.describe()

print('describe: ', desc)

IQR = desc['75%'] - desc['25%']
lower_limit = desc['25%'] - 1.5 * IQR
print('lower_limit: ', lower_limit)
print(series[series > lower_limit])
print(series[series <= lower_limit])

q75, q25 = np.percentile(sample, [75, 25])
iqr = q75 - q25
print(iqr)

df = pd.read_csv('DATA/Ames_Housing_Data.csv')
print(df.head())
only_numeric_df = df._get_numeric_data()
print('how do the features correlate between each other: \n', only_numeric_df.corr())

sns.heatmap(only_numeric_df.corr())

print(only_numeric_df.corr()['SalePrice'].sort_values())

plt.figure()
plt.title("Origin data: ")
sns.scatterplot(data =  df, x = 'Overall Qual', y = 'SalePrice')

plt.figure()
plt.title("Origin data: ")
sns.scatterplot(data =  df, x = 'Gr Liv Area', y = 'SalePrice')

print('Houses that spoil the data: ', df[(df['Overall Qual'] > 8) & (df['SalePrice'] < 200000)])

ind_drop = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index

df = df.drop(ind_drop,axis=0)

plt.figure()
plt.title('Data with dropped values')
sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)
plt.figure()
plt.title('Data with dropped values')
sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)

plt.show()

df.to_csv("DATA/Ames_outliers_removed.csv",index=False)
