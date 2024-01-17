import collections
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.datasets import make_regression, make_classification, make_blobs

# Load digits dataset
digits = datasets.load_digits()
# Create features matrix
features = digits.data
# Create target vector
target = digits.target

# load_boston
# Contains 503 observations on Boston housing prices. It is a good dataset for exploring regression algorithms.
# load_iris
# Contains 150 observations on the measurements of Iris flowers. It is a good dataset for exploring classification algorithms.
# load_digits
# Contains 1,797 observations from images of handwritten digits. It is a
# good dataset for teaching image classification.

print('View first observation: ', features[0])

# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples=100,
                                                 n_features=3,
                                                 n_informative=3,
                                                 n_targets=1,
                                                 noise=0.0,
                                                 coef=True,
                                                 random_state=1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# Generate features matrix and target vector
features, target = make_classification(n_samples=100,
                                       n_features=3,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.25, .75],
                                       random_state=1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# Generate feature matrix and target vector
features, target = make_blobs(n_samples=100,
                              n_features=2,
                              centers=3,
                              cluster_std=0.5,
                              shuffle=True,
                              random_state=1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# View scatterplot
plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()

# Create URL
url = 'https://tinyurl.com/simulated_data'
# Load dataset
dataframe = pd.read_csv(url)
# View first two rows
dataframe.head(2)

# Create URL
url = 'https://tinyurl.com/simulated_excel'
# Load data
dataframe = pd.read_excel(url, sheetname=0, header=1)
# View the first two rows
dataframe.head(2)

# Create URL
url = 'https://tinyurl.com/simulated_json'
# Load data
dataframe = pd.read_json(url, orient='columns')
# View the first two rows
dataframe.head(2)

# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
# View first two rows
dataframe.head(2)

# Create DataFrame
dataframe = pd.DataFrame()
# Add columns
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]
# Show DataFrame
print('Created data frame: ', dataframe)
# Create row
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name','Age','Driver'])
# Append row
dataframe.append(new_person, ignore_index=True)
print('Show length of rows and columns: ',  dataframe.shape)
print('Descriptive statistics for any numeric columns: ', dataframe.describe())
print('Select first row: ', dataframe.iloc[0])
print('Select the second, third, and fourth rows: ', dataframe.iloc[1:4])
print('Select three rows: ', dataframe.iloc[:4])

dataframe = dataframe.set_index(dataframe['Name'])
# Show row
print('Show row: ', dataframe.loc['Allen, Miss Elisabeth Walton'])
print('Show top two rows where column "sex" is "female": ', dataframe[dataframe['Sex'] == 'female'].head(2))
print('Filter rows: ', dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)])

#
# Repacement
#
print('Replace values, show two rows: ', dataframe['Sex'].replace("female", "Woman").head(2))
print('Replace "female" and "male with "Woman" and "Man"', dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))
print('Replace values, show two rows: ', dataframe.replace(1, "One").head(2))
print('Replace values, show two rows: ', dataframe.replace(r"1st", "First", regex=True).head(2))
print('Rename column, show two rows: ', dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2))
print('Rename columns, show two rows: ', dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2))

# Create dictionary
column_names = collections.defaultdict(str)
# Create keys
for name in dataframe.columns:
    column_names[name]
# Show dictionary
print('Created dictionary: ', column_names)

# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
# variance (var), standard deviation (std), kurtosis (kurt), skewness (skew), standard error of the mean (sem), mode (mode), median (median)

print('Show counts: ', dataframe.count())
print('Select unique values', dataframe['Sex'].unique())   
print('Counts of unique value: ', dataframe['Sex'].value_counts())
print('Classes in PClass: ', dataframe['PClass'].value_counts())
print('Show number of unique values: ', dataframe['PClass'].nunique())

print('Select missing values, show two rows: ', dataframe[dataframe['Age'].isnull()].head(2))
print('Select not missing values, show two rows: ', dataframe[dataframe['Age'].notnull()].head(2))

# Replace values with NaN
# dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
#  Don't use NaN - it's not existing in python, use np.nan instead
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

# Load data, set missing values
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

# Delete column
dataframe.drop('Age', axis=1).head(2)
 # Drop columns
dataframe.drop(['Age', 'Sex'], axis=1).head(2)
# Drop column
dataframe.drop(dataframe.columns[1], axis=1).head(2)
# Create a new DataFrame
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)
# Delete rows, show first two rows of output
dataframe[dataframe['Sex'] != 'male'].head(2)
# Delete row, show first two rows of output
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)
# Delete row, show first two rows of output
dataframe[dataframe.index != 0].head(2)

# Drop total duplicates rows, show first two rows of output
dataframe.drop_duplicates().head(2) 
# Drop duplicates by column value Sex
dataframe.drop_duplicates(subset=['Sex'])
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'], keep='last')

# Group rows by the values of the column 'Sex', calculate mean
# of each group
dataframe.groupby('Sex').mean()
# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()
# Group rows, calculate mean
dataframe.groupby(['Sex','Survived'])['Age'].mean()