import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

# 
# Dealing with the categorical data
# 

#
# One hot encoding
#
# We use this categorical data encoding technique when the features are nominal(do not have any order). 
# In one hot encoding, for each level of a categorical feature, we create a new variable. 
# Each category is mapped with a binary variable containing either 0 or 1. 
# Here, 0 represents the absence, and 1 represents the presence of that category.

df = pd.read_csv('DATA/Ames_NO_Missing_Data.csv')
print(df.head())

df['MS SubClass'] = df['MS SubClass'].apply(str)
direction = pd.Series(['Up', 'Up', 'Down'])
print(direction)

# Dummy coding scheme is similar to one-hot encoding. 
# This categorical data encoding method transforms the categorical variable into a set 
# of binary variables (also known as dummy variables). 
# In the case of one-hot encoding, for N categories in a variable, it uses N binary variables. 
# The dummy encoding is a small improvement over one-hot-encoding.

df_object_dummies = pd.get_dummies(direction, drop_first = True)
# translate values to false and true
print('One hot encoding: ', df_object_dummies)

# shows if the value is a string
my_object_df = df.select_dtypes(include = 'object')
my_numeric_df = df.select_dtypes(exclude = 'object')

df_object_dummies = pd.get_dummies(my_object_df, drop_first = True)
final_df = pd.concat([df_object_dummies, my_numeric_df],axis = 1)
print(final_df)

# Label Encoding

# Converting type of columns to category
def covert_to_number_categorical_numbers(column):
    return column.astype('category').cat.codes 

encoded = my_object_df.apply(covert_to_number_categorical_numbers, axis=0) 
print('Label Encoding: ', encoded)

# Create an instance of One-hot-encoder 
enc = OneHotEncoder() 

# Passing encoded columns   
# same as get_dummies - lots of duplicated columns
enc_data = pd.DataFrame(enc.fit_transform(my_object_df).toarray()) 
  
# Merge with main 
New_df = my_numeric_df.join(enc_data) 

print(New_df) 

encoder = ce.OneHotEncoder(cols=my_object_df.columns, handle_unknown='return_nan', return_df=True, use_cat_names=True)
encoded = encoder.fit_transform(my_object_df)
print('OneHot --> ', encoded)

#
# Ordinary encoding
#

sizes = ['small', 'medium', 'large']

# Encode colors as incremental integers 
encoded_sizes = np.arange(len(sizes))

print(encoded_sizes)


encoder = OrdinalEncoder()

# reshape to 2D array
sizes_reshaped = np.array(sizes).reshape(-1,1)

encoded = encoder.fit_transform(sizes_reshaped) 

print(encoded)

#
# Binary encoding
#

encoder = ce.BinaryEncoder(cols = my_object_df.columns , return_df = True)
encoded = encoder.fit_transform(my_object_df)
print('Binary --> ', encoded)

#
# Effect Encoding
# ( aka Deviation Encoding or Sum Encoding)
#
# Effect encoding is almost similar to dummy encoding, with a little difference. 
# In dummy coding, we use 0 and 1 to represent the data but in effect encoding, 
# we use three values i.e. 1,0, and -1.
encoder = ce.SumEncoder(cols = my_object_df.columns, verbose = False)
encoded = encoder.fit_transform(my_object_df)
print('Deviation --> ', encoded)

# 
# Hash Encoder
#
# To understand Hash encoding it is necessary to know about hashing. 
# Hashing is the transformation of arbitrary size input in the form of a fixed-size value. 
# We use hashing algorithms to perform hashing operations i.e to generate the hash value of an input. 
# Further, hashing is a one-way process, in other words, one can not generate original input from the 
# hash representation.

# data=pd.DataFrame({'Month':['January','April','March','April','Februay','June','July','June','September']})
# encoder=ce.HashingEncoder(cols='Month',n_components=6)
# encoded = encoder.fit_transform(data)
# print('Hash Encoder: ', encoded)

#
# Target Encoding
#
# Unlike one-hot encoding, which creates binary columns for each category, 
# target encoding calculates and assigns a numerical value to each category 
# based on the relationship between the category and the target variable. 
# Typically used for classification tasks, it replaces the categorical values with their 
# corresponding mean (or other statistical measures) of the target variable within each category.

data=pd.DataFrame({'class':['A,','B','C','B','C','A','A','A'],'Marks':[50,30,70,80,45,97,80,68]})
#Create target encoding object
encoder=ce.TargetEncoder(cols='class') 
encoded = encoder.fit_transform(data['class'], data['Marks'])
print('TargetEncoder: ', encoded)

#
# Base N Encoding
#
# In the numeral system, the Base or the radix is the number of digits or a combination of digits 
# and letters used to represent the numbers. The most common base we use in our life is 10  or 
# decimal system as here we use 10 unique digits i.e 0 to 9 to represent all the numbers. 
# Another widely used system is binary i.e. the base is 2. It uses 0 and 1 i.e 2 digits to 
# express all the numbers.

data=pd.DataFrame({'City':['Delhi','Mumbai','Hyderabad','Chennai','Bangalore','Delhi','Hyderabad','Mumbai','Agra']})
#Create an object for Base N Encoding
encoder = ce.BaseNEncoder(cols=['city'], return_df=True, base=5)
encoded = encoder.fit_transform(data)
print('Base N: ', encoded)