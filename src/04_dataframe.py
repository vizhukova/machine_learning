import numpy as np
import pandas as pd
import timeit

np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
print(mydata)

myindex = ['CA','NY','AZ','TX']
mycolumns = ['Jan','Feb','Mar']
df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
print(df)
print(df.info())
file_data = pd.read_csv('tips.csv')
print('file data: ',file_data)
print('columns: ', file_data.columns)
print('index: ', file_data.index) # range index object
print('head: ', file_data.head(10))
print('tail:', file_data.tail())
print('info: ', file_data.info()) # returns the structure of the data
print('statistics: ', file_data.describe())
print('statistics: ', file_data.describe().transpose()) #makes easier to read when you see columns on their places

print(file_data['total_bill'])
print(type(file_data['total_bill'])) #series

print(file_data[['total_bill','tip']]) # table with values nx3 + index
print(type(file_data[['total_bill','tip']])) #dataframe
print(file_data['total_bill'] + file_data['tip']) #sum all elements by keys

file_data['tip_percentage'] = 100 * file_data['tip'] / file_data['total_bill']

print('tip_percentage: ', file_data['tip_percentage'])
print(file_data)

file_data['price_per_person'] = np.round(file_data['total_bill'] / file_data['size'], 2)
print(file_data['price_per_person'])

print(file_data.drop("tip_percentage",axis=1))

file_data = file_data.set_index("Payment ID")
print(file_data)

# file_data = file_data.reset_index()
# print(file_data)

print('Get row by index: \n', file_data.iloc[0])
print('Get row by label: \n', file_data.loc['Sun2959'])

print('Slice by index: \n', file_data.iloc[0:4])
print('Slice by label: \n', file_data.loc[['Sun2959', 'Sat2657']])

print(file_data.drop('Sat2657', axis=0)) # can be used without axis value, just to be clear
# drop by index is not an option - will cause an error

#
# Filtering data
#
print(file_data['total_bill'] > 40) #series of boolean values
print('Filtered data: ', file_data[file_data['total_bill'] > 40])
print('Filtered data with and: ', file_data[(file_data['total_bill'] > 30) & (file_data['sex'] == 'Male')])
print('Filtered data with or: ', file_data[(file_data['day'] == 'Sun') | (file_data['day'] == 'Sat') | (file_data['day'] == 'Fri')])
print('Filtered data: ', file_data[file_data['day'].isin(['Sun', 'Sat', 'Fri'])])

def last_four(num):
    return str(num)[-4:]

print('Apply formating: ', file_data['CC Number'].apply(last_four))
file_data['last_four'] = file_data['CC Number'].apply(last_four)
print(file_data)

def yelp(price):
    if price < 10:
        return '$'
    if price < 30:
        return '$$'
    return '$$$'

file_data['yelp'] = file_data['total_bill'].apply(yelp)
print(file_data)

def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return 'Generous'
    return 'Other'

file_data['Quality'] = file_data[['total_bill', 'tip']].apply(lambda df: quality(df['total_bill'], df['tip']), axis = 1)
print(file_data)
# The faster solution: 
file_data['Quality'] = np.vectorize(quality)(file_data['total_bill'], file_data['tip'])
print(file_data)

setup = '''
import numpy as np
import pandas as pd

file_data = pd.read_csv('tips.csv')
def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return 'Generous'
    return 'Other'
'''

stmt_1 = '''
file_data['Quality'] = file_data[['total_bill', 'tip']].apply(lambda df: quality(df['total_bill'], df['tip']), axis = 1)
'''
stmt_2 = '''
file_data['Quality'] = np.vectorize(quality)(file_data['total_bill'], file_data['tip'])
'''

timeit_1 = timeit.timeit(setup=setup, stmt=stmt_1, number=1000)
timeit_2 = timeit.timeit(setup=setup, stmt=stmt_2, number=1000)

print('timeit_1: ', timeit_1)
print('timeit_2: ', timeit_2)

print(file_data.sort_values(['tip', 'size'], ascending = False))

print('Max value: ', file_data['total_bill'].max())
print('Max index value: ', file_data['total_bill'].idxmax())
print(file_data.describe().corr())

print(file_data['sex'].value_counts())
print(file_data['sex'].unique())

#
# Replace
#
print(file_data['sex'].replace(['Female', 'Male'], ['F', 'M']))
mymap = {'Female': 'F', 'Male': 'M'}
print(file_data['sex'].map(mymap))

simple_df = pd.DataFrame([1, 2, 2, 2], ['a', 'b', 'c', 'd'])
print('simple_df', simple_df)
print('Show where are duplications with the bool series: ', simple_df.duplicated())
print('Remove all duplicates: ', simple_df.drop_duplicates())
print('between: ', file_data['total_bill'].between(10, 20))
print('show 2 largest tips: ', file_data.nlargest(2, 'tip'))
#same
print(file_data.sort_values('tip', ascending=False).iloc[0:2])

print('Sample with 5 different random rows: ', file_data.sample(5))
print('Sample with 10%\ different random rows: ', file_data.sample(frac=0.1))