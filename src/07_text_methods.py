import pandas as pd 
from datetime import datetime

mystring = 'hello'
print('capitalize: ', mystring.capitalize())
print('isdigit: ', mystring.isdigit())

names = pd.Series(['andrew','bobo','claire','david','4'])
print('names: ', names)
print('capitalize:', names.str.capitalize())
print('isdigit: ', names.str.isdigit())

tech_finance = ['GOOG,APPL,AMZN','JPM,BAC,GS']
tickers = pd.Series(tech_finance)
print('tickers: ', tickers)
print('tickers splitted: ', tickers.str.split(',', expand=True))

#
# Cleaning or Editing Strings
#

messy_names = pd.Series(["andrew  ","bo;bo","  claire  "])
print(messy_names)
print(messy_names.str.replace(";","").str.strip().str.capitalize())

def cleanup(name):
    name = name.replace(";","")
    name = name.strip()
    name = name.capitalize()
    return name
print(messy_names.apply(cleanup))

my_year = 2017
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15

my_date = datetime(my_year,my_month,my_day)
print(my_date)
my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second)
print(my_date_time)
print(my_date.day)
print(my_date.hour)

myser = pd.Series(['Nov 3, 2000', '2000-01-01', None])
print(myser)
print(pd.to_datetime(myser,format='mixed'))

# As usually month is first, when we put day at first place - we should announce it
obvi_euro_date = '31-12-2000'
print(pd.to_datetime(obvi_euro_date,dayfirst=True) )

# Custom dates
style_date = '12--Dec--2000'
print(pd.to_datetime(style_date, format='%d--%b--%Y'))
strange_date = '12th of Dec 2000'
print(pd.to_datetime(strange_date))

sales = pd.read_csv('RetailSales_BeerWineLiquor.csv')
print(sales)
print(sales.iloc[0]['DATE'])
sales['DATE'] = pd.to_datetime(sales['DATE'])
print(sales)

sales = pd.read_csv('RetailSales_BeerWineLiquor.csv', parse_dates=[0])
print(sales)

sales = sales.set_index('DATE')
# aka group by
print(sales.resample(rule='A').mean())