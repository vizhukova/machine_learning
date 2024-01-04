import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("DATA/Advertising.csv")
print(df.head())

# fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV Spend")

# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].set_title("Radio Spend")
# axes[1].set_ylabel("Sales")

# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].set_title("Newspaper Spend")
# axes[2].set_ylabel("Sales")
# plt.tight_layout()

# sns.pairplot(df) # is a quick way to make all those subplot in one row 
# bit it will also compare features between each other

# get all columns except sales - features
X = df.drop('sales',axis=1)
y = df['sales']

# random_state: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = LinearRegression()
model.fit(X_train,y_train)

# We only pass in test features
# The model predicts its own y hat
# We can then compare these results to the true y test label value
test_predictions = model.predict(X_test) # should be kind of y_test

print('mean value: ', df['sales'].mean())
# sns.histplot(data = df, x = 'sales', bins = 20)
# plt.show()


MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

# Can tell if it was a good choice to take linear regression as method of prediction for that data set
test_residuals = y_test - test_predictions
# no clear line or curve says that it can be a good choice for linear regression, 
# it should be more or less random and distributed around zero line
# sns.scatterplot(x = y_test, y = test_residuals)
# plt.axhline(y = 0, color = 'red', ls = '--')

# kde nearly close to zero - that's good 
# sns.displot(test_residuals, bins = 25, kde = True)

# ANOTHER WAY TO VIEW COMPARISON:
# Create a figure and axis to plot on
fig, ax = plt.subplots(figsize=(6,8),dpi=100)
# probplot returns the raw values if needed
# we just want to see the plot, so we assign these values to _
# red line - is like normal distribution should look like
# sp.stats.probplot(test_residuals, plot = ax)

# Creation of the final model:
final_model = LinearRegression()
final_model.fit(X, y)

# return 3 elements array: coefficients for TV, radio and newspaper feature
# shows relation between each feature and sales (label) if it's nearly 0 - means
# there are no relation bwtween values
# means if we add +1$ to advertisment to some feature - sales will increase on coefficient value
print('coefficients: ', final_model.coef_)

y_hat = final_model.predict(X)

# fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].plot(df['TV'],y_hat,'o', color = 'red')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV Spend")

# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].plot(df['radio'],y_hat,'o', color = 'red')
# axes[1].set_title("Radio Spend")
# axes[1].set_ylabel("Sales")

# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].plot(df['newspaper'],y_hat,'o', color = 'red')
# axes[2].set_title("Newspaper Spend")
# axes[2].set_ylabel("Sales")
# plt.tight_layout()

# plt.show()

# save the model
dump(final_model, 'sales_model.joblib') 

# load model
loaded_model = load('sales_model.joblib')
print(loaded_model.coef_)

# 149 TV, 22 radio, 12 newspaper
campaign = pd.DataFrame(data = [[149, 22, 12]], columns=['TV', 'radio', 'newspaper'])
print(loaded_model.predict(campaign))