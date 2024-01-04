import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("DATA/Advertising.csv")
print(df.head())

# Everything BUT the sales column
X = df.drop('sales',axis=1)
y = df['sales']

polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)

# Converter "fits" to data, in this case, reads in every X column
# Then it "transforms" and ouputs the new polynomial data
# retuns nine features = 3 orginal and other - interactions between each other
poly_features = polynomial_converter.fit_transform(X)

print(poly_features[0])
# returns array([2.301000e+02, 3.780000e+01, 6.920000e+01, 5.294601e+04,
#    8.697780e+03, 1.592292e+04, 1.428840e+03, 2.615760e+03,
#    4.788640e+03])
print('original data: ', poly_features[0][:3])
print(poly_features[0][:3]**2)
# returns  from 3 to 6 elements of poly_features: array([52946.01,  1428.84,  4788.64])
# other : x1*x2, x1*x3, x2*x3

#  random_state: 
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

model = LinearRegression(fit_intercept=True)
model.fit(X_train,y_train)

test_predictions = model.predict(X_test)

MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('mean: ', df['sales'].mean())

# Choosing a Model
# create the different order poly
# split poly feat train/test
# fit on train
# store / save the rmse or BOTH the train
# PLOT the results (error vs poly order)

# TRAINING ERROR PER DEGREE
train_rmse_errors = []
# TEST ERROR PER DEGREE
test_rmse_errors = []

for d in range(1,10):
    
    # CREATE POLY DATA SET FOR DEGREE "d"
    polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)
    
    # SPLIT THIS NEW POLY DATA SET
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
    # TRAIN ON THIS NEW POLY SET
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train,y_train)
    
    # PREDICT ON BOTH TRAIN AND TEST
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate Errors
    
    # Errors on Train Set
    train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
    
    # Errors on Test Set
    test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))

    # Append errors to lists for plotting later
    train_rmse_errors.append(train_RMSE)
    test_rmse_errors.append(test_RMSE)

print('train_rmse_errors: ', train_rmse_errors)
# returns [
# 1.734594124329376, 
# 0.5879574085292231, 
# 0.43393443569020657, 
# 0.35170836883993567, 
# 0.25093429238913045, 
# 0.19370257126598295, 
# 5.421954881974639,  # spike here is a red flag here (overfitting)
# 0.14322153048566827, 
# 0.17199519166776434
# ]

print('test_rmse_errors: ', test_rmse_errors)
# returns [
# 1.5161519375993877, 
# 0.6646431757269278, 
# 0.5803286825238902, 
# 0.5077742639532156, 
# 2.575818786687965, 
# 4.28632967597874, 
# 1378.6346061081026, #exploading here - red flag
# 4270.765913091457, 
# 94443.25605204244
# ]

# region of rist is exploding at 4th degree hight, so it's better to stop on 3th or 2nd degree
# plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN')
# plt.plot(range(1,6),test_rmse_errors[:5],label='TEST')
# plt.xlabel("Polynomial Complexity")
# plt.ylabel("RMSE")
# plt.legend()
# plt.show()

# plt.plot(range(1,10),train_rmse_errors,label='TRAIN')
# plt.plot(range(1,10),test_rmse_errors,label='TEST')
# plt.xlabel("Polynomial Complexity")
# plt.ylabel("RMSE")
# plt.legend()
# plt.show()

# plt.plot(range(1,10),train_rmse_errors,label='TRAIN')
# plt.plot(range(1,10),test_rmse_errors,label='TEST')
# plt.xlabel("Polynomial Complexity")
# plt.ylabel("RMSE")
# plt.ylim(0,100)
# plt.legend()
# plt.show()

# Based on our chart, could have also been degree=4, but 
# it is better to be on the safe side of complexity
final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)
final_model = LinearRegression()
final_model.fit(final_poly_converter.fit_transform(X),y)
dump(final_model, 'sales_poly_model.joblib') 
dump(final_poly_converter,'poly_converter.joblib')

loaded_poly = load('poly_converter.joblib')
loaded_model = load('sales_poly_model.joblib')
# 149 TV, 22 radio, 12 newspaper
campaign = pd.DataFrame(data = [[149, 22, 12]], columns=['TV', 'radio', 'newspaper'])
campaign_poly = loaded_poly.transform(campaign)
print(campaign_poly)
final_model.predict(campaign_poly)
