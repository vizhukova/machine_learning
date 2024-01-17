import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("DATA/Advertising.csv")
print(df.head())
X = df.drop('sales',axis=1)
y = df['sales']

polynomial_converter = PolynomialFeatures(degree=3,include_bias=False)
poly_features = polynomial_converter.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 
# Ridge Regression
# 

# alpha is a penalty param if model is making a mistake during the learning
# in the meanwhile we chose it randomly, later it can be chosen with cross validation method
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train, y_train)
test_predictions = ridge_model.predict(X_test)

MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

# runs cross validation for the variety of the parameters
# Not all scoring can be working with the regression tasks, some for the classification etc
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')

# The more alpha options is passed, the longer this will take.
# Fortunately data set is still pretty small
ridge_cv_model.fit(X_train,y_train)

# Returns which alpha performed the best:
print('alpha: ', ridge_cv_model.alpha_)

test_predictions = ridge_cv_model.predict(X_test)

MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

print('coefficients: ', ridge_cv_model.coef_)
# result is array([ 
# 5.40769392,  0.5885865 ,  0.40390395, -6.18263924,  4.59607939,
#    -1.18789654, -1.15200458,  0.57837796, -0.1261586 ,  2.5569777 ,
#    -1.38900471,  0.86059434,  0.72219553, -0.26129256,  0.17870787,
#     0.44353612, -0.21362436, -0.04622473, -0.06441449])
# considering on every feature

print('best score: ', ridge_cv_model.best_score_)

# 
# Lasso Regression
# 

# perfoming cross validation to get the best alpha param
# with the change of eps it can be consider more features during model trail - with better performance MAE RMSE
# the model will become more complex the, but with better results
#  eps = 0.01 returns almost same results as Ridge Regression with the only usage of half features
lasso_cv_model = LassoCV(eps = 0.1,n_alphas = 100,cv = 5, max_iter = 1000000)
lasso_cv_model.fit(X_train,y_train)
# Returns which alpha performed the best:
print('alpha: ', lasso_cv_model.alpha_)

test_predictions = lasso_cv_model.predict(X_test)

MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

print('coefficients: ', lasso_cv_model.coef_)
# result is array([
# 1.002651  , 0.        , 0.        , 0.        , 3.79745279,
#    0.        , 0.        , 0.        , 0.        , 0.        ,
#    0.        , 0.        , 0.        , 0.        , 0.        ,
#    0.        , 0.        , 0.        , 0.        ])
# considering only on 2 features but it did come with the cost of worst performing modes based on MAE RMSE

#
# 9
#

elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
elastic_model.fit(X_train,y_train)
print('ratio: ', elastic_model.l1_ratio_)
test_predictions = elastic_model.predict(X_test)

MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)
print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

# Training Set Performance
# Training Set Performance
train_predictions = elastic_model.predict(X_train)
MAE = mean_absolute_error(y_train,train_predictions)
print('MAE: ', MAE)

print('coefficients: ', elastic_model.coef_)