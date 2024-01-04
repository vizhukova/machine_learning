import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


df = pd.read_csv("DATA/rock_density_xray.csv")
df.columns=['Signal',"Density"]

print(df.head())

plt.figure(figsize=(12,8),dpi=200)
sns.scatterplot(x='Signal', y='Density', data=df)

# 
# Splitting the Data
# 

X = df['Signal'].values.reshape(-1,1)  
y = df['Density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# 
# Linear Regression
# 

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_preds = lr_model.predict(X_test)

print('mean_squared_error: ', np.sqrt(mean_squared_error(y_test,lr_preds)))

signal_range = np.arange(0,100)

lr_output = lr_model.predict(signal_range.reshape(-1,1))

plt.figure(figsize=(12,8),dpi=200)
sns.scatterplot(x='Signal',y='Density',data=df,color='black')
plt.plot(signal_range,lr_output)
plt.title('Linear Regression')

# 
# Polynomial Regression
# 

model = LinearRegression()
def run_model(model,X_train,y_train,X_test,y_test,title=''):
    
    # Fit Model
    model.fit(X_train,y_train)
    
    # Get Metrics
    
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    print(f'RMSE for {title} : {rmse}')
    
    # Plot results
    signal_range = np.arange(0,100)
    # reshape(-1, 1) is because we've got only 1 feature in the dataset
    output = model.predict(signal_range.reshape(-1,1))
    
    
    plt.figure(figsize=(12,6),dpi=150)
    sns.scatterplot(x='Signal',y='Density',data=df,color='black')
    plt.plot(signal_range,output)
    plt.title(title)

run_model(model,X_train,y_train,X_test,y_test, 'Polynomial Regression')

# 
# Pipeline for Poly Orders
# 

pipe = make_pipeline(PolynomialFeatures(2),LinearRegression())
run_model(pipe,X_train,y_train,X_test,y_test,'Pipeline for Poly Orders 2')


# 
# Comparing Various Polynomial Orders
# 

pipe = make_pipeline(PolynomialFeatures(10),LinearRegression())
run_model(pipe,X_train,y_train,X_test,y_test,'Pipeline for Poly Orders 10')

# 
# KNN Regression
# 

preds = {}
k_values = [1,5,10]
for n in k_values:  
    model = KNeighborsRegressor(n_neighbors=n)
    run_model(model,X_train,y_train,X_test,y_test, f'KNN Regression with {n}')

# 
# Decision Tree Regression
# 

model = DecisionTreeRegressor()
run_model(model,X_train,y_train,X_test,y_test,'Decision Tree Regression')
print('get_n_leaves: ', model.get_n_leaves())

# 
# Support Vector Regression
# 

param_grid = {'C':[0.01,0.1,1,5,10,100,1000],'gamma':['auto','scale']}
svr = SVR()
grid = GridSearchCV(svr,param_grid)
run_model(grid,X_train,y_train,X_test,y_test,'Support Vector Regression')
print('best_estimator_: ', grid.best_estimator_)

# 
# Random Forest Regression
# 

trees = [10,50,100]
for n in trees:
    model = RandomForestRegressor(n_estimators=n)
    run_model(model,X_train,y_train,X_test,y_test,f'Random Forest Regression with ${n}')

# 
# Gradient Boosting 
# 

model = GradientBoostingRegressor()
run_model(model,X_train,y_train,X_test,y_test,'Gradient Boosting')

# 
# Adaboost
# 

model = GradientBoostingRegressor()
run_model(model,X_train,y_train,X_test,y_test,'Adaboost')

plt.show()
