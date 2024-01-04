import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df = pd.read_csv('DATA/cement_slump.csv')
print(df.head())

print(df.corr()['Compressive Strength (28-day)(Mpa)'])

sns.heatmap(df.corr(),cmap='viridis', annot=True)
# plt.show()

print(df.columns)

# clf = make_pipeline(StandardScaler(), SVR())

X = df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
y = df['Compressive Strength (28-day)(Mpa)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Setting C: C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it: decreasing C corresponds to more regularization.
# LinearSVC and LinearSVR are less sensitive to C when it becomes large, and prediction results stop improving after a certain threshold. Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer
# 
# Support Vector Machines - Regression
# 
base_model = SVR()
base_model.fit(scaled_X_train,y_train)
base_preds = base_model.predict(scaled_X_test)

# 
# Evaluation
# 
mean_absolute_error(y_test,base_preds)
np.sqrt(mean_squared_error(y_test,base_preds))
y_test.mean()

# 
# Grid Search in Attempt for Better Model
# 
param_grid = {'C':[0.001,0.01,0.1,0.5,1],
             'kernel':['linear','rbf','poly'],
              'gamma':['scale','auto'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.1,0.5,1,2]}
svr = SVR()
grid = GridSearchCV(svr,param_grid=param_grid)
grid.fit(scaled_X_train,y_train)
print('grid.best_params_: ', grid.best_params_)
grid_preds = grid.predict(scaled_X_test)
print('mean_absolute_error: ', mean_absolute_error(y_test,grid_preds))
print('mean_squared_error: ', np.sqrt(mean_squared_error(y_test,grid_preds)))