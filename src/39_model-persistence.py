import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib

df = pd.read_csv('DATA/Advertising.csv')

print(df.head())
print(df.info())
print(df.describe().transpose())

# Data Preparation

X = df.drop('sales',axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Further split 30% of test into validation and hold-out (15% and 15% each)
X_validation, X_holdout_test, y_validation, y_holdout_test = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

# Model Training

model = RandomForestRegressor(n_estimators=10,random_state=101)
model.fit(X_train,y_train)

# Model Evaluation

validation_predictions = model.predict(X_validation)
print('with 10 MAE: ', mean_absolute_error(y_validation, validation_predictions))
print('with 10 RMSE: ', mean_squared_error(y_validation,validation_predictions)**0.5)

# Hyperparameter Tuning

model = RandomForestRegressor(n_estimators=35,random_state=101)
model.fit(X_train,y_train)

validation_predictions = model.predict(X_validation)
print('with 35 MAE: ', mean_absolute_error(y_validation, validation_predictions))
print('with 35 RMSE: ', mean_squared_error(y_validation,validation_predictions)**0.5)

# Final Hold Out Test Performance for Reporting

model = RandomForestRegressor(n_estimators=35,random_state=101)
model.fit(X_train,y_train)

test_predictions = model.predict(X_holdout_test)

print('final MAE: ', mean_absolute_error(y_holdout_test, test_predictions))
print('final RMSE: ', mean_squared_error(y_holdout_test, test_predictions)**0.5)

# Full Training

final_model = RandomForestRegressor(n_estimators=35,random_state=101)
final_model.fit(X,y)

# Saving Model (and anything else as pickle file)¶

joblib.dump(final_model,'final_model.pkl')
print('X.columns: ', X.columns)
print('list X.columns: ', list(X.columns))
joblib.dump(list(X.columns),'column_names.pkl')

# Loading Model (Model Persistence)

col_names = joblib.load('column_names.pkl')
print('col_names: ', col_names)
loaded_model = joblib.load('final_model.pkl')
print('predicted: ', loaded_model.predict([[230.1,37.8,69.2]]))