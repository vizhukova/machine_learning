import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("DATA/Advertising.csv")
print(df.head())

# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Test for both X and y
# 2. Fit/Train Scaler on Training X Data
# 3. Scale X Test Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6

## CREATE X and y
X = df.drop('sales', axis = 1)
y = df['sales']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SCALE DATA
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Poor Alpha Choice on purpose!
model = Ridge(alpha=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('y_pred test', y_pred)

# evaluation
print('mean_squared_error: ', mean_squared_error(y_test,y_pred))

# Adjust Parameters and Re-evaluate
model = Ridge(alpha=1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('y_pred train', y_pred)

# Another Evaluation
print('mean_squared_error: ', mean_squared_error(y_test,y_pred))

# 
# Train | Validation | Test Split Procedure
# 

# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Validation/Test for both X and y
# 2. Fit/Train Scaler on Training X Data
# 3. Scale X Eval Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Evaluation Data (by creating predictions and comparing to Y_eval)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6
# 8. Get final metrics on Test set (not allowed to go back and adjust after this!)

## CREATE X and y
X = df.drop('sales',axis=1)
y = df['sales']

######################################################################
#### SPLIT TWICE! Here we create TRAIN | VALIDATION | TEST  #########
####################################################################

# 70% of data is training data, set aside other 30%
X_train, X_OTHER, y_train, y_OTHER = train_test_split(X, y, test_size=0.3, random_state=101)

# Remaining 30% is split into evaluation and test sets
# Each is 15% of the original data size
X_eval, X_test, y_eval, y_test = train_test_split(X_OTHER, y_OTHER, test_size=0.5, random_state=101)

# SCALE DATA
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

# Poor Alpha Choice on purpose!
model_one = Ridge(alpha=100)
model_one.fit(X_train,y_train)
y_eval_pred = model_one.predict(X_eval)
print('y_pred eval', y_eval_pred)

# Evaluation

print('evaluation X_eval: ', mean_squared_error(y_eval,y_eval_pred))

# Adjust Parameters and Re-evaluate

model_two = Ridge(alpha=1)
model_two.fit(X_train,y_train)
new_y_eval_pred = model_two.predict(X_eval)
print('y_pred eval X_eval', y_eval_pred)

#Another Evaluation

print('evaluation y_eval_pred: ', mean_squared_error(y_eval,y_eval_pred))

# Final Evaluation (Can no longer edit parameters after this!)

y_final_test_pred = model_two.predict(X_test)
print('evaluation X_test: ', mean_squared_error(y_test,y_final_test_pred))

# SCALE DATA
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)

# SCORING OPTIONS:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_val_score(model,X_train,y_train,
                         scoring='neg_mean_squared_error',cv=5)

print('scores: ', scores)
# Average of the MSE scores (we set back to positive)
abs(scores.mean())

# 
# Adjust model based on metrics
# 

model = Ridge(alpha=1)
# SCORING OPTIONS:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_val_score(model,X_train,y_train,
                         scoring='neg_mean_squared_error',cv=5)
            
# Average of the MSE scores (we set back to positive)
abs(scores.mean())

#Final Evaluation (Can no longer edit parameters after this!)
# Need to fit the model first!
model.fit(X_train,y_train)

y_final_test_pred = model.predict(X_test)

mean_squared_error(y_test,y_final_test_pred)