import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

df = pd.read_csv("DATA/Advertising.csv")
print(df.head())

## CREATE X and y
X = df.drop('sales',axis=1)
y = df['sales']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SCALE DATA
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha = 100)

# SCORING OPTIONS:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_validate(model,X_train,y_train,
                         scoring=['neg_mean_absolute_error','neg_mean_squared_error','max_error'],cv=5)

df_scores = pd.DataFrame(scores)
print('scores: ', df_scores)
print('mean score: \n', abs(df_scores.mean()))

# Adjust model based on metrics

model = Ridge(alpha=1)
scores = cross_validate(model, X_train, y_train,
                         scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error'], cv=5)
df_scores = pd.DataFrame(scores)
print('scores: ', df_scores)
print('mean score: \n', abs(df_scores.mean()))

# Final Evaluation (Can no longer edit parameters after this!)

model.fit(X_train,y_train)

y_final_test_pred = model.predict(X_test)

print('result: ', mean_squared_error(y_test,y_final_test_pred))