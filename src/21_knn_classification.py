import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('DATA/gene_expression.csv')
print(df.head())

sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df,alpha=0.7)
print(len(df))

plt.figure()
sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df)
plt.xlim(2,6)
plt.ylim(3,10)
plt.legend(loc=(1.1,0.5))

X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=1)

knn_model.fit(scaled_X_train,y_train)

y_pred = knn_model.predict(scaled_X_test)

accuracy_score(y_test,y_pred)

print('confusion_matrix: \n', confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

# 
# Elbow Method for Choosing Reasonable K Values
# 

test_error_rates = []


for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train) 
   
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)

print('test_error_rates: ', test_error_rates)
plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")

#
# Full Cross Validation Grid Search for K Value
#

# Step 0: The data are split into TRAINING data and TEST data according to the cv parameter that you specified in the GridSearchCV.
# Step 1: the scaler is fitted on the TRAINING data
# Step 2: the scaler transforms TRAINING data
# Step 3: the models are fitted/trained using the transformed TRAINING data
# Step 4: the scaler is used to transform the TEST data
# Step 5: the trained models predict using the transformed TEST data

scaler = StandardScaler()

knn = KNeighborsClassifier()

print('knn get_params: ', knn.get_params().keys())

# Highly recommend string code matches variable name!
operations = [('scaler',scaler),('knn',knn)]

pipe = Pipeline(operations)

k_values = list(range(1,20))

print('k_values: ', k_values)

param_grid = {'knn__n_neighbors': k_values}

full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')

# Use full X and y if you DON'T want a hold-out test set
# Use X_train and y_train if you DO want a holdout test set (X_test,y_test)
full_cv_classifier.fit(X_train,y_train)

GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                       ('knn', KNeighborsClassifier())]),
             param_grid={'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                              12, 13, 14, 15, 16, 17, 18, 19]},
             scoring='accuracy')

print('full_cv_classifier params: ', full_cv_classifier.best_estimator_.get_params())

print('full_cv_classifier keys: ', full_cv_classifier.cv_results_.keys())

print(len(k_values))

print('cv_results_: ', full_cv_classifier.cv_results_['mean_test_score'])

print(len(full_cv_classifier.cv_results_['mean_test_score']))

#
# Final Model
#

scaler = StandardScaler()
knn14 = KNeighborsClassifier(n_neighbors=14)
operations = [('scaler',scaler),('knn14',knn14)]

pipe = Pipeline(operations)

pipe.fit(X_train,y_train)

Pipeline(steps=[('scaler', StandardScaler()),
                ('knn14', KNeighborsClassifier(n_neighbors=14))])

pipe_pred = pipe.predict(X_test)

print(classification_report(y_test,pipe_pred))

single_sample = X_test.iloc[40]

print('single_sample: ', single_sample)

pipe.predict(single_sample.values.reshape(1, -1))

pipe.predict_proba(single_sample.values.reshape(1, -1))

plt.show()
