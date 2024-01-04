import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

df = pd.read_csv('DATA/hearing_test.csv')
print(df.head())
print('info: ', df.info())
print('describe:\n ', df.describe())
print('test_result value: ', df['test_result'].value_counts())

plt.title('test_result')
sns.countplot(data = df, x = 'test_result')

plt.figure()
sns.boxplot(x = 'test_result', y = 'age', data = df)

plt.figure()
sns.boxplot(x = 'test_result', y = 'physical_score', data = df)

plt.figure()
sns.scatterplot(x = 'age',y = 'physical_score',data = df,hue = 'test_result')

plt.figure()
sns.pairplot(data = df, hue = 'test_result')

plt.figure()
sns.heatmap(df.corr(),annot=True)

plt.figure()
sns.scatterplot(x='physical_score',y='test_result',data=df)

plt.figure()
sns.scatterplot(x='age',y='test_result',data=df)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'],df['physical_score'],df['test_result'],c=df['test_result'])
# plt.show()

X = df.drop('test_result',axis=1)
y = df['test_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(scaled_X_train,y_train)

print('coef_: ', log_model.coef_)
# array([[-0.94953524,  3.45991194]])
# This means:
# We can expect the odds of passing the test to decrease (the original coeff was negative) per unit increase of the age.
# We can expect the odds of passing the test to increase (the original coeff was positive) per unit increase of the physical score.
# Based on the ratios with each other, the physical_score indicator is a stronger predictor than age.

y_pred = log_model.predict(scaled_X_test)
y_pred_proba = log_model.predict_proba(scaled_X_test)
print('prediction: \n', y_pred)
print('prediction of probability: \n', y_pred_proba)

# 
# Clasification performance metrics
# 

print('accuracy_score: ', accuracy_score(y_test,y_pred))
# result: 0.93

print('confusion_matrix: ', confusion_matrix(y_test,y_pred))
# result [[172  21]
#  [ 14 293]]

cm = confusion_matrix(y_test, y_pred, labels=log_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=log_model.classes_)
disp.plot()

cm = confusion_matrix(y_test, y_pred, labels=log_model.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=log_model.classes_)
disp.plot()

print(classification_report(y_test,y_pred))

pr_disp = PrecisionRecallDisplay.from_estimator(log_model,scaled_X_test,y_test)
pr_disp.plot()

roc_disp = RocCurveDisplay.from_estimator(log_model,scaled_X_test,y_test)
roc_disp.plot()

# plt.show()

print(log_model.predict_proba(scaled_X_test)[0], y_test[0])
# [0.02384343 0.97615657]    1
#  it says that model predicts that it has 97% chance to belong class 1 and 23% change to belong the class 0