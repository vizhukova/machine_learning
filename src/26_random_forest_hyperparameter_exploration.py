import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv("DATA/data_banknote_authentication.csv")
print(df.head())

sns.pairplot(df,hue='Class')

X = df.drop("Class",axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

n_estimators = [64, 100, 128, 200]
max_features = [2,3,4]
bootstrap = [True, False]
oob_score = [True, False]

param_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
             'bootstrap': bootstrap,
             'oob_score': oob_score}  # Note, oob_score only makes sense when bootstrap=True!

rfc = RandomForestClassifier()
grid = GridSearchCV(rfc, param_grid)

grid.fit(X_train,y_train)

gb =  grid.best_params_
print('best_params_: \n', gb)

rfc = RandomForestClassifier(max_features = gb['max_features'], n_estimators = gb['n_estimators'], oob_score=True)
rfc.fit(X_train, y_train)
print('oob_score_: ', rfc.oob_score)

predictions = rfc.predict(X_test)

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print('confusion_matrix: \n', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=grid.classes_)
disp.plot()

# No underscore, reports back original oob_score parameter
print(grid.best_estimator_.oob_score)

# With underscore, reports back fitted attribute of oob_score
# print(grid.best_estimator_.oob_score)

# 
# Understanding Number of Estimators (Trees)
# 

errors = []
misclassifications = []

for n in range(1, 64):
    rfc = RandomForestClassifier( n_estimators = n, bootstrap = True, max_features = 2)
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    err = 1 - accuracy_score(preds, y_test)
    n_missed = np.sum(preds != y_test) # watch the video to understand this line!!
    errors.append(err)
    misclassifications.append(n_missed)
    
plt.figure()    
plt.plot(range(1,64), errors)

plt.figure()
plt.plot(range(1,64),misclassifications)

plt.show()