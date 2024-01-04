import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay

df = pd.read_csv("DATA/penguins_size.csv")
df = df.dropna()

print(df.head())

#
# Train | Test Split
#

X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# 
# Random Forest Classification
# 

# Use 10 random trees
model = RandomForestClassifier(n_estimators=10,max_features='sqrt',random_state=101)
model.fit(X_train,y_train)
preds = model.predict(X_test)

print('preds: \n', preds)

# 
# Evaluation
# 

cm = confusion_matrix(y_test,preds)
print('confusion_matrix: \n', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()

#
# Feature Importance
#

print('feature_importances_: ', model.feature_importances_)
print(pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['Feature Importance']).sort_values('Feature Importance'))

#
# Choosing correct number of trees
#

test_error = []

for n in range(1,40):
    # Use n random trees
    model = RandomForestClassifier(n_estimators=n,max_features='sqrt')
    model.fit(X_train,y_train)
    test_preds = model.predict(X_test)
    test_error.append(1-accuracy_score(test_preds,y_test))

plt.figure()
plt.plot(range(1,40),test_error,label='Test Error')
plt.legend()

plt.show()