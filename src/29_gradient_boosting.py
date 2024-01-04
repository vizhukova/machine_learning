import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,accuracy_score

df = pd.read_csv("DATA/mushrooms.csv")
print(df.head())

# 
# Data Prep
# 
X = df.drop('class',axis=1)
y = df['class']
X = pd.get_dummies(X,drop_first=True)
print(X.head())
print(y.head())

# 
# Train Test Split
# 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# 
# Gradient Boosting and Grid Search with CV
# 

param_grid = {"n_estimators":[1,5,10,20,40,100],'max_depth':[3,4,5,6]}
gb_model = GradientBoostingClassifier()
grid = GridSearchCV(gb_model,param_grid)

# 
# Fit to Training Data with CV Search
# 
grid.fit(X_train,y_train)
print('best_params_: ', grid.best_params_)

# 
# Performance
# 

predictions = grid.predict(X_test)
print("predictions: ", predictions)
print(classification_report(y_test,predictions))
print('feature_importances_: ', grid.best_estimator_.feature_importances_)
feat_import = grid.best_estimator_.feature_importances_
imp_feats = pd.DataFrame(index=X.columns,data=feat_import,columns=['Importance'])
print('imp_feats: \n', imp_feats)
print(imp_feats.sort_values("Importance",ascending=False))
print(imp_feats.describe().transpose())

imp_feats = imp_feats[imp_feats['Importance'] > 0.000527]

print(imp_feats.sort_values('Importance'))

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')
plt.xticks(rotation=90)
plt.show()