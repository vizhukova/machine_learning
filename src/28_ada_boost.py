import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,accuracy_score


df = pd.read_csv("DATA/mushrooms.csv")
print(df.head())
print(df.info())
print(df.describe().transpose())

sns.countplot(data=df,x='class')

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=df.describe().transpose().reset_index().sort_values('unique'),x='index',y='unique')
plt.xticks(rotation=90)

# 
# Train Test Split¶
# 

X = df.drop('class',axis=1)
X = pd.get_dummies(X,drop_first=True)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# 
# Modeling
# 

model = AdaBoostClassifier(n_estimators=1)
model.fit(X_train,y_train)

# 
# Evaluation
# 

predictions = model.predict(X_test)
print('predictions: ', predictions)
print(classification_report(y_test,predictions))
print('feature_importances_: ', model.feature_importances_)
print('argmax: ', model.feature_importances_.argmax())
print('the 22 item: ', X.columns[22])

plt.figure()
sns.countplot(data=df,x='odor',hue='class')

# 
# Analyzing performance as more weak learners are added
# 

print('len(X.columns): ', len(X.columns))

error_rates = []

for n in range(1,96):
    
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    err = 1 - accuracy_score(y_test,preds)
    
    error_rates.append(err)

plt.figure()
plt.plot(range(1,96),error_rates)

print('feature_importances_: ', model.feature_importances_)

feats = pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Importance'])

print('feats: ', feats)

imp_feats = feats[feats['Importance']>0]

print('imp_feats: ', imp_feats)

imp_feats = imp_feats.sort_values("Importance")

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')
plt.xticks(rotation=90)

plt.figure()
sns.countplot(data=df,x='habitat',hue='class')

plt.show()