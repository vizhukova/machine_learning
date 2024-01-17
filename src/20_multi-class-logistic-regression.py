import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from plot_multiclass_roc import plot_multiclass_roc_func

df = pd.read_csv('DATA/iris.csv')

print(df.head())
print(df.info())
print(df.describe())

print("df['species'].value_counts(): \n", df['species'].value_counts())

sns.countplot(df['species'])

plt.figure()
sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')

plt.figure()
sns.scatterplot(x='petal_length',y='petal_width',data=df,hue='species')

plt.figure()
sns.pairplot(df,hue='species')

plt.figure()
# the closer value to the 1 - the biggest dependecy between columns
sns.heatmap(df.drop('species', axis=1).corr(),annot=True)

print('unique: ', df['species'].unique())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
ax.scatter(df['sepal_width'],df['petal_width'],df['petal_length'],c=colors)

# plt.show()

X = df.drop('species',axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

print('!!!!!!', X_train)
print('>>>>', scaled_X_train)

# Multi-Class Logistic Regression Model

# Depending on warnings you may need to adjust max iterations allowed 
# Or experiment with different solvers
log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)

# GridSearch for Best Hyper-Parameters
# Main parameter choices are regularization penalty choice and regularization C value.

# Penalty Type
penalty = ['l1', 'l2', 'elasticnet']
l1_ratio = np.linspace(0,1,20)

# Use logarithmically spaced C values (recommended in official docs)
C = np.logspace(0, 10, 20)

grid_model = GridSearchCV(log_model,param_grid={
    'C':C,
    'penalty':
    ['l1', 
    'l2', 
    # 'elasticnet' #takes some time
    ], 
    # 'l1_ratio':l1_ratio #needs for elasticnet
    })

grid_model.fit(scaled_X_train,y_train)

print('best_params_: \n', grid_model.best_params_)
# {'C': 11.28837891684689, 'l1_ratio': 0.0, 'penalty': 'l1'}
# 
# Model Performance on Classification Tasks
# 

y_pred = grid_model.predict(scaled_X_test)
print('y_pred', y_pred)
print('accuracy_score: ', accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print('confusion_matrix: ', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=grid_model.classes_)
disp.plot()

# Scaled so highest value=1
cm = confusion_matrix(y_test,y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=grid_model.classes_)
disp.plot()
# plt.show()

print(classification_report(y_test,y_pred))

# 
# Evaluating Curves and AUC
# 

plot_multiclass_roc_func(grid_model, scaled_X_test, y_test, n_classes=3, figsize=(16, 10))