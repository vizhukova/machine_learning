import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC # Supprt Vector Classifier
from svm_margin_plot import plot_svm_boundary
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("DATA/mouse_viral_study.csv")
print(df.head())

print('columns: ', df.columns)

sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',
                data=df,palette='seismic')

# Separating Hyperplane¶     

sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',palette='seismic',data=df)

# We want to somehow automatically create a separating hyperplane ( a line in 2D)

x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
plt.plot(x,y,'k')

y = df['Virus Present']
X = df.drop('Virus Present',axis=1) 

model = SVC(kernel='linear', C=1000)
model.fit(X, y)
plt.figure()
plot_svm_boundary(model,X,y)

model = SVC(kernel='linear', C=0.05)
model.fit(X, y)
plt.figure()
plot_svm_boundary(model,X,y)

model = SVC(kernel='rbf', C=1) # here is used gamma: 'scaled' as default
model.fit(X, y)
plt.figure()
plot_svm_boundary(model,X,y)

model = SVC(kernel='rbf', C=1, gamma = 'auto') 
model.fit(X, y)
plt.figure()
plot_svm_boundary(model,X,y)

model = SVC(kernel='rbf', C=1, gamma = 1) # auto value for gamma is 0.5 (1/n_features)
# for the extreme nimber of gamma margin is picking up too much information and breaking itself 
# for example gamma = 2
# scale here is a good choice as it balancing the bias 
model.fit(X, y) 
plt.figure()
plot_svm_boundary(model,X,y)

model = SVC(kernel='sigmoid') 
model.fit(X, y)
plt.figure()
# almost all the points were higlighted as a support vectors, so for that dataset it's bad choice
plot_svm_boundary(model,X,y)

model = SVC(kernel='poly', C=1, degree=3) 
model.fit(X, y)
plt.figure()
# curves the margin diagonal
plot_svm_boundary(model,X,y)

svm = SVC()
param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}
grid = GridSearchCV(svm,param_grid)

grid.fit(X,y)

GridSearchCV(estimator=SVC(),
             param_grid={'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']})

print('grid.best_score_: ', grid.best_score_)
print('grid.best_params_: ', grid.best_params_)

model = SVC(kernel='linear', C=0.01) 
model.fit(X, y)
plt.figure()
plt.title('The perfect one')
plot_svm_boundary(model,X,y)

plt.show()
