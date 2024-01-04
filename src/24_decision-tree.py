import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.tree import plot_tree

df = pd.read_csv("DATA/penguins_size.csv")
print(df.head())
print('info: ', df.info())
print('empty values: \n', df.isna().sum())
print('What percentage should be dropped due to na: ', 100*(10/344))
df = df.dropna()
print('info after the drop: ', df.info())
print(df.head())
print('sex: ', df['sex'].unique())
print('island: ', df['island'].unique())
print('species: ', df['species'].unique())

# we can remove value with mistaken sex  
# df = df[df['sex']!='.']
# or assume the sex by our own
# by descripe it's more like a female (but not 100% sure)
print(df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose())
df.at[336, 'sex'] = 'FEMALE'
print(df.loc[336])

#
# Visualizations
#

sns.scatterplot(x='culmen_length_mm',y='culmen_depth_mm',data=df,hue='species',palette='Dark2')

plt.figure()
sns.pairplot(df,hue='species',palette='Dark2')

plt.figure()
sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',col='sex',palette='Dark2')

# plt.show()

print(pd.get_dummies(df.drop('species', axis=1), drop_first=True))

# 
# Train | Test Split
# 

X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# As we are compare dataset on a single feature that's why we do not need scaling
# as we won't use multiple features at the same time (doesn't matter if one feature is a way scaled than the other)

# 
# Decision Tree Classifier
# 

# Default Hyperparameters

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
base_pred = model.predict(X_test)

# 
# Evaluation
# 

cm = confusion_matrix(y_test,base_pred)
print(cm)
plt.figure()
# how many pinguines were sucessfuly clasified and how many were messed up:
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()

# how successfully it defined true positive values:
print('classification_report: ', classification_report(y_test,base_pred))

print('feature_importances_: ', model.feature_importances_)
# shows how much depends prediction on a features
print(pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['Feature Importance']).sort_values('Feature Importance'))

plt.figure()
sns.boxplot(x='species',y='body_mass_g',data=df)

# 
# Visualize the Tree
# 

plt.figure(figsize=(12,8),dpi=150)
plot_tree(model,filled=True,feature_names=X.columns)

# 
# Reporting Model Results
# 

def report_model(model, name = ''):
    model_preds = model.predict(X_test)
    print(name, classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=150)
    if (name):
        plt.title(name)
    plot_tree(model,filled=True,feature_names=X.columns)

# 
# Understanding Hyperparameters (Max Depth)
# 

pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train,y_train)
report_model(pruned_tree, 'Pruned Tree: \n')

# 
# Max Leaf Nodes
# 

pruned_tree = DecisionTreeClassifier(max_leaf_nodes=3)
pruned_tree.fit(X_train,y_train)
report_model(pruned_tree)

# 
# Criterion
# 

entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train,y_train)
report_model(entropy_tree)

plt.show()