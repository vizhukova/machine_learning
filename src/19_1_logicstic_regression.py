from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('DATA/titanic.csv')

print(df.head())

X = df[['Pclass', 'SibSp','Parch', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression(random_state=0, solver = 'lbfgs')
model.fit(X, y)
preds = model.predict(X)

print(preds)

cm = confusion_matrix(y, preds, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()
plt.show()