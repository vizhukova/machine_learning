import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay,classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv("DATA/airline_tweets.csv")
print(df.head())
print(df.info())
print(df.columns)

sns.countplot(data=df, x='airline', hue='airline_sentiment')

plt.figure()
sns.countplot(data=df,x='negativereason')
plt.xticks(rotation=90)

plt.figure()
sns.countplot(data=df,x='airline_sentiment')

print(df['airline_sentiment'].value_counts())

# Features and Label
data = df[['airline_sentiment','text']]
print(data.head())

y = df['airline_sentiment']
X = df['text']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# DO NOT USE .todense() for such a large sparse matrix!!!

# Model Comparisons - Naive Bayes,LogisticRegression, LinearSVC
nb = MultinomialNB()
nb.fit(X_train_tfidf,y_train)

log = LogisticRegression(max_iter=1000)
log.fit(X_train_tfidf,y_train)

svc = LinearSVC()
svc.fit(X_train_tfidf,y_train)

# Performance Evaluation

def report(model, title = ''):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test,preds))
    cm = confusion_matrix(y_test,preds)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    disp.plot()
    plt.title(title)

report(nb, 'NB MODEL')
report(log, 'Logistic Regression')
report(svc, 'SVC')

# Finalizing a PipeLine for Deployment on New Tweets
pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
pipe.fit(df['text'],df['airline_sentiment'])
new_tweet = ['good flight']
print('good: ', pipe.predict(new_tweet))
new_tweet = ['bad flight']
print('bad: ', pipe.predict(new_tweet))
new_tweet = ['ok flight']
print('neutral: ', pipe.predict(new_tweet))

plt.show()