from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

with open('One.txt') as mytext:
    # a = mytext.read()
    lines = mytext.readlines()

with open('Two.txt') as mytext:
    b = mytext.read()
    words = b.lower().split()

# print('full text a: ', a)
print('full text b: ', b)
print('lines a: ', lines)
print('words b: ', words)

# Building a vocabulary (Creating a "Bag of Words")

with open('One.txt') as f:
    words_one = f.read().lower().split()
    uni_words_one = set(words_one)


print('words_one: ', words_one)
print('len: ', len(words_one))
print('uni_words_one: ', uni_words_one)

with open('Two.txt') as f:
    words_two = f.read().lower().split()
    uni_words_two = set(words_two)

# Get all unique words across all documents

all_uni_words = set()
all_uni_words.update(uni_words_one)
all_uni_words.update(uni_words_two)

print('all_uni_words: ', all_uni_words)

full_vocab = dict()
i = 0
for word in all_uni_words:
    full_vocab[word] = i
    i = i+1

print('full_vocab: ', full_vocab)

# Bag of Words to Frequency Counts

# Create an empty vector with space for each word in the vocabulary:
one_freq = [0]*len(full_vocab)
two_freq = [0]*len(full_vocab)
all_words = ['']*len(full_vocab)

print('one_freq: ', one_freq)
print('two_freq: ', two_freq)
print('all_words: ', all_words)

for word in full_vocab:
    word_ind = full_vocab[word]
    all_words[word_ind] = word

print('all_words: ', all_words)

# map the frequencies of each word in 1.txt to our vector:
with open('One.txt') as f:
    one_text = f.read().lower().split()
    
for word in one_text:
    word_ind = full_vocab[word]
    one_freq[word_ind] += 1

print('one_freq: ', one_freq)

# Do the same for the second document:
with open('Two.txt') as f:
    two_text = f.read().lower().split()
    
for word in two_text:
    word_ind = full_vocab[word]
    two_freq[word_ind]+=1

print('two_freq: ', two_freq)

pd.DataFrame(data=[one_freq,two_freq],columns=all_words)

# Scikit-Learn's Text Feature Extraction Options

text = ['This is a line',
        "This is another line",
        "Completely different line"]

cv = CountVectorizer()
cv.fit_transform(text)
sparse_mat = cv.fit_transform(text)
print('todense: ', sparse_mat.todense())
print('vocabulary_: ', cv.vocabulary_)

cv = CountVectorizer(stop_words='english')
cv.fit_transform(text).todense()
print('vocabulary_: ', cv.vocabulary_)

# TfidfTransformer
# TfidfVectorizer is used on sentences, while TfidfTransformer is used on an existing count matrix, such as one returned by CountVectorizer

tfidf_transformer = TfidfTransformer()
cv = CountVectorizer()
counts = cv.fit_transform(text)
print('counts: ', counts)
tfidf = tfidf_transformer.fit_transform(counts)
tfidf.todense()

pipe = Pipeline([('cv',CountVectorizer()),('tfidf',TfidfTransformer())])
results = pipe.fit_transform(text)
print('results: ', results)

results.todense()

# TfIdfVectorizer
# Does both above in a single step!
tfidf = TfidfVectorizer()
new = tfidf.fit_transform(text)
new.todense()



