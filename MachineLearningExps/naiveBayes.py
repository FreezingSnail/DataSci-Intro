#group of fast and simple classification algorithms
#good for high-dimensional data
#quick and ditry baselines for classification

#Bayesian Classification

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TKagg")
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#Gaussian Niave Bayes

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
plt.show()
plt.clf()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6,-14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

#now we can plot the new data
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);
plt.show()
plt.clf()

#we can predict probabilities with predict_prob
#prob depreciated
#yprob = model.predict_prob(Xnew)
#print(yprob[-8:].round(2))

#Multinomial Naive Bayes
###Example, classifying text

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data.target_names)

categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
#example from the data
print(train.data[5])

#we need to convert the text to a vector of numbers
#we will use TF-IDF vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

#lets check the confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
plt.clf()

def predict_category(s, train=train, model=model):
    """returns the prediction for a single string"""
    pred = model.predict([s])
    return train.target_names[pred[0]]

#try it out
print(predict_category('sending a payload to the ISS'))
print(predict_category('determining the screen resolution'))
print(predict_category('discussing islam vs atheism')) # wrong
