#We will take categorical data and turn it into features
#VectorizationL converting arbitray data into well behaved vectors

#Categorical Features
#housing price data
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

#will use one-hot encoding
#DictVectorizer can doi this for us

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
print(vec.fit_transform(data))
#can inspect the feature names
print(vec.get_feature_names())
#this can greatly increase the size of your dataset
#since most data will be 0 though sparce output can be efficient

vec = DictVectorizer(sparse=True, dtype=int)
print(vec.fit_transform(data))
#most estimators will accept sparse inputs

#Text Features
#commonly we need to convert text into a set of numerical values
#simple method is storing counts

sample = ['problem of evil',
          'evil queen',
          'horizon problem']
#CountVectorizer will take care of this for us

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
print(X) #sparse matric
#we can convert to a labled data frame for easier inspection
import pandas as pd
print(pd.DataFrame(X.toarray(), columns=vec.get_feature_names()))

#this can put too much weight on filler words
#can use a term frequency-inverse document frequency (TF-IDF)
#weighs word counts by a measuere of how often they appear in the document

from sklearn.feature_extraction.text import TfidfVectorizer
cev = TfidfVectorizer()
X = vec.fit_transform(sample)
print(pd.DataFrame(X.toarray(), columns=vec.get_feature_names()))
#See Niave bayes classification for more

import matplotlib
matplotlib.use("TKagg")
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);
plt.show()
plt.clf()

#this data clearly doesnt fit a stright line but you cna anyway
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit);
plt.show()
plt.clf()

#we can transform the data adding extra columns of features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2) # we added x^2 and x^3
#now we can fit a better linear regression
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x,y)
plt.plot(x, yfit)
plt.show()
plt.clf()
#will expand on in: In depth: linear regression
#Also known as kernal methods, in-depth: Support Vector machines

#Imputation of missing data
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])

#for basic imputation you can use mean median or mode, from the Imputer class
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
print(X2)
#now we can feed it into a model
model = LinearRegression().fit(X2, y)
print(model.predict(X2))


#Feature Pipelines
#you can streamline this with a pipeline object
from sklearn.pipeline import make_pipeline
model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())

model.fit(X, y)  #this X has the NaN values
print(y)
print(model.predict(X))
#more in in-depth: linear regression & support Vector Machines
