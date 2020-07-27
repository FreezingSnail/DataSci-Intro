import seaborn as sns
iris = sns.load_dataset('iris')
print(iris.head())
sns.set()
import matplotlib
matplotlib.use("TKagg")
import matplotlib.pyplot as plt
#sns.pairplot(iris, hue='species', height=1.5); #slow plot
X_iris = iris.drop('species', axis=1)
print(X_iris.shape)
y_iris = iris['species']

##Supervised learning example
#linear regression

import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);

#step 1, choose class model
from sklearn.linear_model import LinearRegression

#step 2. Choose model hyperparameters
model = LinearRegression(fit_intercept=True)
print(model)

#Step 3. Arrange data into a features matrix and target vector
X = x[:, np.newaxis] #transform from (50,) to (50,1)

#Step 4. Fit the model to your data
model.fit(X,y)
#now we have some computations completed for internal params of the model
print(model.coef_)
print(model.intercept_)

#step 5. Predict labels for unknown data
xfit = np.linspace(-1,11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)



##Supervised learning example:Iris classification
#will use Gausian naive Bayes

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB #1. Choose model
model = GaussianNB()                       #2. Instantiate model
model.fit(Xtrain, ytrain)                  #3. fit model to data
y_model = model.predict(Xtest)             #4. Predict on new data

#can check accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))

#Unsurpervised learning example: Iris dimensionality
#reduce the 4d iris data to two 2d sets
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions

#plot the data
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);

#Unsupervised learning: Iris Clustering
#group the data into gaussian blobs
from sklearn import mixture                  # 1. Choose the model class
model = mixture.GaussianMixture(n_components=3, covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)                        # 4. Determine cluster labels

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False);



plt.show()


