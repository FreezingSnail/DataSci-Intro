#Exploring fancy indexing
import numpy as np
rand = np.random.RandomState(42)

x = rand.randint(100, size=10)
print(x)

ind = [3,7,4]
print(x[ind])

#shape of the result reflects the indexing array
ind = np.array([[3,7],[4,5]])
print("x:", x)
print(x[ind])

#this works in multiple dimesnions
X = np.arange(12).reshape((3,4))
print("X:",X)

row = np.array([0,1,2])
col = np.array([2,1,3])
print("X[row,col]:", X[row,col])
#prints (0,2), (1,1), (2,3)

print(X[row[:, np.newaxis], col])
print(row[:, np.newaxis] * col)

#combined indexing

print("X:",X)
#combining fancy and simple indicies
print(X[2, [2,0,1]])
print(X[1:, [2,0,1]])
print("we can combine fancy indices with masking as well")
mask = np.array([1,0,1,0], dtype=bool)
print(X[row[:, np.newaxis], mask])

print("Example: selecting random points")
mean = [0,0]
cov = [[1,2],[2,5]]
X= rand.multivariate_normal(mean, cov, 100)
print(X.shape) #(100,2)

import matplotlib
matplotlib.use("TKagg")
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # for plot styling

plt.scatter(X[:, 0],X[:,1])
plt.show()

#we can use fancy indixing to select 20 random points

indicies = np.random.choice(X.shape[0], 20, replace=False)
print(indicies)
selection = X[indicies] #fancy indexing
print(selection.shape) #(20,2)

plt.scatter(selection[:,0], selection[:,1], facecolor='red', s=200)
plt.scatter(X[:,0],X[:,1], alpha=0.3)
plt.show()

print("#Modifying values with fancy indexing")

x = np.arange(10)
i = np.array([2,1,8,4])
x[i] = 99
print(x)
x[i] -=10
print(x)

x = np.zeros(10)
x[[0,0]] = [4,6]
print(x)
i = [2,3,3,4,4,4]
x[i] +=1
print(x)

x = np.zeros(10)
np.add.at(x, i, 10)
print(x)


print("Example: Binnng Data")
np.random.seed(42)
x=np.random.randn(100)

#compute a histogram by hand
bins = np.linspace(-5,5,20)
counts = np.zeros_like(bins)

#find the appropriate bin for each x
i = np.searchsorted(bins,x)

#add 1 to each of these bins
np.add.at(counts, i, 1)

#plot the results
plt.plot(bins, counts, linestyle='steps')
plt.show()

#or you can use plt.hist()

plt.hist(x, bins, histtype='step')
plt.show()
