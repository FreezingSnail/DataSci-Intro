import numpy as np

def selection_sort(x):
    """n^2 sort"""
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array([2,1,4,3,5])
selection_sort(x)
print(x)

def bogosort(x):
    """n*n! sort"""
    while np.any(x[:-1] >x[1:]):
        np.random.shuffle(x)
    return(x)

x = np.array([2,1,4,3,5])
bogosort(x)
print(x)
print("#fast sorting in Numpy")
x = np.array([2,1,4,3,5])
print(np.sort(x)) #quicksort
print(x)
x.sort() # sorts array in place
print(x)
x = np.array([2,1,4,3,5])
i = np.argsort(x)
print(i) #indicies of members in sorted order
print(x[i]) #prints the array in order of the sorted indicies

print("##Sorting along rows or columns")
rand = np.random.RandomState(42)
X = rand.randint(0,10, (4,6))
print(X)

#sort each column of X
print(np.sort(X, axis=0))
#sort each row
print(np.sort(X, axis=1))
#this treats each vector on the selected axis independantly

print("#Partial Sorts: Partitioning")
x = np.array([7,2,3,1,6,5,4])
print(np.partition(x,3))

print(np.partition(X, 2, axis=1))
#np.argpartition gives indicies to partion elements

print("#Example: k-Nearest Neighbors")
X = rand.rand(10,2)

import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #plot styling
plt.scatter(X[:,0], X[:,1], s=100)
plt.show()


print("compute the distance between each pair")
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=1)
#or in its components
differences = X[:, np.newaxis,:] - X[np.newaxis, :,:]
print(differences.shape) #(10,10,2)
#square the coordinate differences
sq_differences = differences ** 2
#sum the coordinate differences to get squared distance
dist_sq = sq_differences.sum(-1)
print(dist_sq.shape) #(10,10)

print(dist_sq.diagonal()) #should be all 0's

nearest = np.argsort(dist_sq, axis=1)
print(nearest)

#for only the k-nearest
K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)

plt.scatter(X[:,0], X[:,1], s=100)

#draw lines from each point to its K nearest neighbors
K = 2
for i in (range(X.shape[0])):
    for j in nearest_partition[i, :K+1]:
        #plot a line form X[i] tp X[j]
        #using some zip magic
        plt.plot(*zip(X[j], X[i]), color='black')
plt.show()
