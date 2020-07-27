import numpy as np

np.random.seed(0) # seed for reproducibility

x1 = np.random.randint(10, size=6) #1d array
x2 = np.random.randint(10, size=(3,4)) #2d array
x3 = np.random.randint(10, size=(3,4,5)) #3d array

print("x3 ndim:", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size:", x3.size)

print("dytpe:", x3.dtype)

print("itemsize:", x3.itemsize,  "bytes")
print("nbytes:", x3.nbytes, "bytes")

print(x2[:,0]) # first column of x2
print(x2[0,:]) #first row of x2

print(x2[0]) # equivelent to x2[0, :]
print("x2:")
print(x2)

#views

x2_sub = x2[:2, :2]
print("x2 sub array:")
print(x2_sub)

x2_sub[0, 0] = 99 #this will change the parent array as it is a view, not a copy
print(x2_sub)
print(x2)

#copying arrays

x2_sub_copy = x2[:2,:2].copy()
print(x2_sub_copy)

x2_sub_copy[0,0] = 42
print(x2_sub_copy)
print(x2)

#reshaping arrays

grid = np.arange(1,10).reshape((3,3))
print(grid)

x = np.array([1,2,3])

print(" ")
#row vector via reshape
print(x.reshape((1,3)))

#row vector via newaxis
print(x[np.newaxis, :])

#column vector via reshape
print(x.reshape((3,1)))

#column vector from newaxis
print(x[:, np.newaxis])

#concatenation
x = np.array([1,2,3])
y = np.array([4,5,6])
print(np.concatenate([x,y]))

z = [7,8,9]

print(np.concatenate([x,y,z]))

grid = np.array([[1,2,3],[4,5,6]])
print()
print(grid)
print(np.concatenate([grid,grid]))

print(np.concatenate([grid, grid], axis=1))

grid = np.array([[9,8,7],[6,5,4]])
#vertically stack arrays
print(np.vstack([x,grid]))
print("horizontally stack arrays")
y = np.array([[99],[99]])
print(np.hstack([grid,y]))

#np.dstack goes along third axis

print("splitting arrays")
x = [1,2,3,4,5,6,7,8]
print(x)
x1, x2, x3 = np.split(x, [3,5])
print("split [3,5]")
print(x1)
print(x2)
print(x3)

#hsplit, vsplit simmilar

grid = np.arange(16).reshape((4,4))
print(grid)
upper, lower = np.vsplit(grid, [2])
print("vsplit [2]")
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print("hsplit [2]")
print(left)
print(right)
