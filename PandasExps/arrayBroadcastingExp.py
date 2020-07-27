import numpy as np

#broadcasting allows applying binarry ufuncs to arrays of different sizes

a = np.array([0,1,2])
b = np.array([5,5,5])
print("a+b:",a+b)
print("a+5:", a+5)

M = np.ones((3,3))
print(M)
print("M+a\n", M+a)

a = np.arange(3)
b = np.arange(3) [:, np.newaxis]

print(a)
print(b)

print("a+b:\n", a+b)

M = np.ones((2,3)) #M.shape = (2,3)
a = np.arange(3)   #a.shape = (3, )
#by rule one of broadcasting a has fewer dimensions so its padded on the left
#a.shape -> (1,3)
#by rule 2 the first dimension dissagrees so its stretched
#a.shape -> (2,3)
#thus the final shape is (2,3)
print(M+a)
#exp2
a = np.arange(3).reshape((3,1)) #(3,1)
b = np.arange(3) #(3,)
#rule one pads b
#b.shape->(1,3)
#rule 2 stretches both arrays
#a.shape -> (3,3)
#b.shape -> (3,3)

print(a+b) # (3,3)

#exp3, not conpatable

M = np.ones((3,2)) #(3,2)
a = np.arange(3)   #(3, )
#rule one pads a
#a.shape -> (1,3)
#rule 2 streches a
#a.shape -> (3,3)

#rule 3 has missmatched dimensions, so the operation fails
#M + a  -> walue error, cant broadcast together

#right side padding
print(a[:, np.newaxis].shape)
print(M+a[:, np.newaxis]) #(3,2)

#these rules apply to all binary ufuncs
print(np.logaddexp(M, a[:, np.newaxis]))


##Broadcasting in practice
print("#center an array")

X = np.random.random((10,3))
Xmean = X.mean(0)
print(Xmean)

X_centered = X - Xmean

print(X_centered.mean(0))

##plotting a 2d function
#x & y have 50 steps from 0 to 5
x = np.linspace(0,5,50)
y = np.linspace(0,5,50)[:, np.newaxis]

z= np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

import matplotlib
matplotlib.use('TKagg')

import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()
plt.show()
