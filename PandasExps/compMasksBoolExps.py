##this will demonstrate boolean masks to access ada in numpy arrays

#counting rainydays exp.
import numpy as np
import pandas as pd

#pandas will read in our data
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0 #1/10mm -> inches
print(inches.shape) #(365,)

import matplotlib
matplotlib.use('TKagg')

import matplotlib.pyplot as plt
import seaborn; seaborn.set() #set plot style

plt.hist(inches, 40)
plt.show()

##digging into the data
#Comparison Operators as ufuncs

x = np.array([1,2,3,4,5])
print(x < 3) #less than, also have <, <=, !=, ==, ect

print((2*x) == (x ** 2))

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3,4))
print(x)
print(x<6)

#working with boolean arrays
print("x:\n", x)
#count # of true entries
print(np.count_nonzero(x < 6)) # how many values less that 6
print(np.sum(x < 6)) # same value another way but you can specify axis
print(np.sum(x < 6, axis=1))
print(np.any(x > 8)) # any values greater than 8?
print(np.all(x >2)) # are all values greater than 2?
#also works along axis
print(np.any(x<4, axis = 1))


##Bolean operators
# &, | , ^, ~
print(np.sum((inches > 0.5) & (inches < 1)))
# & equive to np.bitwise_and, ect

print("Number of days without rain:     ", np.sum(inches == 0))
print("number of days with rain:        ", np.sum(inches != 0))
print("Days with more than 0.5 inches:  ", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches:    ", np.sum((inches > 0) & (inches < 0.2)))

#Boolean arrays as masks
print("x:\n", x)

print(x[x<5]) #masking operation

#construct a mask of rainy days
rainy = (inches > 0)

#construct a mask of all summer days ( 6/21 is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("The median precip on rainy days in 2014 (inches):  ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):     ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches):    ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days(inches):    ",
      np.median(inches[rainy & ~summer]))
