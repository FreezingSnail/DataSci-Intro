import numpy as np
np.random.seed(0)
L = np.random.random(100)
print(np.sum(L))
big_array = np.random.random(1000000)
#%timeit sum(big_array)
#timeit np.sum(big_array)

#sum != np.sum

#min and max
print(min(big_array))
print(max(big_array))
print((np.min(big_array), np.max(big_array)))
#shorter syntax
print(big_array.sum(), big_array.min(), big_array.max())

#multi dimensional aggragates
M = np.random.random((3,4))
print(M)

print(M.sum())
print(M.min(axis=0)) #min in each row
print(M.min(axis=1)) #min in each collum

#axis is the dimension collapsed not returned
#hence axis=0 collapses along the first dimension, aggregating each collum in a 2d array


#example, avg height of us presidents 

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
