#example, avg height of us presidents 

import numpy as np
import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

print("25th percentile:  ", np.percentile(heights, 25))
print("median:           ", np.median(heights))
print("75th percentaile: ", np.percentile(heights, 75))

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #set plot style

plt.hist(heights)
plt.title('Height Distrobution of US Presidents')
plt.xlabel('height(cm)')
plt.ylabel('number')
plt.show()
