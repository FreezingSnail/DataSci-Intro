import pandas as pd
import numpy as np

data = pd.Series([0.25,0.5,0.75,1.0])
print(data)
print(data.values)
print(data.index)
print(data[1])
print(data[1:3])
print("#series as general numpy arrays")

data = pd.Series([0.25,0.5,0.75,1.0], index=['a','b','c','d'])
print(data)
print(data['b'])
#you can use non sequential indicies
data = pd.Series([0.25,0.5,0.75,1.0], index=[2,5,1,7])
print(data)

print("#Series as specialized dictionary")
#series can be built directly from python dictionaries
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population)
print(population['California'])
#series support slicing unlike dictionarys
print(population['California':'Illinois'])

print("##Constructing series object")
#all constructions so far have been a vision of
#pd.Series(data, index=index)
#where index is an optional arg

print(pd.Series([2,4,6]))
#data can be a scalar stretched to indicies
print(pd.Series(5, index=[1,2,3]))
#data can be a dictionary
#index can be explicitly set
print(pd.Series({2:'a',1:'b',3:'c'}, index=[3,2]))

#Pandas DataFrame obj

print("##DataFrame as a generalized numpy array")
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
print(area)

states = pd.DataFrame({'population':population,'area':area})
print(states)
print(states.index)
print(states.columns)

print("##DataFrame as a specialized dictionary")
print(states['area'])
print("##Constructing DataFrame objs")
#from single series obj
print(pd.DataFrame(population, columns=['population']))

#from list of dicts
data = [{'a':i, 'b':2 * i} for i in range(3)]
print(pd.DataFrame(data))
#will fill in missing data with NaN
print(pd.DataFrame([{'a':1, 'b':2}, {'b':3, 'c':4}]))

#from dictionary of series objs as before
#pd.DataFrame({'population':population,'area':area})

#from 2d numpy array
x = pd.DataFrame(np.random.rand(3,2), columns=['foo', 'bar'], index=['a','b','c'])
print(x)

##from numpy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
print(A)
print(pd.DataFrame(A))

print("#Pandas index obj")
#index object is alike to an immutable array/ ordered set
ind = pd.Index([2,3,5,7,11])
print(ind)

##index as immutable array
print(ind[1])
print(ind[::2])
print(ind.size, ind.shape, ind.ndim, ind.dtype)

#ind[1] = 0 throws typeerror, not mutable

print("#Index as ordered set")
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])
print(indA & indB) #intersection
print(indA | indB) #union
print(indA ^ indB) #symmetric difference
