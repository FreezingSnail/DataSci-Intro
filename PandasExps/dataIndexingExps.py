import pandas as pd
import numpy as np

print("#Data selection in series")
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])
print( 'a' in data )
print(data.keys())
print(list(data.items()))
data['e'] = 1.25
print(data['e'])

print("#Series as 1d array")
#slicing by explicit index
print(data['a':'c'])
#slicing by implicit integer index
print(data[0:2])
#masking
print(data[(data > 0.3) & (data < 0.8)])
#fancy indexing
print(data[['a','e']])

print("#indexers: loi, iloc, ix")
data = pd.Series(['a','b','c'], index=[1,3,5])
print(data)
#explicit index
print(data[1])
#implicit indexing
print(data[1:3])

#.loc allows explicit indexing wghen slicing
print(data.loc[1])
print(data.loc[1:3])

#.iloc allows indexing using python implicit style
print(data.iloc[1])
print(data.iloc[1:3])

print("#Data selection in DataFrame")
##DataFrame as a dictionary

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data)

print(data.area) # same as data['area']
data['density'] = data['pop'] / data['area']
print(data)

#can view this as an enhanced 2d array
print(data.values)

print(data.T) #transpose

print(data.values[0]) #access a row
print(data['area']) #access a collumn

print(data.iloc[:3, :2]) # array style indexing implicit
print(data.loc['Illinois', 'pop']) # explicit index using names
#print(data.ix[:3, :'pop']) depreciated

print(data.loc[data.density > 100, ['pop', 'density']]) # fancy indexing
#can also modify values
data.iloc[0,2] = 90
print(data)

print("##Additional indexing conventions")
print(data['Florida':'Illinois'])
print(data[1:3])

print(data[data.density > 100])
