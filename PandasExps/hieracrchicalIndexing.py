import numpy as np
import pandas as pd

print("#A multiply indexed series")
#represent 2 dimensions in a Series

##The bad way
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]

pop = pd.Series(populations, index=index)

print(pop)

print(pop[('California', 2010):('Texas', 2000)])

#trying to select all values in 2010
print(pop[[i for i in pop.index if i[1] == 2010]]) #bad


#Better way, Pandas Multindex
#can create a multindex from tuples

index = pd.MultiIndex.from_tuples(index)
print(index)

pop = pop.reindex(index)
print(pop)

#now you can use pandas slicing
print(pop[:, 2010])

#multi index as an extra dimension
#can unpack multiIndex into a DataFrame
pop_df = pop.unstack()
print(pop_df)

#stack is the reverse
print(pop_df.stack())

pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})

print(pop_df)
f_u18 = pop_df['under18'] / pop_df['total']
print(f_u18.unstack())

print("#Methods of MultiIndex creation")
#can pass a list of multiple indecies to index
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
print(df)
#can build from a dictionary
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
print(pd.Series(data))

#explicit MultiIndex constructors
#from list of lists
print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]]))
#from tuples
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]))
#from cartesian product of indicies
print(pd.MultiIndex.from_product([['a', 'b'], [1, 2]]))

print("##MultiIndex level names")
pop.index.names = ['state', 'year']
print(pop)


print("##MultiIndex for columns")

# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

#mock some data
data = np.round(np.random.randn(4,6), 1)
data[:, ::2] *= 10
data +=37

#create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
#can index from top using name
print(health_data['Guido'])

print("#Indexing and slicing a MultiIndex")
##Multiply indexed series
print(pop)
#can access single elements by indexing with multiple terms
print(pop['California', 2000])
#or partial indexing
print(pop['California'])
#as well as partial slicing if the MultiIndex is sorted
print(pop.loc['California':'New York'])
#empty slice for partial
print(pop[:, 2000])
#boolean masking
print(pop[ pop > 22000000])
#fancy indexing
print(pop[['California', 'Texas']])

print("##Multiply indeced DataFrames")

print(health_data)

print(health_data['Guido', 'HR'])
#iloc and loc also work
print(health_data.iloc[:2, :2])
# or you can use a tuple for multiple indicies
print(health_data.loc[:, ('Bob', 'HR')])
#slicing in tuples will not work
#insteaed use IndexSlice object from Pandas

idx = pd.IndexSlice
print(health_data.loc[idx[:, 1], idx[:, 'HR']])

print("#Rearanging Multi-Indicies")
index = pd.MultiIndex.from_product([['a','c','b'], [1,2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print(data)

try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
#partial slices need the MultiIndex to be sorted
data = data.sort_index()
print(data)
#now it works
print(data['a':'b'])
print()
print("#Stacking and unstackigng indicies")
print(pop.unstack(level=0))
print(pop.unstack(level=1))
print(pop.unstack().stack())

##index setting and resetting
pop_flat = pop.reset_index(name='population')
print(pop_flat)

#you can do the reverse with set_index
print(pop_flat.set_index(['state', 'year']))

print("#Data aggrefations on MultiIndex")

print(health_data)
data_mean = health_data.mean(level='year')
print(data_mean)
#using axis lets you take mean along the column
print(data_mean.mean(axis=1, level='type'))
