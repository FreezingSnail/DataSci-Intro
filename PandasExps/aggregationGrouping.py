import numpy as np
import pandas as pd

import seaborn as sns
planets = sns.load_dataset('planets')
print(planets.shape)
print(planets.head)

print("#Aggregation in pandas")
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
print(ser)
print(ser.sum())
print(ser.mean())
df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
print(df)

print(df.mean())
print(df.mean(axis='columns'))

print(planets.dropna().describe()) #computes varios aggregations


print("#GroupBy: Split, Apply, Combine")
#higher lever aggregations

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
print(df)

print(df.groupby('key')) #note this returns a groupby object
print(df.groupby('key').sum())

print("##the groupby object")
###column indexing
print(planets.groupby('method'))
print(planets.groupby('method')['orbital_period'])
print(planets.groupby('method')['orbital_period'].median())
###itteration over groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))

###Dispatch methods
print(planets.groupby('method')['year'].describe().unstack())

print("##Aggregate, filter, transform, apply")


rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
print(df)

###aggregation
print(df.groupby('key').aggregate(['min', np.median, max]))
print(df.groupby('key').aggregate({'data1':'min', 'data2':'max'}))

###Filtering
def filter_func(x):
    return x['data2'].std() > 4
print(df.groupby('key').filter(filter_func))
#group a does not have std > 4 so it is dropped

###Transformation
#can return data the same shape as the df instead of reducing like aggregation
#can center data by subtracting group means
print(df.groupby('key').transform(lambda x: x-x.mean()))

###the apply() method
#apply arbitray function to the group

def norm_by_data2(x):
    """x is a DataFrame of a group values"""
    x['data1'] /= x['data2'].sum()
    return x

print(df.groupby('key').apply(norm_by_data2))

print("##Specifying the split key")
L = [0,1,0,1,2,0]
print(df.groupby(L).sum())
print(df.groupby(df['key']).sum()) #more verbose

###A dictionary or series mapping index  by group

df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2.groupby(mapping).sum())


###any python func
print(df2.groupby(str.lower).mean())

###list of valid keys

print(df2.groupby([str.lower, mapping]).mean())

print("##Grouping Example")

decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))

