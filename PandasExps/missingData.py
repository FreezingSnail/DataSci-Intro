import numpy as np
import pandas as pd

print("#None: Pythonic missing data")
#this can only be used in arrays with data type 'object'

valsl = np.array([1,None, 3, 4])
print(valsl)
print(valsl.dtype)
#operations on this will be done on the python level with all the included overhead and slowdown
#aggregations like sum() or min() will throw a TypeError

print("##NaN: Missing numerical data")

vals2 = np.array([1,np.nan, 3, 4])
print(vals2)
print(vals2.dtype)

#NaN will overtake data mixed with it
print(1 + np.nan) # NaN

#aggregates are well defined but not necissarily useable
print(vals2.sum(), vals2.min())

#you can use special case aggregations instead
print(np.nanmin(vals2), np.nansum(vals2))

print("#NaN and None in Pandas")
#pandas can convert between where appropraite

print(pd.Series([1, np.nan, 2, None]))
x = pd.Series(range(2), dtype=int)
print(x)
x[0] = None
print(x) # upcasts to float as int64 has no NaN value

print("#Operating on Null Values")
#detecting null

data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())
print(data[data.notnull()])

##dropping null values

print(data.dropna())
#more options
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
print(df)
#you cant drop a single value, only full rows or columns
#dropna() defaults to removing all rows with any null value
print(df.dropna())
print(df.dropna(axis='columns'))# drop column instead

df[3] = np.nan
print(df)

print(df.dropna(axis='columns', how='all'))
#finer grain using thresh give a min num of null values
print(df.dropna(axis='rows', thresh=3))


print("##Filling Null values")
data = pd.Series([1,np.nan, 2, None, 3], index=list('abcde'))
print(data)

print(data.fillna(0))

#forward fill
print(data.fillna(method='ffill'))
#backfill
print(data.fillna(method='bfill'))
#simmilar for fd
print(df.fillna(method='ffill', axis=1))
