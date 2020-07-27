import numpy as np
import pandas as pd

print("#Ufuncs: index preservation")

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0,10,4))
print(ser)

df = pd.DataFrame(rng.randint(0,10,(3,4)), columns=['A','B','C','D'])
print(df)

print(np.exp(ser))
print(np.sin(df * np.pi / 4))

print("#Ufuncs: index alignment")
## in series

area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

print(population / area)
print(area.index | population.index)

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A+B)

#can use explicit call to fill missing data
print(A.add(B, fill_value=0))

print("#indexing alignment in dataframe")
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
print(A)
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
print(B)
print(A+B)
#can also fill with a value
fill = A.stack().mean()
print(A.add(B, fill_value=fill))


print("#Ufuncs: operations between DataFrams and Series")
A = rng.randint(10, size=(3, 4))
print(A)
print(A-A[0])

df = pd.DataFrame(A, columns=list('QRST'))
print(df-df.iloc[0])
print(df.subtract(df['R'], axis=0)) #column vs row

halfrow = df.iloc[0, ::2]
print(halfrow)
print(df-halfrow)
