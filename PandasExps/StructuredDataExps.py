import numpy as np

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

#use compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age','weight'),'formats':('U10','i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

print(data['name']) # all names
print(data[0]) # first row
print(data[-1]['name']) # all names from last row
print(data[data['age'] < 30]['name']) #names less than age 30

print("#More advanced compound types")

tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3,3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])


print("#record arrays")

data_rec = data.view(np.recarray)
print(data_rec.age)
# these are slower than structured arrays
