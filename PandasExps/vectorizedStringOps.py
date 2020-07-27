import numpy as np
import pandas as pd
data = ['peter', 'Paul', 'MARY', 'gUIDO']
#pandas has vectorized string ops while numpy does not
names = pd.Series(data)
print(names)

print(names.str.capitalize())

#tables of pandas string methods
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

#Example: Recipe Database
try:
    recipes = pd.read_json('recipeitems-latest.json')
except ValueError as e:
    print("ValueError:", e)
