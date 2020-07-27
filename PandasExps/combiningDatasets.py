import numpy as np
import pandas as pd

def make_df(cols, ind):
    """Quickly make a data frame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

#example use
print(make_df('ABC', range(3)))

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_()) for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a)) for a in self.args)

    
print("#Recall: Concatenation of Numpy arrays")
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
print(np.concatenate([x,y,z]))

x = [[1, 2],
     [3, 4]]
print(np.concatenate([x, x], axis=1))

#Simple cancatenation with pc.concat()
# can be used for simple concatenations of Series or DataFrames
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])

print(pd.concat([ser1, ser2]))

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([def, def])')


df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis='col')")


##duplicate indicies

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # make duplicate indices!
display('x', 'y', 'pd.concat([x, y])')

print("###Catchign repeates as errors")

try:
    pd.concat([x,y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)

##ignore the index
display('x', 'y', 'pd.concat([x, y], ignore_index=True)')

##Adding MultiIndex keys
display('x', 'y', "pd.concat([x, y], keys=['x', 'y'])")


print("##Concatenation with joins")
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')

#default is to fill missing data with NaN
# you can change this using the inner join for the intersection
display('df5', 'df6', "pd.concat([df5, df6], join='inner')")
#depreciated
display('df5', 'df6', "pd.concat([df5, df6], join_axes=[df5.columns])")

print("##The append method")
print(df1.append(df2))
#this creates a new object instead of modifiying the original
