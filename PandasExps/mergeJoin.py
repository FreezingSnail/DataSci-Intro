import numpy as np
import pandas as pd

print("#Relational Algebra")
#catagories of Joins
print("##One to one joins")
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1,'\n', df2)
df3 = pd.merge(df1, df2)
print(df3)


print("##Many to one joins")
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3, '\n', df4)
print(pd.merge(df3, df4))

print("##Many to many joins")
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1, '\n', df5)
print(pd.merge(df1, df5))


print("#Specifications of the Merge Key")
print(pd.merge(df1, df2, on='employee'))
#both DataFrames need this lable of course

print("##left_on and right_on keywords")
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
#these keywords can be used to join along two missmatched column names

print(pd.merge(df1, df3, left_on="employee", right_on="name"))
#the redudnant column can be dropped with .drop('name', axis=1) 

print("#The left_index and right_index")
#instead of a column you can merge on index
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a, '\n', df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
#DataFrames join() method does the same by defult
print(df1a.join(df2a))
#you can combine these for desired behavior
print(df1, '\n', df3)
print(pd.merge(df1a, df3, left_index=True, right_on='name'))


print("#Specifying set arithmetic for joins")
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])



print(df6, '\n', df7)
print(pd.merge(df6, df7)) #inner join (intersection of name)
print(pd.merge(df6, df7, how='inner')) #the same
#outer join
print(pd.merge(df6,df7, how='outer'))
print(pd.merge(df6,df7, how='left')) #joins over left
#how='right' joins over right


print("#Overlapping Column Names: The keyword")
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})

#conflicting names in second column
print(pd.merge(df8,df9)) # renames the conflicting columns
#suffixes=['_L', '_R'] will changed the suffix for any overlapping column name

print("#Example, US States data")
#read int csv
pop = pd.read_csv('data/state-population.csv')
areas = pd.read_csv('data/state-areas.csv')
abbrevs = pd.read_csv('data/state-abbrevs.csv')

print(pop.head(),'\n', areas.head(),'\n', abbrevs.head())
#many to one join
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region',
                  right_on='abbreviation')
merged = merged.drop('abbreviation', 1) #drop dupicate info
print(merged.head())
print(merged.isnull().any())
#some population data is null
print(merged[merged['population'].isnull()].head())
#some states are also null, -> no coresponding entry in the abbrevs key
#find the regions missing this match
print(merged.loc[merged['state'].isnull(), 'state/region'].unique())
#PR and USA missing from the abbreviation key
#add the missing data
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
print(merged.isnull().any())
#can merge wiht area data now
final = pd.merge(merged, areas, on='state', how='left')
print(final.head())
#check for missmatches
print(final.isnull().any())
#find what regions didnt match with an area
print(final['state'][final['area (sq. mi)'].isnull()].unique())
#united states missing its area, just drop it not relevent
final.dropna(inplace=True)
print(final.head())

#now we can find pop density
#pull out the data for 2010
data2010 = final.query("year == 2010 & ages == 'total'") #requires numexpr package
print(data2010.head())
#now find pop density

data2010.set_index('state', inplace=True) #adding index name 
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
print(density.head())
print(density.tail())
