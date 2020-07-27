import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')

print("#Motivating Pivot Tables")
print(titanic.head())

print("#Pivot tabels by hand")
print(titanic.groupby('sex')[['survived']].mean())
print(titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack())

print("#Pivot table syntax")
#this is the equivelent as before
print(titanic.pivot_table('survived', index='sex', columns='class'))
##Mulitlevel pivot tables
age = pd.cut(titanic['age'], [0,18,80])
print(titanic.pivot_table('survived', ['sex', age], 'class'))
fare = pd.qcut(titanic['fare'], 2)
print(titanic.pivot_table('survived', ['sex', age], [fare, 'class']))

##Additional options
# call signature as of Pandas 0.18
#DataFrame.pivot_table(data, values=None, index=None, columns=None,
#                      aggfunc='mean', fill_value=None, margins=False,
#                      dropna=True, margins_name='All')

print(titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'}))
print(titanic.pivot_table('survived', index='sex', columns='class',
                          margins=True))


print("#Example: Birthrate Data")
births = pd.read_csv('data/births.csv')
print(births.head())
births['decade'] = 10*(births['year'] //10)
print(births.pivot_table('births', index='decade', columns='gender',
                         aggfunc='sum'))
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
sns.set() #use seaborn styles
births.pivot_table('births', index='year', columns='gender',
                   aggfunc='sum').plot()
plt.ylabel('total births per year')
plt.show()


##Further data exploration
#cut out outliers
quartiles = np.percentile(births['births'], [25,50,75])
mu = quartiles[1]
sig = 0.74 *(quartiles[2] - quartiles[0])

births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
# set 'day' column to integer; it originally was a string due to nulls
births['day'] = births['day'].astype(int)

#create a datatime index form the year, month, day
births.index = pd.to_datetime(10000 * births.year + 100 * births.month +
                              births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek

births.pivot_table('births', index='dayofweek',
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day');
plt.show()


births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
print(births_by_date.head())
#join with a dummy leap year to plot
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]
print(births_by_date.head())

#plot the results
fig, ax = plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax);
plt.show()
