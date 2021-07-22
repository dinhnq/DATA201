import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

calls = pd.read_excel('../../res/data/ZealandiaKiwiCalls.xls',header=0)
calls["Date"]=pd.to_datetime(calls["Date"])
calls.head()

#task 1:
#- plot the average number of calls in each year for the males and females on one graph. (3 marks)
#- comment on what you notice. (2 marks)

#the magic command that will do nearly all the work for you is a pivot table. look up`pd.pivot_table` and look for the aggregate function (`aggfunc`). the syntax to use is `pd.pivot_table(dataframe, index, values, aggfunc)` where the dataframe is obvious, the index specifies the column(s) that you want to use to group by, values are the columns you want to output, and the aggfunc is how to combine the values.

#after you have made the pivot table you will want to reset the index. this can be done for a dataframe called `pv_year_mean` using:
#`pv_year_mean = pv_year_mean.rename_axis(None, axis=1).reset_index()`

#pl.plot(calls['Total'])

#pv_year_mean = pd.pivot_table(calls, index='Year', aggfunc={'Total':np.mean})
pv_year_female = pd.pivot_table(calls,index='Year',aggfunc={'Female':np.mean})
pv_year_female = pv_year_female.rename_axis(None, axis=1).reset_index()
pl.plot(pv_year_female['Year'], pv_year_female['Female'], 'r', label='Female')

pv_year_male = pd.pivot_table(calls,index='Year',aggfunc={'Male':np.mean})
pv_year_male = pv_year_male.rename_axis(None, axis=1).reset_index()
pl.plot(pv_year_male['Year'], pv_year_male['Male'], 'g', label='Male')

pl.title('Average Nightly Calls - Pylon')
pl.legend()

#pl.plot(pv_year_mean['Year'], pv_year_mean['Total'])
#pv_year_mean.plot(kind='bar')
#pv_year_mean.plot()

pl.show()

print('Comment: {}'.format('The average number of calls by male increases overtime while that of call by female fluctuates and can not reach over 2.5'))

#Task 2.
#- Plot the number of calls detected on each night against the year for the males and females on separate axes. (2 marks)
#- Then use the `polyfit` command that we saw in the lectures to fit a straight line to the plots.
# There are three lines to using polyfit, the next cell should give you the idea. (3 marks)
#- Comment on whether the values of the fits verify you conclusions above. (1 mark)

#You will need to use the `drop_na()` command.

#pv_year = pd.pivot_table(calls, index='Year', aggfunc={'Total':np.sum})
#pv_year = pv_year.rename_axis(None, axis=1).reset_index()
#pl.plot(pv_year['Year'], pv_year['Total'])

#a = np.random.randn(20,2)
#pf = np.polyfit(a[:,0],a[:,1],1)
#f = np.poly1d(pf)
#pl.plot(a[:,0],a[:,1],'.')
#pl.plot(a[:,0],f(a[:,0]))

#pf = np.polyfit(pv_year['Year'], pv_year['Total'], 1)
#f = np.poly1d(pf)
#pl.plot(pv_year['Year'], pv_year['Total'], '.')
#pl.plot(pv_year['Year'], f(pv_year['Year']))

#pl.show()

#print(np.polyfit(pv_year['Year'], pv_year['Total'], 1))

#print(calls.describe())

new_data = calls[['Year', 'Male', 'Female']].dropna(axis=0, how='any')
#print(new_data.describe())

#pv_year_male = pd.pivot_table(new_data, index='Year', aggfunc={'Male':np.sum})
#pv_year_male = pv_year_male.rename_axis(None, axis=1).reset_index()
#pl.plot(pv_year_male['Year'], pv_year_male['Male'], '.')

year = new_data['Year']
male = new_data['Male']
female = new_data['Female']

#slope_m, intercept_m = np.polyfit(year, male, deg=1)
#print('Males:\n{:.4f} x {}'.format(slope_m, round(intercept_m)))

#slope_f, intercept_f = np.polyfit(year, female, deg=1)
#print('Female:\n{:.4f} x {}'.format(slope_f, round(intercept_f)))

#pl.plot(year, male, '.g')
#pl.plot(year, slope_m*year + intercept_m)
#pl.show()

#pl.plot(year, female, '.r')
#pl.plot(year, slope_f*year + intercept_f)
#pl.show()

pf = np.polyfit(year, male, deg=1)
f = np.poly1d(pf)
print('Male:{}'.format(f))
pl.plot(year, male, '.g')
pl.plot(year, f(year))
pl.show()

pf = np.polyfit(year, female, deg=1)
f = np.poly1d(pf)
print('Female:{}'.format(f))
pl.plot(year, female, '.r')
pl.plot(year, f(year))
pl.show()

#print('Comment: {}'.format('the values of the fits verify my conclusions as they show an increase number of calls detected on each night against the year for the males and females over time'))

rain = pd.read_csv('../../res/data/rain.csv',header=2)
rain["Date"]=pd.to_datetime(rain["Date"],format="%Y%m%d")
rain.head()
rain.tail()

#Task 3.
#Merge the two datasets by adding this data to the calls dataframe for the nights when a kiwi call count was carried out (check out `pd.merge`). (2 marks)


new_dataframe = pd.merge(calls, rain, how='inner')
#new_dataframe = pd.merge(calls, rain, how='left', on='Date')
print(new_dataframe.describe())
print(new_dataframe.head())

import datetime, math

def phase(date):
    n = np.floor(12.37 * (date.dt.year - 1900 + ((date.dt.month - 0.5)/12.0)))
    t = n / 1236.85
    az = 359.2242 + 29.105356 * n
    am = 306.0253 + 385.816918 * n + 0.010730 * t * t
    extra = 0.75933 + 1.53058868 * n + ((1.178e-4) - (1.55e-7) * t) * t + (0.1734 - 3.93e-4 * t) * np.sin(np.radians(az)) - 0.4068 * np.sin(np.radians(am))
    extra.where(extra<0,np.floor(extra),inplace=True)
    extra.where(extra>0,np.ceil(extra-1),inplace=True)
    julian = np.where((date.dt.year > 1801) & (date.dt.year < 2099),67 * date.dt.year - ((7 * (date.dt.year + ((date.dt.month + 9) / 12.0).astype(int))) / 4.0).astype(int) + ((275 * date.dt.month) / 9.0).astype(int) + date.dt.day + 1721013.5 + (date.dt.hour + date.dt.minute / 60.0 + date.dt.second / 3600) / 24.0 - 0.5 * np.copysign(1, 100 * date.dt.year + date.dt.month - 190002.5) + 0.5,0)
    jd = (2415020 + 28 * n) + extra
    return (julian-jd + 30)%30

#Task 4.
#- Run this function for each date in the calls dataset and add a new column called "Moon" or similar. (2 marks)
#- Add another column called light, which has 3 categories: 0 - dark, 1 - medium, 2 - bright according to the phase of the moon. (2 marks)

new_dataframe['Moon'] = phase(new_dataframe['Date'])
print(new_dataframe.head())

#new_dataframe.Moon.astype(float)
#df = df.assign(Product=lambda x: (x['Field_1'] * x['Field_2'] * x['Field_3']))
#new_dataframe = new_dataframe.assign(Light=lambda x: 2 if x['Moon'] > 12 else 0)

#new_dataframe['Light'] = new_dataframe['Moon'].apply(lambda x: 2.0 if (x >= 12 and x < 17) else (1. if (x >=9 and x <12) is True else 0.))
#new_dataframe['Light'] = new_dataframe['Moon'].apply(lambda x: 2.0 if (x >= 12.5 and x <= 18.5) is True else (1. if ((x >4.5 and x <12.5) or (x >18.5 and x <25.5)) is True else 0.))
#new_dataframe['Light'] = new_dataframe['Moon'].apply(lambda x: 2.0 if (x >= 12.5 and x < 18.5) is True else (0. if (x <=4.5 or x >25.5) is True else 1.))
new_dataframe['Light']= new_dataframe['Moon'].apply(lambda x: 2.0 if (x>=10 and x<=20) else 1.0 if ((x>=5 and x<10) or (x>20 and x <=25)) else 0.0)

#new_dataframe.loc(new_dataframe['Moon'] < 6, 'Light' = 0)

#moon = new_dataframe['Moon']
#new_dataframe.loc[moon < 6, 'Light'] = 0
#new_dataframe.loc[moon in [0,6], 'Light'] = 1
print(new_dataframe.head())
print(new_dataframe.describe())

#Task 5. Test the correlation of the amount of light and the number of calls and the amount of moisture and the number of calls.
# Comment on your findings. (3 marks)
corr_dataframe = new_dataframe[['Male', 'Female', 'Moon', 'Light', 'Amount (mm)']]
corr_matrix = corr_dataframe.corr()

print(corr_matrix)

print('Comment: ')
print('The amount of light has a negative relationship with the moon phase, the number of female calls, and the amount of moisture.')
print('The amount of light has a positive relationship with the number of male calls.')
print('A change of the amount of light has not been statistically significant affect to the other factors as an increase of one unit in light would not create not larger than 5% change on the number of calls, the amount of moisture, and the number of calls.')
