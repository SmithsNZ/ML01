import csv as csv
import numpy as np
import pandas as pd

from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_colwidth', 50)

print "here"

# df = pd.read_excel(open('C:\\Bob\\Data\\titanic3.xls', 'rb'), sheetname = 'titanic3')
train = pd.read_csv('C:/Bob/Data/titanic/train.csv')
test = pd.read_csv('C:/Bob/Data/titanic/test.csv')

print train.values
print len(train.columns)
print train.columns.values

train.head()
train.info()
train.describe()

# https://mashimo.wordpress.com/2013/07/21/visualize-quartiles-and-summary-statistics-in-python/

# http://seaborn.pydata.org/api.html
# https://elitedatascience.com/python-seaborn-tutorial

p = sns.factorplot('Sex', data=train, kind='count')
plt.show(p)

p = sns.lmplot(data=train, x='Fare', y='Pclass') # regression plot, remove line for scatter
plt.show(p)

p = sns.lmplot(data=train, x='Fare', y='Pclass') # regression plot, remove line for scatter
plt.ylim(0, None) # use plt.ylim(0, None), plt.xlim(0, None) to control axis limits
plt.show(p)

p = sns.lmplot(data=train, x='Fare', y='Pclass', fit_reg=False, hue='Pclass') # scatter plot
plt.show(p)

p = sns.boxplot(data=train) # defaults to any plottable columns
plt.show(p)

# prepare better pandas data frame to reflect required data
train_box01 = train.drop(['PassengerId'], axis=1) # drop cols
p = sns.boxplot(data=train_box01)
plt.show(p)

train_box02 = train.ix[:, ['Fare', 'Age', 'Pclass', 'Survived']] # better to select cols
p = sns.boxplot(data=train_box02)
plt.show(p)

# group by colur for extra dim
p = sns.boxplot(data=train_box02, x='Pclass', y='Age', hue='Survived')
plt.show(p)

p = sns.boxplot(data=train_box02, x='Age', y='Pclass', hue='Survived', orient='h')
plt.show(p)

p = sns.violinplot(data=train, x='Sex', y='Age') # um, grow up
plt.show(p)

p = sns.swarmplot(data=train, x='Sex', y='Age') # dots better
plt.show(p)

# add dots to boxplot
p = sns.boxplot(data=train_box02, x='Pclass', y='Age')
p = sns.swarmplot(data=train_box02, x='Pclass', y='Age', color='0.25')
plt.show()

# use catplot for facets (better than facetgrid)
p = sns.factorplot(data=train, x='Pclass', y='Age', hue='Survived', col='Sex', kind='box')
plt.show(p)
sns.cat
# pandas: melt cols to rows to show as same value with type legend
# help takes object (use intell prompt for syntax)
# help(pd.DataFrame.corr) help(train.corr()) help(pd.lib) help(sns.heatmap)

corr = train.corr(method='pearson') # makes matric of cross relations between numerics (look at grid!)
p = sns.heatmap(corr)
plt.show(p)

# Nice - shows relationship (+ve or -ve change of values) between numerics, whitespace == no relationship
#        eg relationship between pclass going down and fare, survived going up and fare
corr = train.corr(method='pearson')
p = sns.heatmap(data=corr, center=0, vmin=-1, vmax=1, cmap=sns.color_palette("BrBG", 7))
plt.show(p)

# rgb options http://seaborn.pydata.org/tutorial/color_palettes.html?highlight=cmap
# p = sns.heatmap(corr, center=0, vmin=-1, vmax=1, cmap=sns.diverging_palette(10, 220, sep=80, n=7))
# plt.show(p)
# p = sns.heatmap(corr, center=0, vmin=-1, vmax=1, cmap=sns.color_palette("coolwarm", 7))
# plt.show(p)

# histogram == distplot https://elitedatascience.com/python-seaborn-tutorial
p = sns.distplot(train.ix[:, 'Age'].dropna())
plt.show(p)


print(df1.groupby('pclass').mean())

df_class_sex = df1.groupby(['pclass', 'sex']).mean()
df_class_sex

df_age = pd.cut(df1['age'], np.arange(0, 90, 10))
age_group = df1.groupby(df_age).mean()

tips_url = 'https://raw.github.com/pydata/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(tips_url)
tips.head()
# select cols
tips[['total_bill', 'tip', 'smoker', 'time']].head(5)
# where
tips[tips['time'] == 'Dinner'].head(5)
# counts
is_dinner = tips['time'] == 'Dinner'  # createa series object
is_dinner.value_counts()
tips[is_dinner].head()
tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5)]

nullset = pd.DataFrame({'ColA': ['A','B',np.NaN,'C','D'],
                        'ColB': ['F', np.NaN, 'G', 'H', 'I']})
nullset
nullset[nullset['ColA'].isnull()]

#group by == split into groups and apply function
#select sex, count(*) from tips group by sex
tips.groupby('sex').size() # counts the items

tips.groupby('sex').count()

tips.groupby('sex')['total_bill `'].count()

d1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': np.random.randn(4)})