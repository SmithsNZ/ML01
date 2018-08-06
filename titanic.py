#ctl alt e to run selection in wing

# import csv as csv
import numpy as np
import pandas as pd
# import xlrd
# import pylab as P

from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 100) 

1/0
# Excel Syntax
#f1 = pd.ExcelFile("C:\\Bob\\Data\\titanic\\\titanic3.xls")
#df = pd.read_excel(open('C:\\Bob\\Data\\titanic\\titanic3.xls', 'rb'), sheetname = 'titanic3')
#print (df.head())

train = pd.read_csv('C:/Bob/Data/titanic/train.csv')
test = pd.read_csv('C:/Bob/Data/titanic/test.csv')

df_train = pd.read_csv('C:\\Bob\\Data\\titanic\\train.csv')
type(train) # pandas.core.frame.DataFrame

len(train.columns) # col count
train.head(10) # sample data
train.tail(10)
train.info() # file structure
train.describe() # content info
#          Survived  PassengerId      Pclass         Age       SibSp       Parch        Fare
# count  891.000000   891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean     0.383838   446.000000    2.308642   29.699118    0.523008    0.381594   32.204208
# std      0.486592   257.353842    0.836071   14.526497    1.102743    0.806057   49.693429
# min      0.000000     1.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%      0.000000   223.500000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%      0.000000   446.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%      1.000000   668.500000    3.000000   38.000000    1.000000    0.000000   31.000000
# max      1.000000   891.000000    3.000000   80.000000    8.000000    6.000000  512.329200

#std = 68% of values are +- value. avg Age = 29, 68% are between 15 (29-14) and 43 (29+14)

# http://seaborn.pydata.org/api.html

sns.factorplot('Sex', data=train, kind='count')
plt.show()
print train.Age.hist()
print df_train.Age.dropna().hist(bins=20)

df_train.Age[0:10] #select top 10 Age
type(df_train.Age) # pandas.core.series.Series
df_train.mean() # mean of each column
df_train.Age.mean() # mean age 29.69911764705882
df_train.Age.median() # median age 28.0

df_train[df_train.Age > 60] # filter = select * where Age > 60

# select Sex, Pclass, Age, Surviced where Age > 60
df_train[df_train.Age > 60][['Sex', 'Pclass', 'Age', 'Survived']]

# where Age is null
df_train[df_train.Age.isnull()][['Sex', 'Pclass', 'Age', 'Survived']]

# for each pclass select count(*) where Sex = 'male and Pclass == n
for pclass in range(1,4):
    print pclass, len(df_train[ (df_train.Sex == 'male') & (df_train.Pclass == pclass)  ])

# hist is mapped to matplotlib/pylab (and already seems to be imported)
df_train.Age.hist()
df_train.Pclass.hist()
df_train[df_train.Fare <= 100].Fare.hist()
df_train.Age.dropna().hist(bins=10, range=(0,80), alpha = 0.5)
# P.show()

# http://www.python-course.eu/list_comprehension.php
# create/update column with constant value
df_train['Gender'] = 4 
# map (afunction, a sequence) - apply passed fn() to each item in seq and return list results
# update col with function result of uppercase first letter of Sex column
df_train['Gender'] = df_train['Sex'].map( lambda x: x[0].upper() )
df_train['Gender'] = [ Sex[0].upper() for Sex in df_train['Sex'] ] # as list comprehension
# note make function lambda : x**2 == def f (x): return x**2
# filter() return only True, map() return new list, reduce() recursive call r1=(p1,p2), r2=(r1, p3), r3=(r2, p4) etc
# no fn() supplied returns all true entries (identity function)
df_train['Gender'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_train['Gender'] = [ 0 if Sex == 'female' else 1 for Sex in df_train['Sex'] ]
 
# rename column
df_train.rename(columns = {"Gender": "GenderCode"}, inplace=True)
df_train.head(10)

df_train['Embarked'].value_counts() # show col values (count(*) group by)

# convert column, changing non numerics to NaN 
# pd.to_numeric(df_train['EmbarkedCode'], errors = 'coerce')

EmbarkedCodeMap = {"C": 1, "S": 2, "Q":3}
# EmbarkedCodeMap = {"S": 2, "Q":3}
df_train['EmbarkedCode'] = np.NaN # not really needed as apply does this anyway
# replace works, but copies missing lookup values from source
# df_train['EmbarkedCode'] = df_train['Embarked'].replace(EmbarkedCodeMap)
# apply works and replaces missing lookup values with NaN, but converts to float as ints can't hold NaNs
df_train['EmbarkedCode'] = df_train.Embarked.apply(EmbarkedCodeMap.get)
# count isnulls / notnulls
df_train.EmbarkedCode.isnull().sum()
df_train.EmbarkedCode.notnull().sum()
# also total minus non nulls
len(df_train) - df_train.EmbarkedCode.count() # len(df_train)

# show isnulls
df_train[df_train.EmbarkedCode.isnull()]

df_train.head(20)
df_train['EmbarkedCode'].mean()
df_train.describe()

#most ML needs complete set of values
print df_train.Age.mean() # mean age = 29.69911764705882 (avg)
print df_train.Age.median() # median age = 28.0 (middle value)
print df_train.Age.mode() # mode age = 24 (most frequent, can be none if nothing repeated)

print df_train.Fare.mean()  #32 skewed by few v large fares
print df_train.Fare.median() #14 50% value
print df_train.Fare.mode() #8 most frequent

# create set of typical age for sex and pclass
median_ages = np.zeros((2,4))
median_ages

for gendercode in range (0,2):
    for pclass in range (1,4):
        median_ages[gendercode, pclass] = \
            df_train[(df_train.GenderCode == gendercode) &
            (df_train.Pclass == pclass)].Age.median()

# create assumed age column

df_train.head()
df_train['AgeFill'] = np.NaN
df_train[df_train.AgeFill.isnull()][['Sex', 'GenderCode', 'Pclass', 'Age', 'AgeFill']].head()

# update rows, set agefill to actual age or median age for gender and pclass

df_train['AgeFill'] = df_train.Age

for gendercode in range (0,2):
    for pclass in range (1,4):
        df_train.loc[ (df_train.AgeFill.isnull()) & \
                         (df_train.GenderCode == gendercode) & \
                         (df_train.Pclass == pclass), \
                         'AgeFill'] = median_ages[gendercode, pclass]

# set flag to show age was not supplied
df_train['AgeIsNull'] = df_train.Age.isnull().astype(int)

df_train.describe()

# family size
df_train['FamilySize'] = df_train.SibSp + df_train.Parch

# create feature of pclass/age combo - in case useful predictor
df_train['AgeFill*Pclass'] = df_train.AgeFill * df_train.Pclass
# drop column
# df_train.drop(['AgeFill * Pclass'], inplace=True, axis=1) # 0==rows, 1==cols

# select and print cols
col_list = ['Sex', 'GenderCode', 'Pclass', 'AgeFill', 'AgeIsNull', 'FamilySize', 'AgeFill*Pclass']
print df_train[col_list].head()

print df_train[df_train.Age.isnull()][col_list].head()
print df_train[df_train.Age.notnull()][col_list].head()
print df_train[df_train['AgeFill*Pclass'].isnull()][col_list].head()

df_train.FamilySize.hist()
df_train['AgeFill*Pclass'].hist()

# don't want string (object) cols
print df_train.dtypes.head
print df_train.dtypes[df_train.dtypes.map(lambda x: x== 'object')]


df_train.head() # handy for copy col names
df_train.describe() # only shows numercic cols

# create new trial df with selected columns
drop_col_list = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age']
df_trial = df_train.drop(drop_col_list, axis=1)
print df_train[drop_col_list].head()

# or just keep the ones we want!!
trial_col_list = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'GenderCode', \
                  'EmbarkedCode', 'AgeFill',  'AgeIsNull', 'FamilySize',  'AgeFill*Pclass']
df_trial = df_train[trial_col_list]
print df_trial.head()

# check we have no NaN values
print df_trial[pd.isnull(df_trial).any(axis=1)] # see index with boolean series

# if so then set col to its most common values
print df_train.EmbarkedCode.value_counts() 
df_train.loc[df_train.EmbarkedCode.isnull(), 'EmbarkedCode'] = 2

df_trial.describe()

###############################################################################
# scikit-learn models (alomost) all have model.fit(), predict(), score()
# need first col as survived and no passenger id
#
# score 0.79-0.81 = doing well, 0.81-0.82 very good
###############################################################################

train_data = df_trial.values
print train_data
type(train_data) # numpy.ndarray

df_test = pd.read_csv('C:\\Bob\\Data\\titanic\\test.csv')
test_data = df_test.values
type(test_data)

print train_data

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit (train_data[0::, 1::], train_data[0::,0])

output = forest.predict(test_data)

print train_data[0::, 4]

