import csv as csv
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_colwidth', 50)

train = pd.read_csv('C:/Bob/Data/titanic/train.csv')
test = pd.read_csv('C:/Bob/Data/titanic/test.csv')

print train.values
print train.columns.values

train.head()
train.info()
train.describe()

C:\Bob\Data\Titanic

#c1 = pd.ExcelFile("C:\\Bob\\Data\\Titanic\\train.csv")

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_colwidth', 50)


f1 = pd.ExcelFile("C:\\Bob\\Data\\titanic3.xls")
df_train = pd.read_csv('C:\\Bob\\Data\\titanic\\train.csv')
print(len(df_train.columns))
print (df_train.head())

df = pd.read_excel(open('C:\\Bob\\Data\\titanic3.xls', 'rb'), sheetname = 'titanic3')

print (df.head())

df_train.info()

#import sys
#print(sys.executable) #C:\Users\Bob\Anaconda2\envs\Scikitlearn01\pythonw.exe

import sys
print(sys.executable) #C:\Users\Bob\Anaconda2\envs\Scikitlearn01\pythonw.exe

import csv as csv
import numpy as np
import pandas as pd

pd.read('test', )
print ("Hello, World!")

df1 = pd.read_excel("C:\\Bob\\Data\\titanic3.xls", 'titanic3')

print (df1.head())

print ("survived mean = %f" % df1["survived"].mean())

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