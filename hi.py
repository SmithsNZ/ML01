pwd # C:\Bob\Source\SkL01
#import sys
#print(sys.executable) #C:\Users\Bob\Anaconda2\envs\Scikitlearn01\pythonw.exe

#type(iris_ds)
#type(iris_ds) is pandas.core.frame.DataFrame
#help(list) # takes object
#help('def') # lookup string
#help(dir)
#x=5
#dir(x)

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def vers():
    # Check the versions of libraries
    print(sys.executable)
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))

def hello(name):
    """ Hi Bob help"""
    print ("Hello {}".format(name))

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_ds = pandas.read_csv(url, names=names)

type(iris_ds)
type(iris_ds) is pandas.core.frame.DataFrame
help(list) # takes object
help('def') # lookup string
help(dir)
x=5
dir(x)

help() # start interactive
iris_ds.shape


# only start program if run from cmd line (not imported) 
#if __name__  == "__main__":
#    vers()



