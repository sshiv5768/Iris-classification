# Importing useful modules
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
import pickle

warnings.filterwarnings("ignore")

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y)
lin_reg = LinearRegression()
svc_model = SVC()

lin_reg = lin_reg.fit(x_train, y_train)
svc_model = svc_model.fit(x_train, y_train)

pickle.dump(lin_reg, open('lin_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))
