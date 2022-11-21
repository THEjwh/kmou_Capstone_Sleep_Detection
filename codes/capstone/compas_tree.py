import os
import numpy as np
import pandas as pd
import graphviz

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

compas = pd.read_csv('./compas_dataset/preprocessed.csv')

compas = compas.drop(['id', 'name', 'dob', 'score_text'], axis=1)

encoder = LabelEncoder()
encoder.fit(compas['race'])
labels = encoder.transform(compas['race'])
compas['race'] = labels
encoder.fit(compas['c_charge_desc'])
labels = encoder.transform(compas['c_charge_desc'])
compas['c_charge_desc'] = labels
encoder.fit(compas['sex'])
labels = encoder.transform(compas['sex'])
compas['sex'] = labels

compas_y = compas['decile_score']
compas_x = compas.drop(['decile_score'], axis=1)

scaler = StandardScaler()
scaler.fit(compas_x)
compas_x = scaler.transform(compas_x)

X_train, X_test, y_train, y_test = train_test_split(compas_x, compas_y, test_size=0.2, random_state=52)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

result = model.predict(X_test)
print("acc: {0: .4f}".format(accuracy_score(y_test, result)))