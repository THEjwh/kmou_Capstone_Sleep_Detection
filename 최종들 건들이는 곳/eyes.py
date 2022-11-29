# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t_OS-fiGYOgLQSYv7SCtOzD9TxrDEvOe
"""

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

datas = pd.read_csv('eye_Ear_Dataset2 copy.csv')
data = shuffle(datas, random_state=32)

target_y = data['is_closed']
target_x = data.drop(['is_closed'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(target_x, target_y, test_size=0.3, random_state=52)

"""그래디언트는 89%인식"""

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=1000, random_state=32, max_depth=10).fit(X_train, y_train)
print(clf.score(X_test,y_test))
prediction=clf.predict(X_test)
print(accuracy_score(y_test,prediction))

"""#랜덤 포레스트는 91프로 정확도를 보임."""

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# rdf = RandomForestClassifier(criterion= 'entropy',oob_score= True)
# params = {
#         "min_samples_split" : np.arange(2,20,1),
#         "min_samples_leaf" : np.arange(1,20,1)
# }
# grid = GridSearchCV(rdf,params)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# rdf_g=grid.best_estimator_
# print(rdf_g.score(X_test,y_test))
# prediction= rdf_g.predict(X_test)
# print(accuracy_score(y_test,prediction))

"""엑스트라 트리는 89% 인식"""

# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.model_selection import cross_validate
# et = ExtraTreesClassifier(random_state=26)
# score = cross_validate(et,X_train,y_train,return_train_score= True)
# print(np.mean(score['train_score']),np.mean(score['test_score']))
# # print(et.score(X_train,y_train))
# # print(et.score(X_test,y_test))

"""일반 결정트리가 91%"""

# from sklearn.tree import DecisionTreeClassifier
# dt =DecisionTreeClassifier(random_state= 26)
# params={
#         "min_samples_split" : np.arange(2,20,1),
#         "min_samples_leaf" : np.arange(1,20,1)
# }
# grid = GridSearchCV(dt,params)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# dt_g=grid.best_estimator_
# print(dt_g.score(X_test,y_test))
# prediction= dt_g.predict(X_test)
# print(accuracy_score(y_test,prediction))

# from sklearn.linear_model import SGDClassifier
# sgd=SGDClassifier(random_state=26)
# sgd.fit(X_train,y_train)
# print(sgd.score(X_test,y_test))
# prediction=sgd.predict(X_test)
# print(accuracy_score(y_test,prediction))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
hgb=HistGradientBoostingClassifier(loss='auto',random_state=26)
params={
        "tol":np.arange(1e-7,1e-6,1e-7),
        "max_depth":np.arange(3,20,1),
        "min_samples_leaf": np.arange(1,20,1)
}
grid = GridSearchCV(hgb,params)
grid.fit(X_train,y_train)
print(grid.best_params_)
hgb_g=grid.best_estimator_
print(hgb_g.score(X_test,y_test))
prediction= hgb_g.predict(X_test)
print(accuracy_score(y_test,prediction))