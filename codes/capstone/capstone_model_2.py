import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib


datas = pd.read_csv('./head_Pose_Dataset.csv')
data = shuffle(datas, random_state=32)

target_y = data['is_angled']
target_x = data.drop(['is_angled'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(target_x, target_y, test_size=0.2, random_state=52)

sgd_clf = SGDClassifier(max_iter=100, tol=1e-3, random_state=42)
sgd_clf.fit(X_train,y_train)
print(sgd_clf.score(X_test,y_test))
test = [[0.7964, 0.5960, 0.3041, 0.6390,  0.3988, 0.6480, 0.5291]]

print(sgd_clf.predict(test))
print(y_test.iloc[0])
joblib.dump(sgd_clf, 'angle_model.pkl')