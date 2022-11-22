import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

datas = pd.read_csv('./eye_Ear_Dataset2.csv')
data = shuffle(datas, random_state=32)

target_y = data['is_closed']
target_x = data.drop(['is_closed'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(target_x, target_y, test_size=0.3, random_state=52)

clf = GradientBoostingClassifier(n_estimators=1000, random_state=32, max_depth=10).fit(X_train, y_train)

#sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=32, penalty='l2')
#sgd_clf.fit(X_train,y_train)
for i, col in enumerate(target_x.columns):
    print(f'{col} importance : {clf.feature_importances_[i]}')
print(clf.get_params())
print(clf.score(X_test,y_test))

#print(clf.predict([[0.0836, 0.3623, 0.4607, 0.4258, 0.4569, 0.3824, 0.4585, 0.3849, 0.4632, 0.402, 0.4559, 0.4044, 0.4606]]))
#print(clf.predict([[0.15,0.3275,0.5324,0.4364,0.5217,0.3623,0.5195,0.3643,0.5358,0.3947,0.5163,0.3983,0.5323]]))
joblib.dump(clf, 'ear_model2.pkl')