

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

#데이터 읽기
datas = pd.read_csv('./head_Pose_Dataset.csv')

#데이터 섞기
data = shuffle(datas, random_state=32)


target_y = data['is_angled']
target_x = data.drop(['is_angled'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(target_x, target_y, test_size=0.2, random_state=52)


#특성공학
from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(include_bias=False)
poly.fit(X_train)
X_train=poly.transform(X_train)
X_test = poly.transform(X_test)


#정규화 단계
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)




from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
hgb=HistGradientBoostingClassifier(loss='auto',random_state=26)
params={
    "tol":np.arange(1e-7,1e-6,1e-7),
    "max_depth":np.arange(3,20,1),
    "min_samples_leaf":np.arange(1,20,1)
}
grid=GridSearchCV(hgb,params)
grid.fit(X_train,y_train)
print(grid.best_params_)
hgb_g=grid.best_estimator_
print("train: ",hgb_g.score(X_train,y_train))
print("test :",hgb_g.score(X_test,y_test))
predict = hgb_g.predict(X_test)
print("정확도 :" ,accuracy_score(y_test,predict))