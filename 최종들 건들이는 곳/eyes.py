
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