
import pandas as pd
import numpy as np

mouth_data = pd.read_csv("mouth_data.csv")


#데이터 섞기
from sklearn.utils import shuffle
mouth_data = shuffle(mouth_data,random_state=26)

#데이터을 배열로 만들기
train=mouth_data['rate'].to_numpy()
test=mouth_data['state'].to_numpy()

#train, test 만들기
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = train_test_split(train,test,test_size=0.2)

# #데이터 모습 보기
# import matplotlib.pyplot as plt
# plt.scatter(train_input,train_target)
# plt.xlabel('rate')
# plt.ylabel('state')
# plt.show()
# #그림이 매우 극단적임. 게다가 겹치는 곳도 너무 많고. 0.5~1.0 겹치다니... 이런 경우는 선형회귀가 될까...

#차원조정하기
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

#데이터 정규화 단계  -- 릿지랑 라쏘 하려면 꼭 데이터를 정규화 해줘야함
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_input = ss.transform(train_input)
test_input = ss.transform(test_input)



from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import accuracy_score
hgb=HistGradientBoostingClassifier(loss='auto',random_state=26)
params={
        "tol":np.arange(1e-7,1e-6,1e-7),
        "max_depth":np.arange(3,20,1),
        "min_samples_leaf": np.arange(1,20,1)
}
grid = GridSearchCV(hgb,params)
grid.fit(train_input,train_target)
print(grid.best_params_)
hgb_g=grid.best_estimator_
print(hgb_g.score(test_input,test_target))
prediction= hgb_g.predict(test_input)
print(accuracy_score(test_target,prediction))