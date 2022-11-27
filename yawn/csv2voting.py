#만들어진 csv를 섞어서 하나의 csv만들고 하는 방법
#우선 데이터가 no_yawn과 yawn이 있다고 하자

import pandas as pd
import numpy as np

# #pandas로 읽기
# no_yawn = pd.read_csv("no_yawn.csv")
# yawn = pd.read_csv("yawn_data.csv")

# # numpy배열로 만들기
# no_yawn = no_yawn.to_numpy()
# yawn = yawn.to_numpy()

# mouth_data= pd.concat([no_yawn,yawn],axis=0)

mouth_data = pd.read_csv("mouth_data.csv")

from sklearn.utils import shuffle
mouth_data = shuffle(mouth_data,random_state=26)

train=mouth_data['rate'].to_numpy()
test=mouth_data['state'].to_numpy()


#train, test 만들기
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = train_test_split(train,test,test_size=0.2)


train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

#데이터 모습 보기
import matplotlib.pyplot as plt
plt.scatter(train_input,train_target)
plt.xlabel('rate')
plt.ylabel('state')
plt.show()



#정규화 단계
from sklearn.model_selection import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_input = ss.transform(train_input)
train_target = ss.transform(train_input)


#회귀트리 코드 & 그리드 서치
#from scipy.stats import uniform,randint
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

ex=ExtraTreesRegressor()
params={'n_estimators' :np.arange(500), 
        'learning_rate': np.arange(0.1,1,0.1),
        'max_depth':np.arange(10) 
        }
grid = GridSearchCV(ex,params,n_jobs =-1)
grid.fit(train_input,train_target)
print(grid.best_params_)  #최상 조합 출력
print(np.max(grid.cv_results_['mean_test_score']))

#최적의 모델은 best_estimator_ 속에 저장되어 있음
dt=grid.best_estimator_
print(dt.score(test_input,test_target))

#선형회귀
from sklearn.linear_model import LinearRegression
lr_r=LinearRegression()
lr_r.fit(train_input,train_target)


#lr_r.coef_ : 기울기 / lr_r.intercept_ : y절편
# 선 긋는 거 plt.plot() 





#로지트틱 회귀트리
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(train_input,train_target)
decision = lr.predict_proba(train_input)

from scipy.special import expit  #시모이드 함수
print(expit(decision))




# #보팅
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import accuracy_score
# voting_result = VotingClassifier(estimators=[("추정기 이름",추정기),(),()],weights=[비율,,],voting='soft'.fit(train_input,target_input))
# soft_voting_predicted = voting_result.predict(test_input)
# #accuracy_score(test_target, soft_voting_predicted) 정확도 보는 것

