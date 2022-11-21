import cv2
import os
from glob import glob
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.spatial import distance

def point_dist(p1,p2):
    d = distance.euclidean([p1.x, p1.y], [p2.x, p2.y])
    return d

def calculate_EAR(p1,p2):
	ear = p1/p2
	return ear

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FILES = glob('./yawn/yawn/*.jpg')

#print(IMAGE_FILES)


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
yawn=[]
state=[] # 입 닫았는지 안닫았는지 라벨을 달아줄 것임.

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    p1_h = point_dist(results.multi_face_landmarks[0].landmark[0], results.multi_face_landmarks[0].landmark[17])
    p2_w = point_dist(results.multi_face_landmarks[0].landmark[61], results.multi_face_landmarks[0].landmark[291])
    yawn.append(calculate_EAR(p1_h,p2_w))


#csv 만들기
for i in yawn:
    state.append(1)  #no_yawn 갯수만큼 0 이 생길것.


#리스트를 배열로 만들어주기
#차원을 일단 맞춰주기
yawn = np.array(yawn)
state = np.array(state)
yawn_data=np.column_stack((yawn,state))  #2차원 배열로 만들어줌

yawn_data= pd.DataFrame(data=yawn_data,columns=["rate",'state'])

yawn_data.to_csv('yawn.csv', index=False)

    
