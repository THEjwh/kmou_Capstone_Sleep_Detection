import cv2
import time
import pandas as pd
import numpy as np
import mediapipe as mp
import sklearn
import joblib
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

def point_dist(p1, p2):
    d = distance.euclidean([p1.x, p1.y], [p2.x, p2.y])
    return d

def calculate_EAR(p1, p2, p3):
    ear = (p1 + p2) / (2.0 * p3)
    return ear

ear_model = joblib.load('./ear_model2.pkl')
angle_model = joblib.load('./angle_model.pkl')
mouth_model = joblib.load('./mouth_model.pkl')
mp_face_mesh = mp.solutions.face_mesh

video = cv2.VideoCapture(0)
prev_time = 0
FPS = 10

temp = 0
real = 0

hdp = 200
nosetp = 6
nosebp = 94 

with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while True:
        ret, image = video.read()

        current_time = time.time() - prev_time

        if (ret is True) and (current_time > 1./FPS):
            prev_time = time.time()
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                rt = results.multi_face_landmarks[0].landmark[158]
                rb = results.multi_face_landmarks[0].landmark[153]
                lt = results.multi_face_landmarks[0].landmark[160]
                lb = results.multi_face_landmarks[0].landmark[144]
                hl = results.multi_face_landmarks[0].landmark[130]
                hr = results.multi_face_landmarks[0].landmark[243]
                
                nt = results.multi_face_landmarks[0].landmark[nosetp]
                nb = results.multi_face_landmarks[0].landmark[nosebp]
                fb = results.multi_face_landmarks[0].landmark[hdp]
                
                p1_h = point_dist(results.multi_face_landmarks[0].landmark[0], results.multi_face_landmarks[0].landmark[17])
                p2_w = point_dist(results.multi_face_landmarks[0].landmark[61], results.multi_face_landmarks[0].landmark[291])
                
                mouth_ratio = p1_h / p2_w
                
                p1 = point_dist(rt, rb)
                p2 = point_dist(lt, lb)
                p3 = point_dist(hr, hl)

                p4 = point_dist(nt, nb)
                p5 = point_dist(nb, fb)

                ratio = p4 / p5

                EAR_data = round(calculate_EAR(p1,p2,p3), 4)
                tdata = [EAR_data]
                angledata = [ratio]
                angledata.append(round(nt.x,4))
                angledata.append(round(nt.y,4))
                angledata.append(round(nb.x,4))
                angledata.append(round(nb.y,4))
                angledata.append(round(fb.x,4))
                angledata.append(round(fb.y,4))
                #df = pd.DataFrame(EAR_data, columns=['EAR'])

                # tdata.append(round(hl.x,4))
                # tdata.append(round(hl.y,4))
                # tdata.append(round(hr.x,4))
                # tdata.append(round(hr.y,4))
                # tdata.append(round(lt.x, 4))
                # tdata.append(round(lt.y, 4))
                # tdata.append(round(lb.x, 4))
                # tdata.append(round(lb.y, 4))
                # tdata.append(round(rt.x,4))
                # tdata.append(round(rt.y, 4))
                # tdata.append(round(rb.x,4))
                # tdata.append(round(rb.y, 4))
                ttdata =  [tdata]
                aadata = [angledata]
                test = ear_model.predict(ttdata)
                test_a = angle_model.predict(aadata)
                test_m = mouth_model.predict([[mouth_ratio]])
                tfont=cv2.FONT_HERSHEY_SIMPLEX

                if test[0] == 0 and temp == 0:
                    real = 0
                elif test[0] == 1 and temp == 1:
                    real = 1
                
                if real == 0:
                    #print(tdata)
                    cv2.putText(image, 'open', (0,40) , tfont, 1,(0,255,0),2)
                elif real == 1:
                    #print(tdata)
                    cv2.putText(image, 'close', (0,40) , tfont, 1,(0,255,0),2)

                if test_a[0] == 0:
                    cv2.putText(image, 'normal', (0,80) , tfont, 1,(0,255,0),2)
                elif test_a[0] == 1:
                    cv2.putText(image, 'down', (0,80) , tfont, 1,(0,255,0),2)
                
                if round(test_m[0], 3) > 0.96:
                    cv2.putText(image, 'yawn', (0,120) , tfont, 1,(0,255,0),2)
                elif round(test_m[0], 3) <= 0.96:
                    cv2.putText(image, 'no yawn', (0,120) , tfont, 1,(0,255,0),2)
                
                temp = test[0]
                #print(test_m[0])
            
            cv2.imshow('Video', image)
            if cv2.waitKey(1) > 0:
                break