import cv2
import time
import pandas as pd
import numpy as np
import mediapipe as mp
#import sklearn
import serial
import RPi.GPIO as GPIO
import joblib
from scipy.spatial import distance
import warnings

warnings.filterwarnings('ignore')

buzzer = 18
LED = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setup(LED, GPIO.OUT)
GPIO.setwarnings(False)
co2 = serial.Serial('/dev/ttyACM0', 9600)
co2.flushInput()

def point_dist(p1, p2):
    d = distance.euclidean([p1.x, p1.y], [p2.x, p2.y])
    return d

def calculate_EAR(p1, p2, p3):
    ear = (p1 + p2) / (2.0 * p3)
    return ear

ear_model = joblib.load('./eyes_model_2_real.pkl')
angle_model = joblib.load('./angle_model.pkl')
mouth_model = joblib.load('./mouth_model.pkl')
mp_face_mesh = mp.solutions.face_mesh

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
prev_time = 0
FPS = 2

eye_sec = 2 
eye_frame = 0
eye_isclosed = False

angle_sec = 2
angle_frame = 0
angle_frame = 0
angle_isangled = False

yawn_sec = 6
yawn_frame = 0
yawn_count = 0
yawn_maxcount = 3
yawn_isyawned = False
yawn_iscounted = False

sound_time = 0
sound_prev = 0
sound_interval = 0.5

nowco2 = 0

temp = 0
real = 0

hdp = 200
nosetp = 6
nosebp = 94 

use_sound = False
pwm = GPIO.PWM(buzzer, 523)

pwm_time = time.time()



def sound2(r, do):
    global pwm
    global pwm_time
    scale = [262,330,392,494,523]
    if do:
        pwm.ChangeFrequency(scale[0])
        pwm.start(50.0)
        pwm_time = time.time()
    else:
        pwm.stop()

def sound_stop():
    pwm.stop()



def sound(r):
    global sound_time
    global sound_prev
    global sound_interval
    global use_sound

    sound_time = time.time() - sound_prev
    if sound_time > sound_interval:
        sound_prev = time.time()
        print('sound')
        use_sound = ~use_sound
        #sound2(r, use_sound)
        sound2(r, True)



def framecount():
    global eye_isclosed
    global eye_frame
    global yawn_isyawned
    global yawn_frame
    global yawn_iscounted
    global angle_isangled
    global angle_frame
    
    if eye_isclosed:
        eye_frame += 1
    else:
        sound_stop()
        eye_frame = 0
    
    if yawn_isyawned:
        yawn_frame += 1
    else:
        yawn_iscounted = False
        yawn_frame = 0
        sound_stop()
    
    if angle_isangled:
        angle_frame += 1
    else:
        sound_stop()
        angle_frame = 0


with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    ret, image = video.read()
    if ret is True :
        print('camera on')
    else:
        print('camera off')
    while True:
        

        if(time.time() - pwm_time >= 2):
            pwm.stop()
        ret, image = video.read()

        current_time = time.time() - prev_time

        if (ret is True) and (current_time > 1./FPS):
            cy = co2.readline()
            cy = cy.decode()[:-2]
            ppm = 0
            cyl = cy.find('Co2 : ')
            if cyl != -1:
                c2str = cy[cyl:]
                c2str = c2str[-2:]
                print(c2str)

            if ppm >= 1500:
                GPIO.output(LED, True)
            else:
                GPIO.output(LED, False)
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

                
                if test[0] == 0:
                    #print(tdata)
                    eye_isclosed = False
                    print('open')
                    
                elif test[0] == 1:
                    #print(tdata)
                    eye_isclosed = True
                    print('close')

                if test_a[0] == 0:
                    angle_isangled = False
                    print('normal')
                
                elif test_a[0] == 1:
                    angle_isangled = True
                    print('down')
                
                if round(test_m[0], 3) > 0.96:
                    yawn_isyawned = True
                    print('yawn')
                
                elif round(test_m[0], 3) <= 0.96:
                    yawn_isyawned = False
                    print('no yawn')
                
                temp = test[0]
                framecount()

                if eye_frame >= eye_sec * FPS:
                    print('eye sound')
                    sound(1)

                if angle_frame >= angle_sec * FPS:
                    print('angle sound')
                    sound(1)

                if yawn_frame >= yawn_sec * FPS:
                    if yawn_iscounted is False:
                        yawn_count += 1
                        yawn_iscounted = True
            else:
                sound_stop()
            #cv2.imshow('Video', image)
            if cv2.waitKey(1) > 0:
                break