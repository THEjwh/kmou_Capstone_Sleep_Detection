import time as tm
import serial
import PRi.GPIO as GPIO



begin= tm.time()
end =tm.time()
turm = round(end - begin,2)

#부저 공통
buzzer = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setwarnings(False)

#co2 센서 값 받아오기
co2 = serial.Serial('/dev/ttyACM0', 9600)
co2.flushInput() #받은 데이터 폐기

while True:
  y = co2.readline()
  y = y.decode()[:-2]
  ppm = int(y)

  print(ppm)


global yawn_count = 0

#하품 카운터 
if(yawn_state >0.96 and turm >=6):
     yawn_count=yawn_count+1 

#co2농도가 높은데 하품을 1회 하였는가
if(ppm>=1500 and yawn_count ==1){
     }


#하품을 3번 했다면 알람을 울리고 리셋하라
if(yawn_count==3 ):
    if(1000<ppm<=2000):
         alam =1
         yawn_count =0

    elif (2000<ppm<=3000):
        alam =2
        yawn_count =0

    elif (3000<ppm<=4000):
        alam =3
        yawn_count =0
        
    else:
        alam =4
        yawn_count =0


#눈과 고개가 2초 이상 정적이면 알람을 울려라
if(eyes_state>0.96 and turm>=2) or (face_angle >0.96 and turm>=2):
    if(1000<ppm<=2000):
         alam =1

    elif (2000<ppm<=3000):
        alam =2

    elif (3000<ppm<=4000):
        alam =3
        
    else:
        alam =4
    

     


