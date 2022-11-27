# -*- coding: utf-8 -*-
"""raspberry.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7Hs0IqL0Z0bh1N4GACSNuYd74ivu2lg
"""

import serial
import PRi.GPIO as GPIO
import time

#부저 공통
buzzer = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setwarnings(False)

"""
#부저가 1.5초동안 1회 울리고 종료
pwm = GPIO.PWM(buzzer,523)
pwm.start(50.0)
time.sleep(1.5) #1.5초

pwm.stop()
GPIO.cleanup()

#부저가 1초 간격으로 도,레,미 음 3번 반복 후 종료
pwm = GPIO.PWM(buzzer, 1.0) #초기 주파수 1hz로 설정
pwm.start(50.0)

for cnt in range(0, 3):
  pwm.ChangeFrequency(262)
  time.sleep(1.0)
  pwm.ChangeFrequency(294)
  time.sleep(1.0)
  pwm.ChangeFrequency(330)
  time.sleep(1.0)

pwm.ChangeDutyCycle(0.0)

pwm.stop()
GPIO.cleanup()

#부저가 도,레,미,파,솔,라,시,도 출력 후 종료( 이 부분 8번줄 없어도 됨 )
pwm = GPIO.PWM(buzzer, 1.0) #초기 주파수 1hz로 설정
pwm.start(50.0)

scale = [262,294,330,349,392,440,494,523] #4옥타브

for i in range(0,8):
  pwm.ChangeFrequency(scale[i])
  time.sleep(1.0)

pwm.stop()
GPIO.cleanup()
"""

#co2 센서 값 받아오기
co2 = serial.Serial('/dev/ttyACM0', 9600)
co2.flushInput() #받은 데이터 폐기

while True:
  y = co2.readline()
  y = y.decode()[:-2]
  ppm = float(y)

  print(ppm)

  if ppm > 600.0:
    pwm = GPIO.PWM(buzzer, 32)
    pwm.start(50.0)
    time.sleep(1.0)
    pwm.stop()
  
  time.sleep(10.0)

#라즈베리파이 부팅 시 파이썬 자동 실행