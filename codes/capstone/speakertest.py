import serial
import time
import RPi.GPIO as GPIO


buzzer = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
pwm = GPIO.PWM(buzzer, 523)

scale = [262, 330, 392, 494, 523]
l = 0
while True:
    if l > 4:
        l = 0
    
    pwm.ChangeFrequency(scale[l])
    pwm.start(50.0)
    time.sleep(3)
    l += 1
    pwm.stop()
    time.sleep(1)
