# -*- coding: utf-8 -*-

import cv2
print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Yogish Pai and Aditya Baliga B')

cascade_src = 'cars.xml' #car
cascade_src2 = 'two_wheeler.xml' #Bike
video_src = 'video1.avi'

cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)
car_cascade2 = cv2.CascadeClassifier(cascade_src2)
count_car = 0



font = cv2.FONT_HERSHEY_PLAIN


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    bike = car_cascade2.detectMultiScale(gray, 1.1, 2)

    cv2.line(img,(20,30),(300,30),(0, 0, 255),2)
    cv2.putText(img, "Count:" + str(count_car), (150,20), font, 1, (0, 0, 255), 2)
    for (x,y,w,h) in cars:
        if(x < 30):
            count_car=count_car+1
            print("CAR")
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,99,77),1)
        cv2.putText(img, "100 km/hr", (x, y), font, 1.1, (255,0,0), 1)
        
    for (x,y,w,h) in bike:
        if(x < 30):
            print("BIKE")
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video', img)
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
