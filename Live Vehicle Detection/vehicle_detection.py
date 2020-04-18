import cv2
import numpy as np

cascade_src = 'cars.xml'

cap = cv2.VideoCapture(0)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLORMAP_AUTUMN)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    s=np.array(cars)
    l=list(s.shape)
    print("Number of Square Draw:",l[0])
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()