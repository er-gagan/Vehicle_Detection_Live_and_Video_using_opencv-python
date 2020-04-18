import cv2
import numpy as np

cascade_src = 'cars.xml'

video_src = 'video1.avi'
# video_src = 'video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    color = cv2.cvtColor(img, cv2.COLORMAP_AUTUMN)
    
    cars = car_cascade.detectMultiScale(color, 1.1, 1)
    s=np.array(cars)
    print("Number of Shape Draw:",s.shape[0])
    
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()