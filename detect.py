import numpy as np
import cv2
from imutils.video import FPS
face_detect=cv2.CascadeClassifier('C:\Users\Adithya\Desktop\isi\haarcascade_frontalface_default.xml')
eye_detect=cv2.CascadeClassifier("isi\haarcascade_eye.xml")
cap=cv2.VideoCapture('C:\Users\Adithya\Desktop\sample.mp4')
#cap = cv2.VideoCapture('C:\Users\Adithya\Videos\HDRip x264 AAC...Hon3y.mkv')
fps=FPS().start()
while True:
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_detect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
		roi_gray=gray[x:x+w,y:y+h]
		roi_color=img[x:x+w,y:y+h]
		eyes=eye_detect.detectMultiScale(roi_gray)	
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),3)
	

	cv2.imshow("final",img)
	cv2.waitKey(30) 
	fps.update()
fps.stop()
cv2.destroyAllWindows
cap.release()
		
	
