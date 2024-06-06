import cv2
facemodel=cv2.CascadeClassifier('face.xml')
vid=cv2.VideoCapture('maskvid.mp4')
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)
        for(x,y,l,h) in faces:
            cv2.rectangle(frame,(x,y),(x+l,y+h),(127,0,255),3)
        cv2.namedWindow('desh window',cv2.WINDOW_NORMAL)
        cv2.imshow('desh window',frame)
        k=cv2.waitKey(10) #FPS: 1000/24
        if(k==ord('x')):
            break
    else:
        break
cv2.destroyAllWindows()
        