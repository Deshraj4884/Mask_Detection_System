import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
st.title('FACE MASK DETECTION SYSTEM')
st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7hIEkiQkp4L32BsGJXA4HQ4_bm9XbKmYJLQ&s')
choise=st.sidebar.selectbox('Menu',('HOME','URL','CAMERA'))
if(choise=='HOME'):
    st.image('https://www.researchdive.com/blogImages/NfBEa3zk4o.jpeg')
elif(choise=='URL'):
    url=st.text_input('Enter your URL')
    btn=st.button('Start Detection')
    window=st.empty()
    if btn:
        i=1
        btn2=st.button('Stop Detection')
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier('face.xml')
        maskmodel=load_model('mask.h5',compile=False)
        vid=cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for(x,y,l,h) in faces:
                    #cropping the Faces
                    face_img=frame[y:y+h,x:x+l]
                    #Resize the cropped face
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    #converting shep of image according to desire dimension of model
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    #normalize the image
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path='NO MASK/'+str(i)+'.jpg'
                        cv2.imwrite(path,frame[y:y+h,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+h),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+h),(0,255,0),3)
                window.image(frame,channels='BGR')
       
elif(choise=='CAMERA'):
    cam=st.selectbox('Choose Camera',('None','Primary','Secondary'))
    btn=st.button('Start Detection')
    window=st.empty()
    if btn:
        i=1
        btn2=st.button('Stop Detection')
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier('face.xml')
        maskmodel=load_model('mask.h5')
        if cam=='Primary':
            cam=0
        else:
            cam=1
        vid=cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for(x,y,l,h) in faces:
                    #cropping the Faces
                    face_img=frame[y:y+h,x:x+l]
                    #Resize the cropped face
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    #converting shep of image according to desire dimension of model
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    #normalize the image
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path='NO MASK/'+str(i)+'.jpg'
                        cv2.imwrite(path,frame[y:y+h,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+h),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+h),(0,255,0),3)
                window.image(frame,channels='BGR')
       
