import numpy as np
import cv2
import pickle 

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle","rb") as f:
    org_labels = pickle.load(f)
    labels = {v:k for k,v in org_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors =5)
    
    ####
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        id, conf = recognizer.predict(roi_gray)
        if conf >=70:
            print(id)
            print(labels[id])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item = "my-img.png"
        cv2.imwrite(img_item,roi_color)

        color = (255,0,0) #BGR
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w,y+h),color,stroke)
        smile = smile_cascade.detectMultiScale(roi_gray,scaleFactor = 1.5, minNeighbors =5)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_gray, (sx,sy), (sx+sw,sy+sh),(0,255,0),stroke)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()