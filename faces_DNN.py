import numpy as np
import cv2
import pickle 

modelFile = r"Inputs\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"Inputs\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
conf_threshold = .5

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    image = frame
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = 
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape)
    bboxes = []
    #for i in range(0, detections.shape[2]):
    #    confidence = detections[0, 0, i, 2]
    #    print(confidence)
    #    if confidence > conf_threshold:
    #        x1 = int(detections[0, 0, i, 3] * w)
    #        y1 = int(detections[0, 0, i, 4] * h)
    #        x2 = int(detections[0, 0, i, 5] * w)
    #        y2 = int(detections[0, 0, i, 6] * h)
    
    #        color = (255,0,0) #BGR
    #        stroke = 2
    #        cv2.rectangle(frame, (x1,y1), (x2,y2),color,stroke)
    
    # Display the resulting frame
    for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
        confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
        if confidence > .6:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            roi_gray = gray[startY:endY,startX:endX]
            id, conf = recognizer.predict(roi_gray)
            if conf >=70:

                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name, (startX,startY),font,1,color,stroke,cv2.LINE_AA)
        # draw the bounding box of the face along with the associated
        # probability
            #text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                      (255, 0, 0), 2)
            #cv2.putText(image, text, (startX, y),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)  
        cv2.imshow("Frames",image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()