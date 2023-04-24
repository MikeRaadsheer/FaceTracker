import os
import cv2
import time
import pickle
import datetime
import numpy as np


now = datetime.datetime.now()



vid = "videos/"+str(now)+".avi"
filename = vid.replace(":", ";").replace(" ", "").lower()
fps = 30
myRes = '720p'
line_length = 13
num = 0

print('\n\n---------------------------------------------------------------------------------------')
print(filename)
print('---------------------------------------------------------------------------------------\n')
print('=======================================================================================')


log = open(filename.replace(".avi", "")+"LOG.txt",'w')
log.write("Video : " + filename.replace(";", ":").replace("videos/", "") + '\n'*2)


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
left_eye_cascade = cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

time.sleep(0.1)

def setRes(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS = {
   # "240p": (352, 240),
   # "360p": (480, 360),
   # "480p": (858, 480),
    "720p": (1280, 720) ,
   # "1080p": (1920, 1080),
   # "1440p": (2560, 1440),
   # "4k": (3840, 2160),
   # "8k": (7680, 4320),
}

def getDims(cap, res=myRes):
    width, height = STD_DIMENSIONS["720p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    setRes(cap, width, height)
    return width, height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def getVideoType(filename):
    filename, ext = os.path.splitext(filename)
    if ext in  VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, getVideoType(filename), fps, getDims(cap, myRes))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"preson_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


def draw_border(frame, point1, point2, point3, point4, line_length):

    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4

    cv2.line(frame, (x1, y1), (x1 , y1 + line_length), (0, 255, 0), 2)  #-- top-left
    cv2.line(frame, (x1, y1), (x1 + line_length , y1), (0, 255, 0), 2)

    cv2.line(frame, (x2, y2), (x2 , y2 - line_length), (0, 255, 0), 2)  #-- bottom-left
    cv2.line(frame, (x2, y2), (x2 + line_length , y2), (0, 255, 0), 2)

    cv2.line(frame, (x3, y3), (x3 - line_length, y3), (0, 255, 0), 2)  #-- top-right
    cv2.line(frame, (x3, y3), (x3, y3 + line_length), (0, 255, 0), 2)

    cv2.line(frame, (x4, y4), (x4 , y4 - line_length), (0, 255, 0), 2)  #-- bottom-right
    cv2.line(frame, (x4, y4), (x4 - line_length , y4), (0, 255, 0), 2)

    return frame

while(True):
    ret, frame = cap.read()


    if int(cv2.__version__.split('.')[0]) >= 3:
        cv_flag = cv2.CASCADE_SCALE_IMAGE
    else:
        cv_flag = cv2.cv.CV_HAAR_SCALE_IMAGE

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # left_eyes = left_eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # right_eyes = right_eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print("Face : ",x,y,w,h)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        img_item = "people/mike-raadsheer/miker"+str(num)+".png"
        num += 1

        '''
        SO DIS A COMMENT
        HUH?!
        '''
        

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 60 and conf <= 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            print("Time : " + str(datetime.datetime.now())[:19] + " | ID : " + str(id_) + " | Name : " + name.replace("-", " ") + " | Similarity : " + str(conf)[:4] + '%')
            log.write("Time : " + str(now) + " | ID : " + str(id_) + " | Name : " + name.replace("-", " ") + " | Similarity : " + str(conf)[:4] + '%\n')

            point1, point2, point3, point4 = (x,y), (x,y+h), (x+w,y), (x+w,y+h)
            cv2.putText(frame, name, (x, y - 20), font, 1, color, stroke, cv2.LINE_AA)
            draw_border(frame, point1, point2, point3, point4, line_length)


        # For adding a new person tot the system--------------------------------
        # if conf >= 70:
        #     cv2.imwrite(img_item.replace(" ", ""), roi_gray)
        #     print(img_item)

        # for (x, y, w, h) in smiles:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        #    print("Smile : ",x,y,w,h)

        # for (x, y, w, h) in left_eyes:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #    print("Left Eye : ",x,y,w,h)

        # for (x, y, w, h) in right_eyes:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #    print("Right Eye : ",x,y,w,h)
        #----------------------------------------------------------------------
        out.write(frame)
        cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print('=======================================================================================\n')
log.close()
cap.release()
cap.release()
cv2.destroyAllWindows()
