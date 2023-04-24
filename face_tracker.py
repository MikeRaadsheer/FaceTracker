import cv2
import time

cascPath = "haarcascade.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
                
time.sleep(0.1)

while True:
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if int(cv2.__version__.split('.')[0]) >= 3:
        cv_flag = cv2.CASCADE_SCALE_IMAGE
    else:
        cv_flag = cv2.cv.CV_HAAR_SCALE_IMAGE

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv_flag
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
