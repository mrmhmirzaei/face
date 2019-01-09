import cv2
from time import sleep
import pickle

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

camera = cv2.VideoCapture(0)

        # Settings #
border_color = (255, 0, 0)
border_witdh = 2
text_color = (255, 255, 255)
text_font = cv2.FONT_HERSHEY_SIMPLEX

labels = {}
with open("labels.pickle", "rb") as file:
    data = pickle.load(file)
    labels = { v:k for k,v in data.items() }

def detecte():
    _ , frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.flip(frame, 1)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        recognize(gray, frame, x, y, w, h)
        cv2.rectangle(frame, (x , y), (x+w , y+h), border_color, border_witdh)

    cv2.imshow('', frame)

def recognize(gray, frame, x, y, w, h):
    face = gray[y:y+h, x:x+w]
    _id, percent = recognizer.predict(face)
    if 50 <= percent and percent <= 70:
        name = "None"
        if _id in labels:
            name = labels[_id]
        cv2.putText(frame, name, (x,y-10), text_font, 1, text_color, 2, cv2.LINE_AA)


detecte()
while(True):
    detecte()
    cv2.waitKey(1)

cv2.destroyAllWindows()
