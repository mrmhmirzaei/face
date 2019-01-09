import os
import cv2
import numpy as np
from PIL import Image
import pickle

IMAGE_PATH = "./images"
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

_id = 0
labels_list = {}
labels = []
train = []

for root, dirs, files in os.walk(IMAGE_PATH):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root)
        if label in labels_list:
            pass
        else:
            labels_list[label] = _id
            _id+=1
        id_ = labels_list[label]
        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image,"uint8")
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = image_array[y:y+h, x:x+w]
            train.append(face)
            labels.append(id_)

with open("labels.pickle", "wb") as file:
    pickle.dump(labels_list, file)

recognizer.train(train, np.array(labels))
recognizer.save("trainer.yml")
