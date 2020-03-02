import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "/Users/shaik/PycharmProjects/Training imgs")

face_cascade = cv2.CascadeClassifier('/Users/shaik/PycharmProjects/cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
          path = os.path.join(root, file)

          label = os.path.basename(root).replace(" ","-").lower()
          print(label, path)
          if not label in label_ids:
              label_ids[label] = current_id




              
              current_id += 1
          id_ = label_ids[label]
          print(label_ids)
          #y_labels.append(label)
          #x_train.append(path) #turn into numpy array, GRAY.
          pil_image = Image.open(path).convert("L") #gray scale
          Image_array = np.array(pil_image, "uint8")
          print(Image_array)
          faces = face_cascade.detectMultiScale(Image_array, 1.1, 4)

          for (x,y,w,h) in faces:
              roi = Image_array[y:y+h, x:x+w]
              x_train.append(roi)
              y_labels.append(id_)
#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")