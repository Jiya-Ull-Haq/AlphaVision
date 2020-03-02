import cv2
import pickle
#creating CascadeClassifier Object
face_cascade = cv2.CascadeClassifier("/Users/shaik/PycharmProjects/cascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/Users/shaik/PycharmProjects/venv/trainner.yml")

lables = {'Persons_name' : 1}
with open("/Users/shaik/PycharmProjects/venv/labels.pickle", 'rb') as f:
      og_labels = pickle.load(f)
      lables = {v:k for k,v in og_labels.items()}

# Reading the face
cap = cv2.VideoCapture(0)
while cap.isOpened():    # Capturing the video
    _, frame = cap.read()

#now reading image as grayscale(this is face classifier)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray ,1.1 ,4)

    #print(type(faces))
    print(faces)
 #this is for iteration
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #[cord1-height, cord2-height]  /setting face coordinates/
        roi_color = frame[y:y+h, x:x+w]

        id_,conf =recognizer.predict(roi_gray)
        if conf>=0 and conf <=100:
            print(id_)
            print(lables[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = lables[id_]
            color = (255,255,0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "Face_captured.png" # capturing picture
        cv2.imwrite(img_item, roi_gray)
        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)
 #display output
    cv2.imshow("face_detection proj",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()