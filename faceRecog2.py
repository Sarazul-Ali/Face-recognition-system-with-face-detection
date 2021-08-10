import cv2
import numpy as np
from os import listdir
from os.path import isfile, join,split


#Trainer

data_path = "E:/MINOR PROJ2/Dataset/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]



Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype="uint8"))
    Labels.append(split(image_path)[1].split(".")[1])

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

#RECOGNITION

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id=0
display_string=""

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    cv2.rectangle(frame, (0, 0), (640, 60), (224, 224, 224), -1)
    cv2.rectangle(frame, (0, 420), (640, 480), (224, 224, 224), -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


        id, result = model.predict(gray[y:y+h, x:x+w])

        #MATCHING

        if id==1 and result<55:
            display_string = "NAME: NOAMAAN"
        elif id==2 and result<55:
            display_string ="NAME: LAIBA"
        elif id == 3 and result < 55:
            display_string = "NAME: ARIBA"
        elif id==4 and result<55:
            display_string =  "NAME: MARYAM"
        elif id==5 and result<55:
            display_string =  "NAME: SARAZUL"
        elif id==6 and result<55:
            display_string =  "NAME: XYZ"
        else:
            display_string = "NOT FOUND"



        wd,hei=cv2.getTextSize(display_string,cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y - 50), (x +wd+3 , y - 10), (0, 255, 255), -1)
        cv2.putText(frame, display_string, (x+4, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)






    if faces is():
         cv2.putText(frame, "FACE NOT FOUND", (185, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)



    cv2.imshow('Face Cropper', frame)
    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()
