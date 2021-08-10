import cv2
import numpy as np
import webbrowser
from os import listdir
from os.path import isfile, join,split

data_path = 'E:/MINOR PROJ2/Dataset/'
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

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(15,160,190),2)

        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))


    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        id,result = model.predict(face)
        print(result)
        print(id)


        if id==1 and result<45:
            display_string ='HEY SARAZUL, PRESS ENTER'
        elif id==2 and result<45:
            display_string = 'HEY NOAMAAN, PRESS ENTER'
        elif id==3 and result<45:
            display_string= 'HEY ZUCK PRESS ENTER'
        elif id==4 and result<45:
            display_string='HEY JASMINE PRESS ENTER'
        elif id==5 and result<45:
            display_string=''
        else:
            display_string = "      USER NOT MATCHED"
        cv2.rectangle(image, (0, 0), ( 640, 60), (163,194,194), -1)
        cv2.putText(image,display_string,(60,40), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        if result<45:
            cv2.rectangle(image, (0, 420), (640, 480), (163, 194, 194), -1)
            cv2.putText(image, "UNLOCKED", (250, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 106, 8), 2)
            cv2.imshow('Face Cropper', image)

            if cv2.waitKey(1)==13:
                if id==1:
                    webbrowser.open("http://isarazulali.cf/?i=1",new=2)
                    break
                elif id==2:
                    webbrowser.open("www.google.co.in",new=2)
                    break
                elif id==3:
                    webbrowser.open("https://en.wikipedia.org/wiki/Mark_Zuckerberg",new=2)
                    break
                elif id==4:
                    webbrowser.open("",new=2)
                    break
                elif id==5:
                    webbrowser.open("",new=2)
                    break

        else:
            cv2.rectangle(image, (0, 420), (640, 480), (163, 194, 194), -1)
            cv2.putText(image, "LOCKED", (250, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.rectangle(image, (0, 0), (640, 60), (163, 194, 194), -1)
        cv2.rectangle(image, (0, 420), (640, 480), (163, 194, 194), -1)
        cv2.putText(image, "FACE NOT FOUND", (185, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()