import cv2
import numpy as np
from keras.models import load_model

face_cascade_path = "D:\\nigga_detect\\haarcascade_frontalface_default.xml"
model_path = "D:\\nigga_detect\\model\\converted_keras\\keras_model.h5"
labels_path = "D:\\nigga_detect\\model\\converted_keras\\labels.txt"

face_classifier = cv2.CascadeClassifier(face_cascade_path)

model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").readlines()

# image_path = "D:\\nigga_detect\\dataset\\nigga_face\\6_face_1.jpg"  
image_path = "D:\\nigga_detect\\dataset\\white_monkey_face\\3_face_1.jpg"  
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(image_bw, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f'There are {len(faces)} faces found.')

for (x, y, w, h) in faces:

    face = image_bgr[y:y + h, x:x + w]

    face_resized = cv2.resize(face, (224, 224))

    face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    face_array = (face_array / 127.5) - 1  

    prediction = model.predict(face_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    label = f"{class_name}: {np.round(confidence_score * 100, 2)}%"
    print(label)
    cv2.putText(image_bgr, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

image_resized = cv2.resize(image_bgr, (500, 500))

cv2.namedWindow("Face Detection and Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Detection and Classification", 500, 500)

cv2.imshow("Face Detection and Classification", image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()