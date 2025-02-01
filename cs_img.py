from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("D:\\innoboard_ai\\esp32cam_AI\\converted_keras\\keras_model.h5", compile=False)

# Load the labels
class_names = open("D:\\innoboard_ai\\esp32cam_AI\\converted_keras\\labels.txt", "r").readlines()

# เลือกไฟล์ภาพที่ต้องการทำนาย
image_path = "D:\\innoboard_ai\\esp32cam_AI\\Input\\tech\\b3.jpg"

# โหลดภาพจากไฟล์
image = cv2.imread(image_path)

# Resize ภาพให้เป็นขนาด 224x224 ตามที่โมเดลต้องการ
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image_show = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

# แสดงภาพ (สามารถลบได้หากไม่ต้องการ)
cv2.imshow("Input Image", image_show)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# แปลงภาพเป็นอาเรย์ NumPy และปรับขนาดให้ตรงกับโมเดล
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize ค่า pixel ให้อยู่ในช่วง -1 ถึง 1
image = (image / 127.5) - 1

# ทำการพยากรณ์ด้วยโมเดล
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

# แสดงผลลัพธ์
print(f"Class: {class_name}")
print(f"Confidence Score: {confidence_score * 100:.2f}%")
