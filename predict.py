import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import keras.losses
import os

# Đường dẫn đến mô hình và tệp ảnh
model_path = "models/100Epochs/model.h5"
img_path = "images/pam.jpg"
output_folder = "results"

# Tạo thư mục "result" nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Đường dẫn lưu ảnh sau khi dự đoán
output_path = os.path.join(output_folder, "pam.jpg")

# Tải mô hình
keras.losses.mse = keras.losses.MeanSquaredError()
model = load_model(model_path, custom_objects={'mse': keras.losses.mse})

# Tải mô hình phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Đọc ảnh và chuyển sang ảnh xám
pic = cv2.imread(img_path)
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Dự đoán tuổi và giới tính cho từng khuôn mặt
age_ = []
gender_ = []
for (x, y, w, h) in faces:
    img = gray[y-50:y+40+h, x-10:x+10+w]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (200, 200))
    predict = model.predict(np.array(img).reshape(-1, 200, 200, 3))
    age_.append(predict[0])
    gender_.append(np.argmax(predict[1]))
    gend = np.argmax(predict[1])
    if gend == 0:
        gend = 'Man'
        col = (255, 0, 0)
    else:
        gend = 'Woman'
        col = (203, 12, 255)
    cv2.rectangle(pic, (x, y), (x+w, y+h), (0, 225, 0), 4)
    cv2.putText(pic, "Age : " + str(int(predict[0])) + " / " + str(gend), (x, y), cv2.FONT_HERSHEY_SIMPLEX, w*0.005, col, 4)

# Hiển thị ảnh sau khi dự đoán
pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.imshow(pic1)
plt.show()

# In ra tuổi và giới tính dự đoán
print(age_, gender_)

# Lưu ảnh sau khi dự đoán vào thư mục "result"
cv2.imwrite(output_path, pic)
