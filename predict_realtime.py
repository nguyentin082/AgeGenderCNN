import numpy as np
import cv2
from keras.models import load_model
import keras.losses
import matplotlib.pyplot as plt

# Load the model
model_path = "models/100Epochs/model.h5"
keras.losses.mse = keras.losses.MeanSquaredError()
model = load_model(model_path, custom_objects={'mse': keras.losses.mse})

# Load face detection model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

def predict_age_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 4)
        cv2.putText(frame, "Age : "+str(int(predict[0]))+" / "+str(gend), (x, y), cv2.FONT_HERSHEY_SIMPLEX, w*0.005, col, 4)
    return frame, age_, gender_

# Access webcam
cap = cv2.VideoCapture(0)
plt.ion()  # Turn on interactive mode for matplotlib
fig, ax = plt.subplots()
ret, frame = cap.read()
output_image, _, _ = predict_age_gender(frame)
im = ax.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict age and gender for faces in the frame
    output_image, age_, gender_ = predict_age_gender(frame)

    # Update the image content instead of redrawing the whole plot
    im.set_data(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)

    # Check for keyboard events to exit
    if plt.waitforbuttonpress(0.001):
        break

# Release resources
cap.release()
plt.close(fig)
