import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, Input
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import seaborn as sns


# Load dữ liệu
path = "UTKFace"
pixels = []
age = []
gender = []
for img in os.listdir(path):
    ages = img.split("_")[0]
    genders = img.split("_")[1]
    img = cv2.imread(str(path) + "/" + str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))  # Thêm bước chia tỷ lệ ảnh
    pixels.append(np.array(img))
    age.append(np.array(ages))
    gender.append(np.array(genders))
age = np.array(age, dtype=np.int64)
pixels = np.array(pixels) / 255.0  # Chuẩn hóa giá trị pixel
gender = np.array(gender, np.uint64)

# Shuffle dữ liệu
pixels, age, gender = shuffle(pixels, age, gender, random_state=42)

# Chia dữ liệu thành train, validation và test (60%, 20%, 20%)
num_samples = pixels.shape[0]
train_end = int(0.6 * num_samples)
val_end = int(0.8 * num_samples)

x_train_img = pixels[:train_end]
y_train_age = age[:train_end]
y_train_gender = gender[:train_end]

x_val_img = pixels[train_end:val_end]
y_val_age = age[train_end:val_end]
y_val_gender = gender[train_end:val_end]

x_test_img = pixels[val_end:]
y_test_age = age[val_end:]
y_test_gender = gender[val_end:]

# In kích thước của các tập dữ liệu
print("Total number of images:", num_samples)
print("Training data shape:", x_train_img.shape)
print("Training labels shape (age):", y_train_age.shape)
print("Training labels shape (gender):", y_train_gender.shape)
print("Validation data shape:", x_val_img.shape)
print("Validation labels shape (age):", y_val_age.shape)
print("Validation labels shape (gender):", y_val_gender.shape)
print("Test data shape:", x_test_img.shape)
print("Test labels shape (age):", y_test_age.shape)
print("Test labels shape (gender):", y_test_gender.shape)

# Định nghĩa mô hình
input = Input(shape=(200, 200, 3))
conv1 = Conv2D(140, (3, 3), activation="relu")(input)
conv2 = Conv2D(130, (3, 3), activation="relu")(conv1)
batch1 = BatchNormalization()(conv2)
pool3 = MaxPool2D((2, 2))(batch1)
conv3 = Conv2D(120, (3, 3), activation="relu")(pool3)
batch2 = BatchNormalization()(conv3)
pool4 = MaxPool2D((2, 2))(batch2)
flt = Flatten()(pool4)
# age
age_l = Dense(128, activation="relu")(flt)
age_l = Dense(64, activation="relu")(age_l)
age_l = Dense(32, activation="relu")(age_l)
age_l = Dense(1, activation="relu", name="age_output")(age_l)
# gender
gender_l = Dense(128, activation="relu")(flt)
gender_l = Dense(80, activation="relu")(gender_l)
gender_l = Dense(64, activation="relu")(gender_l)
gender_l = Dense(32, activation="relu")(gender_l)
gender_l = Dropout(0.5)(gender_l)
gender_l = Dense(2, activation="softmax", name="gender_output")(gender_l)

# Tùy chỉnh tốc độ học
learning_rate = 0.001  # Bạn có thể thay đổi giá trị này để tùy chỉnh tốc độ học
optimizer = Adam(learning_rate=learning_rate)

# TRAINING
model = Model(inputs=input, outputs=[age_l, gender_l])
model.compile(optimizer=optimizer, loss=["mse", "sparse_categorical_crossentropy"], metrics=['mae', 'accuracy'])
save = model.fit(x_train_img, [y_train_age, y_train_gender], validation_data=(x_val_img, [y_val_age, y_val_gender]), epochs=50, batch_size=32)  # 50
model.save("model.h5")

# PLOTTING
# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(save.history['loss'], label='Training Loss')
plt.plot(save.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Plot Accuracy for Gender Prediction
plt.figure(figsize=(12, 6))
plt.plot(save.history['gender_output_accuracy'], label='Training Accuracy')  # Sử dụng 'gender_output_accuracy'
plt.plot(save.history['val_gender_output_accuracy'], label='Validation Accuracy')  # Sử dụng 'val_gender_output_accuracy'
plt.title('Training and Validation Accuracy for Gender Prediction')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# Plot MAE for Age Prediction
plt.figure(figsize=(12, 6))
plt.plot(save.history['age_output_mae'], label='Training MAE')  # Sử dụng 'age_output_mae'
plt.plot(save.history['val_age_output_mae'], label='Validation MAE')  # Sử dụng 'val_age_output_mae'
plt.title('Training and Validation MAE for Age Prediction')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('mae_plot.png')
plt.show()




# TESTING
age_pred, gender_pred = model.predict(x_test_img)

# Kiểm tra một số dự đoán
for i in range(10):
    print(f"True Age: {y_test_age[i]}, Predicted Age: {age_pred[i][0]}")
    print(f"True Gender: {y_test_gender[i]}, Predicted Gender: {np.argmax(gender_pred[i])}")

# Test Accuracy for Gender Prediction
correct_gender_predictions = np.sum(np.argmax(gender_pred, axis=1) == y_test_gender)
gender_accuracy = correct_gender_predictions / y_test_gender.shape[0]
print("Test Accuracy for Gender Prediction: {:.2f}%".format(gender_accuracy * 100))

# Confusion Matrix for Gender Prediction
conf_matrix = confusion_matrix(y_test_gender, np.argmax(gender_pred, axis=1))
print("Confusion Matrix for Gender Prediction:\n", conf_matrix)
print("Classification Report for Gender Prediction:\n", classification_report(y_test_gender, np.argmax(gender_pred, axis=1)))
# Vẽ confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Gender Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Lưu hình ảnh confusion matrix dưới dạng PNG
plt.savefig('confusion_matrix.png')
plt.show()






# MAE for Age Prediction on Test Data
age_mae = np.mean(np.abs(age_pred - y_test_age))
print("Test MAE for Age Prediction: {:.2f}".format(age_mae))

# RMSE for Age Prediction on Test Data
age_rmse = np.sqrt(mean_squared_error(y_test_age, age_pred))
print("Test RMSE for Age Prediction: {:.2f}".format(age_rmse))

# Vẽ biểu đồ scatter so sánh tuổi thật và tuổi dự đoán
plt.figure(figsize=(12, 6))
plt.scatter(y_test_age, age_pred, alpha=0.5)
plt.plot([0, 116], [0, 116], 'r--')  # Đường y=x để so sánh
plt.title('True Age vs. Predicted Age')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.grid(True)
plt.savefig('true_vs_predicted_age.png')
plt.show()