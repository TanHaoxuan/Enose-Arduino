import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# READ & PROCESS CSV DATA
sensor_data_all = []
for root, dirs, files in os.walk("./data_drive/data/cleaned_data/Banana"):
    for name in files:
        if name.endswith('.csv'):
            file_path = os.path.join(root, name)
            sensor_data = pd.read_csv(file_path, names=["timestamp", "temp", "humd", "MQ2_alcohol", "MQ2_H2", "MQ2_Propane",
                                                        "MQ4_LPG", "MQ4_CH4", "MQ5_LPG", "MQ5_CH4"], skiprows=1)
            print("Loaded " + file_path)  # Check if the file path is correct

            day = name.split('_')[1]  # Extracts the particular day
            fresh = 1 if name.split('_')[2] == 'fresh' else 0
            sensor_data['Fresh'] = fresh
            sensor_data['day'] = day
            sensor_data_all.append(sensor_data)

#Data_comb = pd.concat(sensor_data_all)
# Concatenate sensor_data_all into a single DataFrame
Data_comb = pd.concat(sensor_data_all, ignore_index=True)
Data_comb_raw = Data_comb

# Reset the index to make sure it's unique and starts from 1 instead of 0
Data_comb.index = np.arange(1, len(Data_comb) + 1)

print("All files are loaded.")

# VISUALIZE DATA
# Group the data by day and sensor, and calculate the average sensor readings
average_data = Data_comb.groupby(['day']).mean()

# Plot the average sensor readings for each day
plt.figure(figsize=(12, 8))
for column in average_data.columns:
    if column != 'timestamp':
        plt.plot(average_data.index, average_data[column], label=column)
plt.xlabel('Day')
plt.ylabel('Average Sensor Reading')
plt.title('Average Sensor Data:')
plt.legend()
plt.show()


# Print summary statistics
print("\nRaw data summary")
print(Data_comb_raw.describe())

# PROCESS MODEL DATA
X = Data_comb.drop(['Fresh', 'day', 'timestamp'], axis=1)
y = Data_comb['Fresh']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# DEFINE TENSORFLOW MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# TRAIN MODEL
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
end_time = time.time()

#model.save('tf-rf-model',save_format='tf')

# EVALUATE MODEL
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

y_pred = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Calculate ROC curve and AUC
y_score = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Training Details
print("\nModel Name:", model.name)
print("Training Time: {:.2f} seconds".format(end_time - start_time))


# Path to the SavedModel directory
saved_model_dir = './saved_model_rf/my_model'

# Create a TFLiteConverter object from the SavedModel directory
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# (Optional) Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the converted model to a file
tflite_model_path = './saved_model/my_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted to TFLite and saved to {tflite_model_path}")

