import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc


data_folder_path= "../data/cleaned_data"
fruit = 'Banana'

def load_sensor_data(fruit_name, fruit_id):
    sensor_data_all = []
    path = data_folder_path + f"/{fruit_name}"
    for root, dr, files in os.walk(path):
        for name in files:
            if name.endswith('.csv'):
                file_path = os.path.join(root, name)
                sensor_data = pd.read_csv(file_path, names=[
                    "timestamp", "temp", "humd", "MQ2_alcohol", "MQ2_H2", "MQ2_Propane",
                    "MQ4_LPG", "MQ4_CH4", "MQ5_LPG", "MQ5_CH4"
                ], skiprows=1)
                print("Loaded " + file_path)
                
                day = name.split('_')[1]
                fresh = 1 if name.split('_')[2] == 'fresh' else 0
                
                sensor_data['Fruit'] = fruit_id
                sensor_data['Fresh'] = fresh
                sensor_data['day'] = day
                sensor_data_all.append(sensor_data)
    
    print("All files are loaded for " + f"{fruit_name}")            
    return pd.concat(sensor_data_all)

def plot_sensor_data(df, title):
    df['day'] = df['day'].astype(int) #Hard code
    average_data = df.groupby(['day']).mean()
    plt.figure(figsize=(12, 8))
    for column in average_data.columns:
        if column != 'timestamp':
            plt.plot(average_data.index, average_data[column], label=column)
    plt.xlabel('Day')
    plt.ylabel('Average Sensor Reading/ppm')
    plt.title(title)
    plt.legend()
    plt.show()

'''''''''' READ & PROCESS CSV DATA '''''''''
# 0-Banana 1-Orange 2-Apple 3-Blueberry
Data_comb = load_sensor_data(fruit, 0)


'''''''''' VISUALISE DATA '''''''''
plot_sensor_data(Data_comb, f'Average Sensor Data of {fruit}')




'''''''''' PROCESS MODEL DATA '''''''''

# Separate the features (X) and the target variable (y)
X = Data_comb.drop(['Fresh', 'day', 'timestamp'], axis=1)  # Drop 'day' and 'timestamp' as well if they are not features
y = Data_comb['Fresh']

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


'''''''''' TRAIN MODEL '''''''''

# Define the SVM model
model = SVC(
    C=1.0,  # Regularization strength
    kernel='rbf',  # Kernel type
    gamma='scale',  # Kernel coefficient
    degree=3,  # Polynomial kernel degree
    coef0=0.0,  # Independent term in kernel function
    shrinking=True,  # Use shrinking heuristic
    probability=True,  # Enable probability estimates
    tol=1e-3,  # Tolerance for stopping criterion
    cache_size=200,  # Kernel cache size (in MB)
    class_weight=None,  # Class weights
    verbose=False,  # Verbose output
    max_iter=-1,  # Max iterations (-1 for no limit)
    decision_function_shape='ovr',  # Decision function shape
    break_ties=False,  # Break ties according to confidence
    random_state=42  # Seed for random number generation
)

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()


y_pred = model.predict(X_test)


'''''''''' EVALUATE MODEL '''''''''

print("\nModel Name:", model.__class__.__name__)
model_params = model.get_params()
for param, value in model_params.items():
    print(f"{param}: {value}")
print("\nTraining Time: {:.2f} seconds".format(end_time - start_time))

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)  # Adjust for binary classification
recall = recall_score(y_test, y_pred, zero_division=1)  # Adjust for binary classification
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()


# Calculate ROC curve and AUC for the binary classification case
y_score = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
