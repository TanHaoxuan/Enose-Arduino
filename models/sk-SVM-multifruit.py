import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

data_folder_path= "./data_drive/data/cleaned_data"


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
Data_comb_banana = load_sensor_data('Banana', 0)
Data_comb_orange = load_sensor_data('Orange', 1)
#Data_comb_apple = load_sensor_data('Apple', 2)
#Data_comb_blueberry = load_sensor_data('Blueberry', 3)

'''''''''' VISUALISE DATA '''''''''
plot_sensor_data(Data_comb_banana, 'Average Sensor Data of Banana')
plot_sensor_data(Data_comb_orange, 'Average Sensor Data of Orange')
#plot_sensor_data(Data_comb_apple, 'Average Sensor Data of Apple')
#plot_sensor_data(Data_comb_blueberry, 'Average Sensor Data of Blueberry')





'''''''''' Combine DF of Different Fruit '''''''''


Data_comb = pd.concat([Data_comb_banana, Data_comb_orange], axis=0)



'''''''''' PROCESS MODEL DATA '''''''''

# Separate the features (X) and the target variable (y)
X = Data_comb.drop(['Fresh', 'Fruit','day', 'timestamp'], axis=1)  # Drop 'day' and 'timestamp' as well if they are not features
y = Data_comb[['Fruit', 'Fresh']]
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
multi_model = MultiOutputClassifier(model)

start_time = time.time()

multi_model.fit(X_train, y_train)

end_time = time.time()


y_pred = multi_model.predict(X_test)


'''''''''' EVALUATE MODEL '''''''''

print("\nModel Name:", model.__class__.__name__)
model_params = model.get_params()
for param, value in model_params.items():
    print(f"{param}: {value}")
print("\nTraining Time: {:.2f} seconds".format(end_time - start_time))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Specify average method for multi-class/multi-label targets
recall = recall_score(y_test, y_pred, average='macro')  # Specify average method for multi-class/multi-label targets
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Calculate confusion matrix
for i, target_name in enumerate(y.columns):
    cm = confusion_matrix(y_test[target_name], y_pred[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {target_name}')
    plt.show()


# Calculate ROC curve and AUC for the binary classification case
# The predict_proba will give us a list of [probabilities_for_fruit, probabilities_for_fresh]

# Get the probabilities for all classes for the 'Fresh' output
probs_fresh = model.predict_proba(X_test)[1]  # This will give us the probabilities for the 'Fresh' output

# Now you can get the probabilities for the positive class of 'Fresh'
y_pred_prob_fresh = probs_fresh[:, 1]  # This is assuming that '1' signifies the positive class for 'Fresh'

# Now you can use y_pred_prob_fresh to compute ROC curve and AUC as before
fpr, tpr, _ = roc_curve(y_test['Fresh'], y_pred_prob_fresh)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Fresh')
plt.legend(loc="lower right")
plt.show()
