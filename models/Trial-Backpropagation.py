import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

'''''''''' READ & PROCESS CSV DATA '''''''''

sensor_data_all = []

for root, dr, files in os.walk(r"./data_drive/data/cleaned_data/Banana"):
    for name in files:
        if name.endswith('.csv'):
            file_path = os.path.join(root, name)
            sensor_data = pd.read_csv(file_path, names=[ "timestamp", "temp", "humd", "MQ2_alcohol", "MQ2_H2", "MQ2_Propane",
                                                        "MQ4_LPG", "MQ4_CH4", "MQ5_LPG", "MQ5_CH4"],skiprows=1)
            print("Loaded " + file_path)  # Check if the file path is correct
            
            day = name.split('_')[1]  # gets the partciular day
            if  name.split('_')[2] == 'fresh':
                fresh = 1
            else:
                fresh = 0
            sensor_data['Fresh'] = fresh
            sensor_data['day'] = day
            sensor_data_all.append(sensor_data)
    
Data_comb = pd.concat(sensor_data_all)
print("All files are loaded.")


'''''''''' VISUALISE DATA '''''''''

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


'''''''''' PROCESS MODEL DATA '''''''''

# Separate the features (X) and the target variable (y)
X = Data_comb.drop(['Fresh', 'day', 'timestamp'], axis=1)  # Drop 'day' and 'timestamp' as well if they are not features
y = Data_comb['Fresh']

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


'''''''''' TRAIN MODEL '''''''''

# Define and train the MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Example: one hidden layer with 100 neurons
    activation='relu',  # Activation function for the hidden layer
    solver='adam',  # The solver for weight optimization
    alpha=0.0001,  # L2 penalty (regularization term) parameter
    batch_size='auto',  # Size of minibatches for stochastic optimizers
    learning_rate='constant',  # Learning rate schedule for weight updates
    learning_rate_init=0.001,  # The initial learning rate
    max_iter=2000,  # Maximum number of iterations
    shuffle=True,  # Whether to shuffle samples in each iteration
    random_state=42,  # Ensures reproducibility
    tol=0.0001,  # Tolerance for the optimization
    verbose=False,  # Whether to print progress messages to stdout
    warm_start=False,  # Reuse the solution of the previous call to fit as initialization
    momentum=0.9,  # Momentum for gradient descent update
    nesterovs_momentum=True,  # Whether to use Nesterov’s momentum
    early_stopping=False,  # Whether to use early stopping to terminate training when validation score is not improving
    validation_fraction=0.1,  # The proportion of training data to set aside as validation set for early stopping
    beta_1=0.9,  # Exponential decay rate for estimates of first moment vector in adam
    beta_2=0.999,  # Exponential decay rate for estimates of second moment vector in adam
    epsilon=1e-08,  # Value for numerical stability in adam
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

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # Specify average method for multi-class/multi-label targets
recall = recall_score(y_test, y_pred, average='binary')  # Specify average method for multi-class/multi-label targets
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
