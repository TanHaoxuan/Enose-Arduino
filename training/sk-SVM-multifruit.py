import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


data_folder_path= "../data/cleaned_data"


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
                #print("Loaded " + file_path)
                
                day = name.split('_')[1]
                fresh = 1 if name.split('_')[2] == 'fresh' else 0
                
                sensor_data['Fruit'] = fruit_id
                sensor_data['Fresh'] = fresh
                sensor_data['day'] = day
                sensor_data_all.append(sensor_data)
    
    output_df = pd.concat(sensor_data_all)
    print("All files are loaded for " + f"{fruit_name}. No of samples:{output_df.shape[0]}")    
    return output_df

def plot_sensor_data(df, title):
    df['day'] = df['day'].astype(int) #Hard code
    average_data = df.groupby(['day']).mean()
    plt.figure(figsize=(12, 8))
    for column in average_data.columns:
        if column not in ['timestamp', 'Fruit']:
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
Data_comb_apple = load_sensor_data('Apple', 2)
Data_comb_blueberry = load_sensor_data('Blueberry', 3)

'''''''''' VISUALISE DATA '''''''''
VISUALISE_DATA = False
if VISUALISE_DATA:
    plot_sensor_data(Data_comb_banana, 'Average Sensor Data of Banana')
    plot_sensor_data(Data_comb_orange, 'Average Sensor Data of Orange')
    plot_sensor_data(Data_comb_apple, 'Average Sensor Data of Apple')
    plot_sensor_data(Data_comb_blueberry, 'Average Sensor Data of Blueberry')





'''''''''' Combine DF of Different Fruit '''''''''


Data_comb = pd.concat([Data_comb_banana, Data_comb_orange, Data_comb_apple, Data_comb_blueberry], axis=0)


'''''''''' Balancing Data '''''''''
balancing_data=True

if balancing_data:

    # Calculate the count of each combination
    counts = Data_comb.groupby(['Fruit', 'Fresh']).size().reset_index(name='counts')
    min_count = counts['counts'].min()
    
    def resample_group(group):
        return group.sample(min_count, replace=True, random_state=42)
    
    # Resample each group to have the same number of samples
    balanced_data = Data_comb.groupby(['Fruit', 'Fresh']).apply(resample_group).reset_index(drop=True)

else:
    balanced_data = Data_comb

'''''''''' PROCESS MODEL DATA '''''''''
# Separate the features (X) and the target variable (y)
X = balanced_data.drop(['Fresh', 'Fruit','day', 'timestamp'], axis=1)  # Drop 'day' and 'timestamp' as well if they are not features
y = balanced_data[['Fruit', 'Fresh']]


'''''''''' Polynomial Features '''''''''
order = 1
poly = PolynomialFeatures(degree=order, include_bias=False)
X = poly.fit_transform(X)


'''''''''' K Fold '''''''''
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# To store the performance of each fold
fold_accuracy_scores = []
fold_precision_scores = []
fold_recall_scores = []
training_time=0
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):

    # Splitting dataset into training and testing set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # No change if y is a DataFrame, just illustrating the point
   
    
    '''''''''' TRAIN MODEL '''''''''

    # Define the SVM model
    base_model = SVC(
        C=1.0,  # Regularization strength
        kernel='rbf',  # Kernel type
        gamma='scale',  # Kernel coefficient
        degree=3,  # Polynomial kernel degree
        coef0=0.0,  # Independent term in kernel function
        shrinking=True,  # Use shrinking heuristic
        probability=False,  # Enable probability estimates
        tol=1e-3,  # Tolerance for stopping criterion
        cache_size=200,  # Kernel cache size (in MB)
        class_weight=None,  # Class weights
        verbose=False,  # Verbose output
        max_iter=-1,  # Max iterations (-1 for no limit)
        decision_function_shape='ovr',  # Decision function shape
        break_ties=False,  # Break ties according to confidence
        random_state=42  # Seed for random number generation
    )
    model = MultiOutputClassifier(base_model)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time += (time.time()-start_time)
    

    y_pred = model.predict(X_test)
    accuracy =np.mean([accuracy_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    precision = np.mean([
        precision_score(y_test.iloc[:, i], y_pred[:, i], average='macro', zero_division=0) 
        for i in range(y_test.shape[1])
    ])
    recall = np.mean([
        recall_score(y_test.iloc[:, i], y_pred[:, i], average='macro', zero_division=0) 
        for i in range(y_test.shape[1])
    ])
    
    fold_accuracy_scores.append(accuracy)
    fold_precision_scores.append(precision)
    fold_recall_scores.append(recall)

    print(f"Fold #{fold} - Accuracy: {accuracy} Precision: {precision} Recall: {recall}")

#save the model
save = False
if save:
    joblib.dump(model, f'./models/SVM-{balancing_data}-order{order}-k{n_splits}.joblib')
    joblib.dump(poly, './models/SVM-poly_features.joblib')

'''''''''' EVALUATE MODEL '''''''''

print("\nModel Name:", model.__class__.__name__)
model_params = model.get_params()
for param, value in model_params.items():
    print(f"{param}: {value}")
print("\nTraining Time: {:.2f} seconds".format(training_time))
print(f"Balance of data: {balancing_data}")
print(f"Polynomial Features order: {order}")
print(f"K Fold: {n_splits}")


#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred, average='macro')  # Specify average method for multi-class/multi-label targets
#recall = recall_score(y_test, y_pred, average='macro')  # Specify average method for multi-class/multi-label targets
print("Accuracy:", np.mean(fold_accuracy_scores))
print("Precision:", np.mean(fold_precision_scores))
print("Recall:", np.mean(fold_recall_scores))


# Calculate confusion matrix
for i, target_name in enumerate(y.columns):
    cm = confusion_matrix(y_test[target_name], y_pred[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'SVM Confusion Matrix for {target_name}')
    plt.show()


# Calculate ROC curve and AUC for the binary classification case
# The predict_proba will give us a list of [probabilities_for_fruit, probabilities_for_fresh]


# Assuming 'Fresh' is the second target and you're interested in its probabilities
probs_fresh_list = model.predict_proba(X_test)  # This returns a list of arrays
if len(probs_fresh_list) > 1:  # Ensure there's more than one target
    probs_fresh = probs_fresh_list[1]  # Get the array for 'Fresh', assuming it's the second target
    y_pred_prob_fresh = probs_fresh[:, 1]  # Probabilities of the positive class for 'Fresh'

    # Compute ROC curve and AUC for 'Fresh'
    fpr, tpr, _ = roc_curve(y_test['Fresh'], y_pred_prob_fresh)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM Receiver Operating Characteristic for Fresh')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("Unexpected structure of probabilities returned.")

# Assuming 'Fruit' is the first target
probs_fruit_list = model.predict_proba(X_test)
probs_fruit = probs_fruit_list[0]  # Probabilities for 'Fruit'

# Binarize the 'Fruit' outcomes for multi-class ROC/AUC calculation
y_test_fruit_binarized = label_binarize(y_test['Fruit'], classes=np.unique(y_test['Fruit']))

# Calculate AUC for each class
for i, class_name in enumerate(np.unique(y_test['Fruit'])):
    if len(y_test_fruit_binarized[0]) > 1:  # Ensure it's multi-class
        fpr, tpr, _ = roc_curve(y_test_fruit_binarized[:, i], probs_fruit[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for class {class_name} (area = {roc_auc:.2f})')
    else:
        print(f"Skipping AUC for 'Fruit' class {class_name} due to insufficient data.")

# Plotting details
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic for Fruit')
plt.legend(loc="lower right")
plt.show()