import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures


min_max = {
    "temp": {"Min": 10.0, "Max": 38.0},
    "humd": {"Min": 50.0, "Max": 98.0},
    "MQ2_alcohol": {"Min": 1.0, "Max": 50.29},
    "MQ2_H2": {"Min": 1.3, "Max": 31.02},
    "MQ2_Propane": {"Min": 0.85, "Max": 20.48},
    "MQ4_LPG": {"Min": 0.18, "Max": 3639.0},
    "MQ4_CH4": {"Min": 0.13, "Max": 191.09},
    "MQ5_LPG": {"Min": 0.01, "Max": 253.84},
    "MQ5_CH4": {"Min": 0.01, "Max": 592.31},
    "LIG": {"Min": 52.0, "Max": 443.0},
    "Fruit": {"Min": 0, "Max": 3},
    "Fresh": {"Min": 0, "Max": 1},
    "day": {"Min": 1, "Max": 9}
}



data = """
23293,27.00,98.00,7.76,6.88,4.49,36.62,15.90,0.92,1.56,152.00
23793,27.00,98.00,7.71,6.79,4.40,36.61,15.90,0.93,1.61,152.00
24293,27.00,98.00,7.66,6.70,4.40,36.60,15.82,0.90,1.62,152.00
24793,27.00,98.00,7.62,6.69,4.9,36.63,15.80,0.95,1.63,152.00
25293,27.00,98.00,7.55,6.68,4.37,35.85,15.43,0.91,1.69,152.00
25793,27.00,98.00,7.52,6.65,4.35,36.40,15.89,0.95,1.72,152.00
"""

# Split the string into lines, then split each line into its values, and convert each to float
raw_data_ls = [[float(value) for value in line.split(',')] for line in data.strip().split('\n')]


raw_data = pd.DataFrame(raw_data_ls, columns =["timestamp", "temp", "humd", "MQ2_alcohol", "MQ2_H2", "MQ2_Propane",
"MQ4_LPG", "MQ4_CH4", "MQ5_LPG", "MQ5_CH4","LIG"], dtype = float).drop(['timestamp', 'LIG'], axis=1)


def normalize_data_points(raw_data, fruit_dict):
    normalized_rows = []  # List to hold the normalized rows
    for index, row in raw_data.iterrows():
        normalized_row = {}
        for feature in raw_data.columns:
            X = row[feature]
            X_min = fruit_dict[feature]["Min"]
            X_max = fruit_dict[feature]["Max"]
            X_normalized = (X - X_min) / (X_max - X_min)
            normalized_row[feature] = X_normalized
        normalized_rows.append(normalized_row)  # Append the normalized row to the list
    return pd.DataFrame(normalized_rows)  # Create a DataFrame from the list of normalized rows

# Example usage with the raw_data DataFrame
normalized_df = normalize_data_points(raw_data, min_max)


model_name = 'RF'
balancing_data = True
order=1
n_splits  =5
model = joblib.load(f'./models/{model_name}-{balancing_data}-order{order}-k{n_splits}-nn.joblib')
poly = joblib.load(f'./models/{model_name}-poly_features-nn.joblib')

# Transform normalized data using polynomial features and predict
pred_df = pd.DataFrame(model.predict(poly.transform(normalized_df)))



# Define mappings for fruit and freshness
fruit_mapping = {0: "Apple", 1: "Orange", 2: "Banana", 3: "Blueberry"}
freshness_mapping = {0: "Not Fresh", 1: "Fresh"} #try to have more status

result = pd.DataFrame()
result['fruit_column'] = pred_df[0].map(fruit_mapping)
result['freshness_column'] = pred_df[1].map(freshness_mapping)
print(result)

most_frequent_predictions = pred_df.mode().iloc[0]
labeled_fruit = fruit_mapping[most_frequent_predictions.iloc[0]]
labeled_freshness = freshness_mapping[most_frequent_predictions.iloc[1]]
print("Most Frequent Fruit:", labeled_fruit)
print("Most Frequent Freshness:", labeled_freshness)






'''

new_data = pd.DataFrame({
    "temp": [0.1],
    "humd": [0.001],
    "MQ2_alcohol": [0.3],
    "MQ2_H2": [0.1],
    "MQ2_Propane": [0.05],
    "MQ4_LPG": [0.02],
    "MQ4_CH4": [0.01],
    "MQ5_LPG": [0.005],
    "MQ5_CH4": [0.007],
})

Apple
temp: Min=28.0, Max=33.0
humd: Min=82.0, Max=95.0
MQ2_alcohol: Min=6.17, Max=8.14
MQ2_H2: Min=5.64, Max=7.06
MQ2_Propane: Min=3.71, Max=4.64
MQ4_LPG: Min=30.32, Max=40.79
MQ4_CH4: Min=13.38, Max=17.46
MQ5_LPG: Min=0.85, Max=1.06
MQ5_CH4: Min=1.47, Max=1.85
LIG: Min=52.0, Max=290.0

Orange
temp: Min=17.0, Max=33.0
humd: Min=47.0, Max=63.0
MQ2_alcohol: Min=6.37, Max=11.3
MQ2_H2: Min=5.87, Max=9.25
MQ2_Propane: Min=3.86, Max=6.09
MQ4_LPG: Min=29.7, Max=58.18
MQ4_CH4: Min=13.14, Max=23.98
MQ5_LPG: Min=0.69, Max=1.15
MQ5_CH4: Min=1.17, Max=2.02
LIG: Min=82.0, Max=764.0

Banana
temp: Min=31.0, Max=35.0
humd: Min=85.0, Max=90.0
MQ2_alcohol: Min=4.89, Max=8.06
MQ2_H2: Min=4.7, Max=7.04
MQ2_Propane: Min=3.09, Max=4.63
MQ4_LPG: Min=20.27, Max=37.85
MQ4_CH4: Min=9.33, Max=16.32
MQ5_LPG: Min=0.9, Max=26.97
MQ5_CH4: Min=1.56, Max=55.88
LIG: Min=68.0, Max=255.0

Blueberry
temp: Min=23.0, Max=28.0
humd: Min=56.0, Max=98.0
MQ2_alcohol: Min=1.67, Max=7.85
MQ2_H2: Min=1.95, Max=6.99
MQ2_Propane: Min=1.28, Max=4.9
MQ4_LPG: Min=7.25, Max=37.55
MQ4_CH4: Min=3.72, Max=16.5
MQ5_LPG: Min=0.88, Max=5.89
MQ5_CH4: Min=1.5, Max=11.26
LIG: Min=82.0, Max=189.0
'''