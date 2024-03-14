import pandas as pd

apple = {
    "temp": {"Min": 28.0, "Max": 33.0},
    "humd": {"Min": 82.0, "Max": 95.0},
    "MQ2_alcohol": {"Min": 6.17, "Max": 8.14},
    "MQ2_H2": {"Min": 5.64, "Max": 7.06},
    "MQ2_Propane": {"Min": 3.71, "Max": 4.64},
    "MQ4_LPG": {"Min": 30.32, "Max": 40.79},
    "MQ4_CH4": {"Min": 13.38, "Max": 17.46},
    "MQ5_LPG": {"Min": 0.85, "Max": 1.06},
    "MQ5_CH4": {"Min": 1.47, "Max": 1.85},
    "LIG": {"Min": 52.0, "Max": 290.0}
}

orange = {
    "temp": {"Min": 17.0, "Max": 33.0},
    "humd": {"Min": 47.0, "Max": 63.0},
    "MQ2_alcohol": {"Min": 6.37, "Max": 11.3},
    "MQ2_H2": {"Min": 5.87, "Max": 9.25},
    "MQ2_Propane": {"Min": 3.86, "Max": 6.09},
    "MQ4_LPG": {"Min": 29.7, "Max": 58.18},
    "MQ4_CH4": {"Min": 13.14, "Max": 23.98},
    "MQ5_LPG": {"Min": 0.69, "Max": 1.15},
    "MQ5_CH4": {"Min": 1.17, "Max": 2.02},
    "LIG": {"Min": 82.0, "Max": 764.0}
}

banana = {
    "temp": {"Min": 31.0, "Max": 35.0},
    "humd": {"Min": 85.0, "Max": 90.0},
    "MQ2_alcohol": {"Min": 4.89, "Max": 8.06},
    "MQ2_H2": {"Min": 4.7, "Max": 7.04},
    "MQ2_Propane": {"Min": 3.09, "Max": 4.63},
    "MQ4_LPG": {"Min": 20.27, "Max": 37.85},
    "MQ4_CH4": {"Min": 9.33, "Max": 16.32},
    "MQ5_LPG": {"Min": 0.9, "Max": 26.97},
    "MQ5_CH4": {"Min": 1.56, "Max": 55.88},
    "LIG": {"Min": 68.0, "Max": 255.0}
}

blueberry = {
    "temp": {"Min": 23.0, "Max": 28.0},
    "humd": {"Min": 56.0, "Max": 98.0},
    "MQ2_alcohol": {"Min": 1.67, "Max": 7.85},
    "MQ2_H2": {"Min": 1.95, "Max": 6.99},
    "MQ2_Propane": {"Min": 1.28, "Max": 4.9},
    "MQ4_LPG": {"Min": 7.25, "Max": 37.55},
    "MQ4_CH4": {"Min": 3.72, "Max": 16.5},
    "MQ5_LPG": {"Min": 0.88, "Max": 5.89},
    "MQ5_CH4": {"Min": 1.5, "Max": 11.26},
    "LIG": {"Min": 82.0, "Max": 189.0}
}


raw_data_ls=[
[21293,29.00,79.00,8.23,7.19,4.73,38.69,16.65,0.86,1.49,270.00],
[21793,29.00,80.00,8.23,7.19,4.77,38.69,16.65,0.87,1.50,270.00],
[22293,29.00,80.00,8.32,7.25,4.77,38.69,16.65,0.87,1.50,270.00],
[22793,29.00,80.00,8.32,7.25,4.77,38.69,16.65,0.88,1.52,270.00],
]

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
apple_normalized = normalize_data_points(raw_data, apple)
orange_normalized = normalize_data_points(raw_data, orange)
banana_normalized = normalize_data_points(raw_data, banana)
blueberry_normalized = normalize_data_points(raw_data, blueberry)



# Transform normalized data using polynomial features and predict
banana_pred_df = pd.DataFrame(model.predict(poly.transform(banana_normalized)))
orange_pred_df = pd.DataFrame( model.predict(poly.transform(orange_normalized)))
apple_pred_df = pd.DataFrame(model.predict(poly.transform(apple_normalized)))
blueberry_pred_df = pd.DataFrame(model.predict(poly.transform(blueberry_normalized)))


combined_pred_df = pd.concat([apple_pred_df, orange_pred_df, banana_pred_df, blueberry_pred_df], ignore_index=True)

# Define mappings for fruit and freshness
fruit_mapping = {0: "Apple", 1: "Orange", 2: "Banana", 3: "Blueberry"}
freshness_mapping = {0: "Not Fresh", 1: "Fresh"}

most_frequent_predictions = combined_pred_df.mode().iloc[0]
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