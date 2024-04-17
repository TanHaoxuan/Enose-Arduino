#   Machine Learning Based Electronic Nose

## Abstract
This project focuses on addressing the prevalent issue of food waste in Singapore, particularly concerning the ambiguity in determining the freshness of perishable goods post-purchase. Individuals often discard fruits and vegetables based on subjective observations, leading to significant wastage. The primary objective is to develop an Electronic Nose (E-Nose) system capable of accurately detecting and assessing the spoilage status of these perishable items. Leveraging the capabilities of an E-Nose, the project aims to revolutionise the way consumers gauge the freshness of fruits and vegetables, providing a more objective and reliable method of evaluation. By integrating this technology, the project seeks to minimise unnecessary food waste and promote more informed decision-making among consumers.

## Objective
To develop a hardware and software system for an E-Nose to objectively determine the freshness of fruits and vegetables.

## Methodology
Utilizes Arduino and various sensors to collect data on gas emissions from fruits, which are then analyzed using machine learning algorithms to determine spoilage status.\
Algorithms tested:
* Random Forest
* Support Vector Machine (SVM)
* Back Propagation Neural Network (BPNN)

## Result
### Random Forest
0: "fresh", 1: "rotten"
![This is an alt text.](/images/RF_fresh.png )

0: "Apple", 1: "Orange", 2: "Banana", 3: "Blueberry" 
![This is an alt text.](/images/RF_fruit.png )

### Support Vector Machine (SVM)
0: "fresh", 1: "rotten"
![This is an alt text.](/images/SVM_fresh.png )

0: "Apple", 1: "Orange", 2: "Banana", 3: "Blueberry" 
![This is an alt text.](/images/SVM_fruit.png )

### Back Propagation Neural Network (BPNN)
0: "fresh", 1: "rotten"
![This is an alt text.](/images/BPNN_fresh.png )

0: "Apple", 1: "Orange", 2: "Banana", 3: "Blueberry" 
![This is an alt text.](/images/BPNN_fruit.png )

## Comparsion

| Models            | Random Forest | Random Forest (Higher Order = 2) | SVM     | BPNN       |
|-------------------|---------------|----------------------------------|---------|------------|
| Accuracy          | 0.9582        | 0.9581                           | 0.8691  | 0.939398799|
| Precision         | 0.9638        | 0.9638                           | 0.8718  | 0.943749   |
| Recall            | 0.9551        | 0.9546                           | 0.8516  | 0.9377955  |
| Training Time     | 82.04 seconds | 211.50 seconds                   | 4316.29 seconds | 1206.72 seconds |

![This is an alt text.](/images/result_compare_bar.png )


