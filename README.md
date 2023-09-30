# diabetes prediction model

## Project Description
This project aims to provide a predictive model for diabetes diagnosis using neural network technology. Through meticulous data processing and feature selection, we study the performance of single-layer perceptrons versus multi-layer neural networks and provide directions for future optimization work.

## File Description
* diabetes.csv: training data. `Outcome` is the target variable.
* script.ipynb: jupyter notebook code
* script.py: python script

 ## Dataset description
* Data integrity: This data set is very complete, and no obvious outliers or missing values ​​were found.
* Sample label: Positive samples (with diabetes) account for 34.9%, and negative samples (without diabetes) account for the remaining proportion.
* Feature description: The original data contains 8 features. Except for “BloodPressure,” all other features were significantly associated with diabetes diagnosis.

## Model performance
* The classification accuracy of the single-layer perceptron on the test data is 0.708.
* The classification accuracy of the SimpleMLP(Single-hidden-layer MLP) on the test data is 0.727.
* The classification accuracy of the multi-layer neural network(Double-hidden-layer MLP) is 0.766.

Although the performance of the two MLPs is similar, there is still room for improvement. The project also provides visual charts of confusion matrices and ROC curves for user reference.

## future work

* Feature engineering: Further explore the interactions between features and try to add more diabetes-related features.
*  Model optimization: Try more complex network structures and parameter combinations to improve prediction accuracy.
