# Hotel Booking Cancellation Prediction

## About the Project

This project focuses on predicting whether a hotel booking will be cancelled or not using **Logistic Regression**, a machine learning algorithm.
The main goal is to understand how different booking details, such as lead time, number of guests, hotel type, and deposit type, can help predict cancellations.

## Dataset 

The project uses the Hotel Booking Demand dataset, which contains information about bookings made at City Hotels and Resort Hotels.

The dataset includes details such as:
- Hotel type
- Lead time
- Number of guests
- Length of stay
- Deposit type
- Previous cancellations
- Booking cancellation status

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib 
- Scikit-learn

## Project Workflow

The project starts with data cleaning, where missing values and duplicate records are handled. Categorical features are converted into numerical values using One-Hot Encoding, and numerical features are scaled using StandardScaler.
The dataset is then divided into training and testing sets. A Logistic Regression model is trained to predict whether a booking will be cancelled.

The model is evaluated using:
- Accuracy
- Precision
- Confusion Matrix
- ROC Curve

The project also displays predictions for real bookings from the test dataset and compares them with their actual cancellation status.

## Conclusion

This project helped me understand the complete machine learning workflow, from preparing a dataset to training and evaluating a classification model. It also shows how Logistic Regression can be used to identify hotel bookings that are more likely to be cancelled.
