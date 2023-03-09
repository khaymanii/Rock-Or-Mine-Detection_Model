# Importing the Libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Processing

sonar_data = pd.read_csv('sonar_data.csv', header=None)
sonar_data.head()
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()


# Seperating data and labels

X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
print(X)
print(y)


# Training and Test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)


# Model Training using Logistics Regression

model = LogisticRegression()


model.fit(X_train, y_train)


# Model Evaluation for accuracy in training and test data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)


print('Accuracy on training data : ', training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy on test data : ', test_data_accuracy)


# Making a predictive system

input_data = (0.0335, 0.0134, 0.0696, 0.1180, 0.0348, 0.1180, 0.1948, 0.1607, 0.3036, 0.4372, 0.5533, 0.5771, 0.7022, 0.7067, 0.7367, 0.7391, 0.8622, 0.9458, 0.8782, 0.7913, 0.5760, 0.3061, 0.0563, 0.0239, 0.2554, 0.4862, 0.5027, 0.4402, 0.2847,
              0.1797, 0.3560, 0.3522, 0.3321, 0.3112, 0.3638, 0.0754, 0.1834, 0.1820, 0.1815, 0.1593, 0.0576, 0.0954, 0.1086, 0.0812, 0.0784, 0.0487, 0.0439, 0.0586, 0.0370, 0.0185, 0.0302, 0.0244, 0.0232, 0.0093, 0.0159, 0.0193, 0.0032, 0.0377, 0.0126, 0.0156)

# changing the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


predictions = model.predict(input_data_reshaped)
print(predictions)

if (predictions[0] == 'R'):
    print('The Object is a Rock')
else:
    print('The object is a Mine')
