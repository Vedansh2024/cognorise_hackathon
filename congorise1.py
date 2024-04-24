import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('credit_records.csv')

# Drop the sno column as it is not relevant for predicting creditworthiness
df = df.drop('sno', axis=1)

# Handle missing values by filling them with the mean of the column
df.fillna(df.mean(), inplace=True)

# Encode categorical variables as integers
le = LabelEncoder()
df['credit_history'] = le.fit_transform(df['credit_history'])
df['purpose'] = le.fit_transform(df['purpose'])
df['employment_st'] = le.fit_transform(df['employment_st'])
df['personal_status'] = le.fit_transform(df['personal_status'])
df['guarantors'] = le.fit_transform(df['guarantors'])
df['property_type'] = le.fit_transform(df['property_type'])
df['installment_type'] = le.fit_transform(df['installment_type'])
df['housing_type'] = le.fit_transform(df['housing_type'])
df['job_type'] = le.fit_transform(df['job_type'])

# Split the dataset into features (X) and target (y)
X = df.drop('Group_no', axis=1)
y = df['Group_no']

# Scale the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the performance of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))