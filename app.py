import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scipy import stats

st.title('Breast Cancer Prediction with Outlier Handling')

# Load Dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Outlier Detection using IQR Method
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    df_clean = df[~(is_outlier.any(axis=1))]
    return df_clean

# Clean data
df_clean = remove_outliers(df)

X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

st.write(f"Model Accuracy (after outlier removal): {model.score(X_test, y_test):.2f}")

# User Input
st.header('Enter Tumor Features to Predict')

def user_input_features():
    inputs = []
    for feature in data.feature_names:
        value = st.number_input(f'Input {feature}', min_value=0.0, value=0.0)
        inputs.append(value)
    return np.array(inputs).reshape(1, -1)

input_data = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error('Prediction: Malignant Tumor')
    else:
        st.success('Prediction: Benign Tumor')
