import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ----------------------------
# Title & Description
# ----------------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("""
This application predicts whether a breast tumor is **Benign (non-cancerous)** 
or **Malignant (cancerous)** using Machine Learning.
""")

# ----------------------------
# Load Dataset
# ----------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# ----------------------------
# Outlier Removal (IQR Method)
# ----------------------------
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    df_clean = df[~(is_outlier.any(axis=1))]
    return df_clean

df_clean = remove_outliers(df)

# ----------------------------
# Train Model
# ----------------------------
X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Model Accuracy: **{accuracy*100:.2f}%**")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Outlier Handling: IQR Method")

# ----------------------------
# User Input Section
# ----------------------------
st.header("🔍 Enter Tumor Features")

    def user_input_features():
    input_data = []
    for feature in data.feature_names:  # Take ALL features
        value = st.number_input(f"{feature}", min_value=0.0, value=0.0)
        input_data.append(value)
    return np.array(input_data).reshape(1, -1)
input_data = user_input_features()
# ----------------------------
# User Input
st.header("🔍 Enter Tumor Features")

input_data = user_input_features()

# Prediction
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🔴 The tumor is predicted as **Malignant (Cancerous)**")
    else:
        st.success("🟢 The tumor is predicted as **Benign (Non-Cancerous)**")
