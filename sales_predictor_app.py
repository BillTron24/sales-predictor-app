import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# ---- TRAINING MOCK MODEL ----
# Dummy training data (replace this with your own model if needed)
data = pd.read_csv("Sample - Superstore.csv", encoding="ISO-8859-1")
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Month_num'] = data['Order Date'].dt.to_period('M').astype(str)
data['Month_num'] = pd.to_datetime(data['Month_num']).rank(method='dense').astype(int)

monthly = data.groupby('Month_num').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Discount': 'mean',
    'Profit': 'sum'
}).reset_index()

X = monthly[['Month_num', 'Quantity', 'Discount', 'Profit']]
y = monthly['Sales']

model = LinearRegression()
model.fit(X, y)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Sales Predictor", layout="centered")

st.title("📈 Sales Prediction App")
st.markdown("Enter values to predict future sales:")

# User Inputs
month = st.number_input("Month Number (e.g., next month = 50+)", min_value=1, step=1)
quantity = st.number_input("Expected Quantity", value=50)
discount = st.slider("Average Discount", 0.0, 0.9, 0.1)
profit = st.number_input("Estimated Profit", value=1000.0)

# Predict
if st.button("Predict Sales"):
    input_data = np.array([[month, quantity, discount, profit]])
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Sales: ${prediction:.2f}")
