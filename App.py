import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("E-Commerce Sales Prediction")

# Load data
df = pd.read_excel("ecommerce_dataset.xlsx")

# Convert date
df['Order_Date'] = pd.to_datetime(df['Order_Date'], origin='1899-12-30', unit='D')
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Day'] = df['Order_Date'].dt.day

# Drop columns
df = df.drop(columns=['Order_Date', 'Order_ID'])

# Save original
original_df = df.copy()

# Convert categorical
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop(columns=['Total_Sales'])
y = df['Total_Sales']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

st.write("Model trained successfully")

# User input
input_data = {}

for col in original_df.columns:
    if col == "Total_Sales":
        continue

    if original_df[col].dtype == 'object':
        input_data[col] = st.selectbox(col, original_df[col].unique())
    else:
        input_data[col] = st.number_input(col, value=0)

input_df = pd.DataFrame([input_data])

# Encode input
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")