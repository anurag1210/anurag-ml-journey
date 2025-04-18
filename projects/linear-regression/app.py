import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and train the model
@st.cache_resource
def load_model():
    df = pd.read_csv("housing.csv")
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    X = df[['housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'median_income']]
    y = df['median_house_value']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = load_model()

# UI
st.title("🏡 California House Price Predictor")
st.markdown("Enter housing details below to predict the median house value.")

# Input fields
housing_median_age = st.number_input("Housing Median Age", min_value=1.0, max_value=100.0, value=20.0)
total_rooms = st.number_input("Total Rooms", min_value=1.0, max_value=50000.0, value=2000.0)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, max_value=10000.0, value=400.0)
population = st.number_input("Population", min_value=1.0, max_value=50000.0, value=1000.0)
households = st.number_input("Households", min_value=1.0, max_value=10000.0, value=400.0)
median_income = st.number_input("Median Income", min_value=0.0, max_value=15.0, value=3.0)

if st.button("Predict"):
    input_data = np.array([[housing_median_age, total_rooms, total_bedrooms,
                            population, households, median_income]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Predicted Median House Value: ${int(prediction):,}")
