import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("please enter the values and hit the predict button for getting prediction")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30    )


tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthly_charges = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

gender = st.selectbox("Gender",["Male","Female"])

st.divider()

predictbutton = st.button("Submit")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthly_charges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"predicted:  {predicted}")

else:
    st.write("Please enter the values and use predict button")
