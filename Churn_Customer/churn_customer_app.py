import streamlit as st
import pandas as pd
import pickle

# Load the pickle model for prediction
with open('LogisticRegression_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pickle model for Location Encoding
with open('OrdinalEncoder_Locations.pkl', 'rb') as file:
    OE = pickle.load(file)

def predict_churn(data):
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
st.title('Churn Prediction App')

# Collecting user inputs
CustomerID = st.text_input('CustomerID')
Name = st.text_input('Name')
Age = st.slider('Age', 18, 100)
Gender = st.selectbox('Gender', ['Male', 'Female'])
Location = st.text_input('Location')
Subscription_Length_Months = st.slider('Subscription Length (Months)', 1, 36)
Monthly_Bill = st.number_input('Monthly Bill ($)')
Total_Usage_GB = st.number_input('Total Usage (GB)')
Churn = st.empty()





# Predict button
if st.button('Predict'):
    Gender = 1 if Gender == "Male" else 0
    Location = OE.transform([[Location]])


    data = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender],
        'Location': [Location],
        'Subscription_Length_Months': [Subscription_Length_Months],
        'Monthly_Bill': [Monthly_Bill],
        'Total_Usage_GB': [Total_Usage_GB]
    })
    prediction = predict_churn(data)
    Churn.write(f'Predicted Churn: {"Customer will stay" if prediction == 0 else "Customer will Exit" }')

