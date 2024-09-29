import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load model and index
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('index.pkl', 'rb') as file:
    index = pickle.load(file)

# Load existing dataset
X = pd.read_csv('X.csv').drop(columns=['Unnamed: 0'])

# Streamlit app title
st.title("Prediction App")

# Create input fields for user inputs
x0 = st.selectbox("Please select 'srcDistrictName'", ['ri bhoi', 'khasi hills', 'garo hills', 'jaintia hills'])
x1 = st.number_input('Number of Households issue Job Card Schedule Castes (SC)', min_value=0, max_value=210)
x2 = st.number_input('Number of Households issue Job Card Others', min_value=0, max_value=1303)
x3 = st.number_input('Job Issues in SC', min_value=0, max_value=1626)
x4 = st.number_input('Number of person days generated Scheduled Caste (SCs)', min_value=0, max_value=15119)
x5 = st.number_input('Total number of person days generated', min_value=0, max_value=124099)
x6 = st.number_input('Families completed 100 days Scheduled Caste (SCs)', min_value=0, max_value=145)
x7 = st.number_input('Families completed 100 days Others', min_value=0, max_value=645)
x8 = st.number_input('Total Families completed 100 days', min_value=0, max_value=852)
x9 = st.number_input('YearCode', min_value=2014, max_value=2021)

# When the user clicks the button, we process the input
if st.button('Predict'):
    # Create a new DataFrame for the input data
    X_new = pd.DataFrame([[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]], columns=[
        'srcDistrictName',
        'Number of Households issue Job Card Schedule Castes (SC)',
        'Number of Households issue Job Card Others',
        'Job Issues in SC',
        'Number of person days generated Scheduled Caste (SCs)',
        'Total number of person days generated',
        'Families completed 100days Scheduled Caste (SCs)',
        'Families completed 100days Others',
        'Total Families completed 100 days',
        'YearCode'
    ])

    # Concatenate the input data with the existing dataset
    X = pd.concat([X, X_new], axis=0, ignore_index=True).drop(columns='Job Issues in SC')

    # Standardize the numeric columns
    scaler = StandardScaler()
    numeric_list = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_list]), columns=numeric_list)

    # Encode the categorical 'srcDistrictName' column
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X[['srcDistrictName']])
    X_encoded = pd.DataFrame(X_encoded, columns=['0', '1', '2', '3'])

    # Combine the scaled numeric data and encoded categorical data
    X_final = pd.concat([X_scaled, X_encoded], axis=1)

    # Get the user input row (the last row)
    user_input = X_final.iloc[-1].values.reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(user_input)[0]

    # Display the prediction
    st.success(f'The Total number of Household provided Employment predicted value is: {prediction}')
