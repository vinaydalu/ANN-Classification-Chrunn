import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoder
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#load label encoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

#load one hot encoding
with open('onehot_encoder_geography.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

## streamlit app
st.title('Customer Churn Predictions')

#User Input

geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame(
    {
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure':[tenure],
        'Balance':[balance], 
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary]
    }
)

## One-hot encode "Geography"
geo_encoded = one_hot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale Input Data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
predictions_proba = prediction[0][0]

st.write("Churn Probability:", round(predictions_proba, 2))

if predictions_proba > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')



