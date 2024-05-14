# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 07:54:14 2022

@author: Avish
"""

import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from constants import DASHBOARD_URL


diabetes_model = pickle.load(open('dog_diabetes_model.sav','rb'))
heart_model = pickle.load(open('dog_heart_model.sav','rb'))

diabetes_dataset = pd.read_csv('dogDiabetesData.csv')
heart_dataset = pd.read_csv('dogHeartData.csv')


# st.markdown(f'<a href="javascript:window.history.back();">Back to Landing Page</a>', unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu('Dog Disease Prediciton System',
                           ['Dog Diabetes Prediction',
                            'Dog Heart Disease Prediction'],
                           icons = ['activity','heart'],
                           default_index = 0)
    
    
breeds = ['Yorkshire Terrier', 'Siberian Husky', 'Labrador Retriever',
          'Chihuahua', 'Great Dane', 'Bulldog', 'Golden Retriever',
          'Dachshund', 'Beagle', 'Poodle', 'Pomeranian', 'Shih Tzu',
          'German Shepherd', 'Boxer', 'Rottweiler']

breed_dict = {breed: index for index, breed in enumerate(sorted(breeds))}

sorted_breed_dict = dict(sorted(breed_dict.items()))

#for diabetes prediction
if selected == 'Dog Diabetes Prediction':
    
    
    st.title('Dog Diabetes Prediciton')
        # getting the input data from the user

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Glucose = st.text_input('Glucose Level')
        
    with col2:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col3:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col1:
        Insulin = st.text_input('Insulin Level')
    
    with col2:
        Polyuria = st.text_input('Polyuria value')
    
    with col3:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col1:
        Age = st.text_input('Age of the Dog')

    
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        
        std_data = [[int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(Polyuria), float(DiabetesPedigreeFunction), int(Age)]]

        diab_prediction = diabetes_model.predict(std_data)
        if diab_prediction[0] == 0:
            diab_diagnosis = 'Your Dog does not have Diabetes'
        else:
            diab_diagnosis = 'Your Dog might have Diabetes'
            
    st.success(diab_diagnosis)
        
    
if selected == 'Dog Heart Disease Prediction':
    
    st.title('Dog Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)  
    
    with col1:
        Stroke = st.text_input('Stroke (1 for Yes, 0 for No)')
        
    with col2:
        DiffWalking = st.text_input('Difficulty in Walking (1 for Yes, 0 for No)')
    
    with col3:
        Sex = st.text_input('Gender (1 for Male, 0 for Female)')
    
    with col1:
        Diabetic = st.text_input('Diabetic (1 for Yes, 0 for No)')
    
    with col2:
        PhysicalActivity = st.text_input('Physical Activity (1 for Yes, 0 for No)')
    
    with col3:
        GenHealth = st.text_input('General Health (Range 1(poor) to 5(excellent))')
    
    with col1:
        SleepTime = st.text_input('SleepTime (In hours 0-23)')
    
    with col2:
        KidneyDisease = st.text_input('Kidney Disease(1 for Yes, 0 for No)')
    
    with col3:
        Breed = st.selectbox("Select an option:", breeds)

    with col1:
        Age = st.text_input('Age (max. 14)')
        
    
    
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Heart Disease Test Result"):
        std_data = [[int(Stroke), int(DiffWalking), int(Sex), int(Diabetic), int(PhysicalActivity),
                     int(GenHealth), int(SleepTime), int(KidneyDisease), int(Age), 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        std_data[0][9+breed_dict[Breed]] = 1
        
        heart_prediction = heart_model.predict(std_data)                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = "Your Dog might have Heart disease"
        else:
          heart_diagnosis = "Your Dog does not have Heart disease"
        
    st.success(heart_diagnosis)
