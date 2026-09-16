import streamlit as st
import pandas as pd
import joblib



model = joblib.load('/Users/jaydipchauhan/Documents/DS/Projects/Titanic/titanic_model.pkl')

st.set_page_config(page_title='Titanic Survival Predictor',
                   page_icon='🚢')

st.title('🚢 Titanic Survival Predictor')

st.write(
    'Enter Passenger information to predict the probability of survival.'
)

Pclass = st.selectbox(
    'Passenger Class',
    [1,2,3]
)

Sex = st.selectbox(
    'Sex',
    ['Male', 'Female']
)

Age = st.number_input(
    'Age',
    min_value = 0,
    max_value = 100,
    value = 0
)

SibSp = st.number_input(
    'Number of Siblings/Spouses',
    min_value=0,
    max_value=10,
    value=0
)

Parch = st.number_input(
    'Number of Paretns/Children',
    min_value=0,
    max_value=10,
    value=0
)

Fare = st.number_input(
    'Fare',
    min_value=0.0,
    value=30.0
)

Embarked = st.selectbox(
    'Embarked',
    ['S','C','Q']
)

CabinKnown = st.selectbox(
    'Cabimn Information Available?',
    [0, 1],
    format_func=lambda x: 'Yes' if x==1 else 'No'
)

FamilySize = SibSp + Parch + 1

IsAlone = 1 if FamilySize == 1 else 0

input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked':[Embarked],
    'CabinKnown':[CabinKnown],
    'FamilySize': [FamilySize],
    'IsAlone': [IsAlone]
})

if st.button('Predict Survival'):


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success('🟢 Predicted: Passenger survived')

    else:
        st.error('🔴 Predicted: Passenger did not survive')

    st.write(
        f'Estimated survival probability: **{probability:.2%}**'
    )