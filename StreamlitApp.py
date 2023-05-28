import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Streamlit App
def main():
    st.title("Diabetes Prediction")
    st.write("Enter the following parameters to predict diabetes:")

    # Create input fields for user input
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin Level", 0, 846, 79)
    bmi = st.slider("BMI", 0.0, 67.1, 25.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.47)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

    # Create a dictionary from user input
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Make predictions
    predictions = model.predict(df)

    # Display the prediction
    if predictions[0] == 0:
        st.write("The person is not diabetic.")
    else:
        st.write("The person is diabetic.")


if __name__ == '__main__':
    main()
