import streamlit as st
import pandas as pd
import pickle
st.title("    🩺 Diabetes Prediction App")

st.image("p3.avif")
df=pd.read_csv("final.csv")


model = pickle.load(open("model.pkl", "rb"))



st.markdown("""
<style>
.stApp {
    background-color:#E8F2FD;
}
</style>
""", unsafe_allow_html=True)

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0)
glucose = st.sidebar.number_input("Glucose", min_value=0)
bloodpressure = st.sidebar.number_input("Blood Pressure", min_value=0)
skinthickness = st.sidebar.number_input("Skin Thickness", min_value=0)
insulin = st.sidebar.number_input("Insulin", min_value=0)
bmi = st.sidebar.number_input("BMI", min_value=0.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.sidebar.number_input("Age", min_value=0)

if st.sidebar.button("Predict"):

    myinput = [[pregnancies, glucose, bloodpressure,
                skinthickness, insulin, bmi,
                dpf, age]]

    columns = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    myinput = pd.DataFrame(myinput, columns=columns)

    result = model.predict(myinput)
    st.sidebar.subheader("Prediction Result")

    if result[0] == 1:
        st.sidebar.error("⚠️ Diabetes Detected")
    else:
        st.sidebar.success("✅ No Diabetes Detected")
