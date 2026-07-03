import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
#st.write(" ")

df=pd.read_csv("final.csv")
st.title("    🩺 Diabetes Prediction App")
st.sidebar.write("🩺 Diabetes Prediction System predict your diabetes risk using machine learning")
st.write(" ")
st.sidebar.image("p3.avif")
st.sidebar.markdown("_______")
st.sidebar.subheader("ℹ️ About Diabetes")

st.sidebar.write("Diabetes affects how your body processes blood sugar.")
st.sidebar.image("level.webp")
model = pickle.load(open("model.pkl", "rb"))
st.markdown("""
<style>
.stApp {
    background-color:#E3F0F8;
}
</style>
""", unsafe_allow_html=True)
st.subheader("👤 Patient Information")
st.markdown("______")
col1,col2=st.columns([1,1],gap='large')
with col1:
    #st.write(" ")
    pregnancies = st.number_input(" 🤰Pregnancies", min_value=0)
    glucose = st.number_input(" 💧Glucose", min_value=0)
    bloodpressure = st.number_input(" 🫀Blood Pressure", min_value=0)
    skinthickness = st.number_input(" 📏Skin Thickness", min_value=0)
with col2:    
    insulin = st.number_input(" 💉Insulin", min_value=0)
    bmi = st.number_input(" ⚖️BMI", min_value=0.0)
    dpf = st.number_input(" 👨‍👩‍👧‍👦Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input(" 📅Age", min_value=0)

if st.button("**Predict diabetes**"):

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
    st.subheader("Predicted Result 🩺"
    "")

    if result[0] == 1:
        st.error("⚠️ Diabetes Detected")

    else:
        st.success("✅ No Diabetes Detected")
            
        st.subheader("📊 Patient graph")

        data = pd.DataFrame(data=[
                ["Glucose", glucose],
                ["BMI", bmi],
                ["Age", age],
                ["BP", bloodpressure]
            ],
        columns=["Feature", "Value"]
        )

        st.bar_chart(data.set_index("Feature"),color="#AFCFE3")
        st.write(data)
    st.subheader("Health Tips")
    st.write("🍎 Healthy Diet")
    st.write("🏃 Regular Exercise")
    st.write("😴 Good Sleep")
    st.write("⚖️ Maintain Weight")

    st.markdown("_____")
    st.subheader(" 💡Stay Healthy")
    st.write("Small healthy habits today can lead to a healthier tomorrow.")
