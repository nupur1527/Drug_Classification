import joblib
import streamlit as st
import pandas as pd


header = st.container()
dataset = st.container()
features = st.container()
model_Training = st.container()


with header:
    st.title("Welcome to my Machine Learning Project!!!")
    st.text("This database contains information about certain drug types.")

with dataset:
    st.header('Drug Classification Dataset')
    with st.expander("See explanation"):
        st.write("This would be a great opportunity to try some *techniques* to predict the outcome of the drugs that might be *accurate* for the patient.")
    drug_data = pd.read_csv(r"C:\Users\nupur\Desktop\drug200.csv")
    st.write(drug_data.head())

    st.subheader('Age distribution on Drug Classification dataset')
    age_dist = pd.DataFrame(drug_data['Age'].value_counts())
    st.bar_chart(age_dist)

    st.subheader(
        'Sodium to Potassium Ratio distribution on Drug Classification dataset')
    st.area_chart(drug_data['Na_to_K'])

    st.subheader('Drug Type Distribution on Drug Classification dataset')
    st.area_chart(drug_data['Drug'].value_counts())
    

with features:
    st.header("The features I created")
    st.text('The feature sets are:')
    st.text('1. Age')
    st.text('2. Sex')
    st.text('3. Blood Pressure Levels (BP)')
    st.text('4. Cholesterol Levels')
    st.text('5. Sodium to Potassium Ration')
    st.text('6. Sodium to Potassium Ratio bigger than 15')

with model_Training:
    st.header("It's time to train the model!")

    sel_col, disp_col= st.columns(2)

    Age = sel_col.number_input("Please Enter Your Age", min_value=10, max_value=74)
    Sex = sel_col.selectbox("Select Your Gender", ('Male', 'Female'))
    if Sex == "Male":
        Sex = 1
    elif Sex == "Female":
        Sex = 0

    BP = sel_col.selectbox('what is your Blood Pressure type?',
                      ('High', 'Normal', 'Low'))
    if BP == 'High':
        BP = 0
    elif BP == 'Low':
        BP = 1
    elif BP == 'Normal':
        BP = 2

    Cholesterol = sel_col.selectbox('what is your Cholesterol type?', ('High', 'Normal'))
    if Cholesterol == 'High':
        Cholesterol = 0
    elif Cholesterol == 'Normal':
        Cholesterol = 1

    Na_to_K = sel_col.number_input("Enter Your Sodium to potassium value", min_value=0, max_value=100)

    if Na_to_K > 15:
        Na_to_K_bigger_than_15 = 1
    else:
        Na_to_K_bigger_than_15 = 0

    df = pd.DataFrame([[Age, Sex, BP, Cholesterol, Na_to_K, Na_to_K_bigger_than_15]], columns=[
                  'Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Na_to_K_bigger_than_15'])
    disp_col.subheader('Given values:')
    disp_col.write(df)

    model = joblib.load(r"C:\Users\nupur\Desktop\data.sav")
    Y_predict = model.predict(df)


    disp_col.subheader("Predicted Drug Type:")
    if Y_predict==0:
        disp_col.write('Drug Y')
    elif Y_predict== 1:
        disp_col.write('Drug A')
    elif Y_predict==2:
        disp_col.write('Drug B')
    elif Y_predict==3:
        disp_col.write('Drug C')
    elif Y_predict==4:
        disp_col.write('Drug X')

        
    disp_col.subheader('Accuracy:')
    disp_col.write("0.975")
    disp_col.subheader('mean_squared_error:')
    disp_col.write("0.025",)


#To run this app first change the directory where this file is stored then run using streamlit run app2.py in terminal.