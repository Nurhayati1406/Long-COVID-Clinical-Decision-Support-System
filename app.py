import streamlit as st
import pandas as pd
import joblib

# --- 1. LOAD THE TRAINED MODEL AND ENCODERS ---
@st.cache_resource
def load_assets():
    try:
        # Load the file
        saved_assets = joblib.load('model.pkl')
        
        # Depending on how you saved it in your training script, unpack it
        # Assuming you saved a tuple: (model, encoders, target_encoder)
        model = saved_assets[0]
        encoders = saved_assets[1]
        target_encoder = saved_assets[2]
        
        # RETURN them to the dashboard (This breaks the loop!)
        return model, encoders, target_encoder
        
    except Exception as e:
        st.error(f"🚨 THE REAL ERROR IS: {e}")
        return None, None, None

# Call the function OUTSIDE the loop, pushed to the left wall
model, encoders, target_encoder = load_assets()

# --- 2. SET UP THE DASHBOARD UI ---
st.set_page_config(page_title="Long COVID CDSS", layout="wide")
st.title("Zx Long COVID Clinical Decision Support System")
st.markdown("Enter patient acute infection details to predict the risk of developing Brain Fog post-recovery.")

if model is None:
    st.warning("G 'model.pkl' not found. Please run your training script or cell first to generate the model assets!")
else:
    # --- 3. SIDEBAR: PATIENT DATA INPUT ---
    st.sidebar.header("Patient History Input")

    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", encoders['Gender'].classes_)
    covid_severity = st.sidebar.selectbox("COVID Severity", encoders['COVID_Severity'].classes_)
    hospitalized = st.sidebar.selectbox("Hospitalized?", encoders['Hospitalized'].classes_)
    days_to_recovery = st.sidebar.slider("Days to Recovery (Acute)", 1, 60, 14)
    physical_activity = st.sidebar.selectbox("Current Physical Activity Level", encoders['Physical_Activity_Level'].classes_)

    # Store user input into a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'COVID_Severity': [covid_severity],
        'Hospitalized': [hospitalized],
        'Days_to_Recovery': [days_to_recovery],
        'Physical_Activity_Level': [physical_activity]
    })

    st.subheader("Patient Profile Summary")
    st.table(input_data)

    # --- 4. PREDICTION ENGINE ---
    st.markdown("---")
    st.subheader("Risk Assessment Engine")

    if st.button("Predict Long COVID Risk"):
        with st.spinner("Analyzing complex relationships in healthcare data..."):
            processed_data = input_data.copy()
            processed_data['Gender'] = encoders['Gender'].transform(processed_data['Gender'])
            processed_data['COVID_Severity'] = encoders['COVID_Severity'].transform(processed_data['COVID_Severity'])
            processed_data['Hospitalized'] = encoders['Hospitalized'].transform(processed_data['Hospitalized'])
            processed_data['Physical_Activity_Level'] = encoders['Physical_Activity_Level'].transform(processed_data['Physical_Activity_Level'])
            
            raw_prediction = model.predict(processed_data)
            final_prediction = target_encoder.inverse_transform(raw_prediction)[0]
            
            st.success("Analysis Complete!")
            st.metric(label="Predicted Brain Fog Outcome", value=final_prediction)
            st.info("B Note: In future deployments, this panel will integrate Explainable AI (SHAP) to visually illustrate how features like Age and COVID Severity contributed to this specific prediction.")
