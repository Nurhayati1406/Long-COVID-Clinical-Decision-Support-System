import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SET UP THE DASHBOARD UI ---
# (Moved to the top before any visual elements are rendered)
st.set_page_config(page_title="Long COVID CDSS", page_icon="🩺", layout="wide")

# --- 2. LOAD THE TRAINED MODEL AND ENCODERS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        return model, encoders, target_encoder
    except Exception as e:
        st.error(f"🚨 Failed to load model assets. Error: {e}")
        return None, None, None

model, encoders, target_encoder = load_assets()

# Initialize session state for prediction so it doesn't disappear on widget interaction
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probability' not in st.session_state:
    st.session_state.probability = None

# --- 3. MAIN APP ---
st.title("🩺 Long COVID Clinical Decision Support System")
st.markdown("Enter patient acute infection details to predict the risk of developing Brain Fog post-recovery.")

if model is None or encoders is None:
    st.warning("⚠️ Model assets not found. Please ensure 'model.pkl', 'encoders.pkl', and 'target_encoder.pkl' are in the same directory.")
else:
    # --- SIDEBAR: PATIENT DATA INPUT ---
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

    # --- LAYOUT: COLUMNS ---
    col1, col2 = st.columns([1, 1.5]) # Left column slightly narrower than right

    with col1:
        st.subheader("Patient Profile Summary")
        # Transpose table for better vertical reading in a column
        st.table(input_data.T.rename(columns={0: "Patient Data"})) 

    with col2:
        st.subheader("Risk Assessment Engine")
        
        # --- PREDICTION LOGIC ---
        if st.button("Predict Long COVID Risk", type="primary"):
            with st.spinner("Analyzing complex relationships in healthcare data..."):
                # Data processing
                processed_data = input_data.copy()
                processed_data['Gender'] = encoders['Gender'].transform(processed_data['Gender'])
                processed_data['COVID_Severity'] = encoders['COVID_Severity'].transform(processed_data['COVID_Severity'])
                processed_data['Hospitalized'] = encoders['Hospitalized'].transform(processed_data['Hospitalized'])
                processed_data['Physical_Activity_Level'] = encoders['Physical_Activity_Level'].transform(processed_data['Physical_Activity_Level'])
                
                # Make prediction
                raw_prediction = model.predict(processed_data)
                final_prediction = target_encoder.inverse_transform(raw_prediction)[0]
                
                # Check if model supports probabilities for better clinical context
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(processed_data)[0]
                    # Assuming target class 1 is "Yes" (Brain Fog). Adjust index if needed.
                    risk_prob = np.max(probs) * 100 
                    st.session_state.probability = f"{risk_prob:.1f}% confidence"
                else:
                    st.session_state.probability = "Probability not available for this model type"

                # Save to session state
                st.session_state.prediction = final_prediction

        # --- DISPLAY RESULTS (Using session state) ---
        if st.session_state.prediction is not None:
            st.success("Analysis Complete!")
            
            # Display metrics visually
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(label="Predicted Brain Fog Outcome", value=st.session_state.prediction)
            
            if st.session_state.probability:
                metric_col2.metric(label="Model Confidence / Risk", value=st.session_state.probability)
            
            # Future deployment note
            st.info("💡 Note: In future deployments, this panel will integrate Explainable AI (SHAP) to visually illustrate how features like Age and COVID Severity contributed to this specific prediction.")
