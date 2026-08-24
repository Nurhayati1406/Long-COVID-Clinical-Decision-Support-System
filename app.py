import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SET UP THE DASHBOARD UI ---
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

# Initialize session state so results don't disappear
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

    # Pre-process the data immediately so it is always available for SHAP
    processed_data = input_data.copy()
    processed_data['Gender'] = encoders['Gender'].transform(processed_data['Gender'])
    processed_data['COVID_Severity'] = encoders['COVID_Severity'].transform(processed_data['COVID_Severity'])
    processed_data['Hospitalized'] = encoders['Hospitalized'].transform(processed_data['Hospitalized'])
    processed_data['Physical_Activity_Level'] = encoders['Physical_Activity_Level'].transform(processed_data['Physical_Activity_Level'])

    # --- LAYOUT: COLUMNS ---
    col1, col2 = st.columns([1, 1.5]) 

    with col1:
        st.subheader("Patient Profile Summary")
        st.table(input_data.T.rename(columns={0: "Patient Data"})) 

    with col2:
        st.subheader("Risk Assessment Engine")
        
        # --- PREDICTION BUTTON ---
        if st.button("Predict Long COVID Risk", type="primary"):
            with st.spinner("Analyzing complex relationships in healthcare data..."):
                
                # Make prediction
                raw_prediction = model.predict(processed_data)
                final_prediction = target_encoder.inverse_transform(raw_prediction)[0]
                
                # Check for probabilities
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(processed_data)[0]
                    risk_prob = np.max(probs) * 100 
                    st.session_state.probability = f"{risk_prob:.1f}% confidence"
                else:
                    st.session_state.probability = "Probability not available"

                # Save to session state
                st.session_state.prediction = final_prediction

        # --- DISPLAY RESULTS & SHAP ---
        if st.session_state.prediction is not None:
            st.success("Analysis Complete!")
            
            # Display metrics visually
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(label="Predicted Brain Fog Outcome", value=st.session_state.prediction)
            
            if st.session_state.probability:
                metric_col2.metric(label="Model Confidence / Risk", value=st.session_state.probability)
            
            # --- SHAP EXPLAINABILITY ---
            st.markdown("---")
            st.subheader("📊 Prediction Explanation (SHAP)")
            st.write("This chart shows exactly which factors increased (red) or decreased (blue) the patient's risk.")
            
            with st.spinner("Generating AI explanation..."):
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    
                    # Generate SHAP explanation
                    explainer = shap.Explainer(model) 
                    shap_values = explainer(processed_data)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values[0], show=False) 
                    st.pyplot(fig)
                    plt.clf() 
                    
                except ImportError:
                    st.error("⚠️ Missing libraries. Please ensure 'shap' and 'matplotlib' are in your requirements.txt file.")
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation for this model type. Error: {e}")
