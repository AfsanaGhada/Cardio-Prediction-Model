import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Predictor - Step by Step",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem 0;
        min-height: 450px;
    }
    .step-icon {
        font-size: 5rem;
        text-align: center;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .step-title {
        font-size: 2.5rem;
        color: #667eea;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .step-description {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .review-page {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
    }
    .prediction-box {
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .progress-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 30px;
        margin: 2rem 0;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .summary-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(data_dict):
    """Preprocess input data for prediction"""
    feature_order = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df = pd.DataFrame([data_dict], columns=feature_order)
    return df

# Define all steps with their information
STEPS = [
    {
        'id': 'age',
        'title': '👤 Patient Age',
        'description': 'Enter the patient\'s age in years',
        'icon': '👤'
    },
    {
        'id': 'gender',
        'title': '⚧️ Gender',
        'description': 'Select the patient\'s gender',
        'icon': '⚧️'
    },
    {
        'id': 'height',
        'title': '📏 Height',
        'description': 'Enter the patient\'s height in centimeters',
        'icon': '📏'
    },
    {
        'id': 'weight',
        'title': '⚖️ Weight',
        'description': 'Enter the patient\'s weight in kilograms',
        'icon': '⚖️'
    },
    {
        'id': 'ap_hi',
        'title': '🩺 Systolic Blood Pressure',
        'description': 'Enter the systolic (upper) blood pressure reading',
        'icon': '🩺'
    },
    {
        'id': 'ap_lo',
        'title': '🩺 Diastolic Blood Pressure',
        'description': 'Enter the diastolic (lower) blood pressure reading',
        'icon': '🩺'
    },
    {
        'id': 'cholesterol',
        'title': '🧪 Cholesterol Level',
        'description': 'Select the patient\'s cholesterol level',
        'icon': '🧪'
    },
    {
        'id': 'gluc',
        'title': '🍬 Glucose Level',
        'description': 'Select the patient\'s glucose level',
        'icon': '🍬'
    },
    {
        'id': 'smoke',
        'title': '🚬 Smoking Status',
        'description': 'Does the patient smoke?',
        'icon': '🚬'
    },
    {
        'id': 'alco',
        'title': '🍷 Alcohol Intake',
        'description': 'Does the patient consume alcohol?',
        'icon': '🍷'
    },
    {
        'id': 'active',
        'title': '🏃 Physical Activity',
        'description': 'Does the patient engage in physical activity?',
        'icon': '🏃'
    }
]

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

def get_progress_percentage():
    """Calculate progress percentage"""
    return int(((st.session_state.current_step + 1) / len(STEPS)) * 100)

def render_progress_bar():
    """Render the progress bar"""
    progress = get_progress_percentage()
    st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress}%">
                Step {st.session_state.current_step + 1} of {len(STEPS)} ({progress}%)
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_step():
    """Render the current step"""
    step = STEPS[st.session_state.current_step]
    step_id = step['id']
    
    st.markdown(f'<div class="step-container">', unsafe_allow_html=True)
    
    # Large icon display
    st.markdown(f'<div class="step-icon">{step["icon"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="step-title">{step["title"]}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="step-description">{step["description"]}</p>', unsafe_allow_html=True)
    
    # Add visual separator
    st.markdown("---")
    
    # Render appropriate input based on step
    if step_id == 'age':
        value = st.session_state.patient_data.get('age', 50)
        # Visual age display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">👤</div>', unsafe_allow_html=True)
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=value,
            key='age_input',
            help="Patient's age in years"
        )
        st.session_state.patient_data['age'] = age
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #667eea; font-weight: bold; margin: 1rem 0;">{age} years old</div>', unsafe_allow_html=True)
        st.info("💡 Age is an important factor in cardiovascular risk assessment")
        
    elif step_id == 'gender':
        value = st.session_state.patient_data.get('gender', 2)
        # Visual gender display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">⚧️</div>', unsafe_allow_html=True)
        gender = st.radio(
            "Select Gender",
            options=[1, 2],
            index=0 if value == 1 else 1,
            format_func=lambda x: "👩 Female" if x == 1 else "👨 Male",
            key='gender_input',
            help="Patient's gender"
        )
        st.session_state.patient_data['gender'] = gender
        selected_gender = "👩 Female" if gender == 1 else "👨 Male"
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {selected_gender}</div>', unsafe_allow_html=True)
        
    elif step_id == 'height':
        value = st.session_state.patient_data.get('height', 170)
        # Visual height display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">📏</div>', unsafe_allow_html=True)
        height = st.slider(
            "Height (cm)",
            min_value=120,
            max_value=220,
            value=value,
            key='height_input',
            help="Patient's height in centimeters"
        )
        st.session_state.patient_data['height'] = height
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #667eea; font-weight: bold; margin: 1rem 0;">{height} cm</div>', unsafe_allow_html=True)
        st.info(f"📊 Current height: {height} cm ({height/100:.2f} meters)")
        
    elif step_id == 'weight':
        value = st.session_state.patient_data.get('weight', 70)
        # Visual weight display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">⚖️</div>', unsafe_allow_html=True)
        weight = st.slider(
            "Weight (kg)",
            min_value=30,
            max_value=200,
            value=value,
            key='weight_input',
            help="Patient's weight in kilograms"
        )
        st.session_state.patient_data['weight'] = weight
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #667eea; font-weight: bold; margin: 1rem 0;">{weight} kg</div>', unsafe_allow_html=True)
        # Calculate BMI if height is available
        if 'height' in st.session_state.patient_data:
            height_m = st.session_state.patient_data['height'] / 100
            bmi = weight / (height_m ** 2)
            if bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "blue"
            elif bmi < 25:
                bmi_status = "Normal"
                bmi_color = "green"
            elif bmi < 30:
                bmi_status = "Overweight"
                bmi_color = "orange"
            else:
                bmi_status = "Obese"
                bmi_color = "red"
            st.metric("BMI", f"{bmi:.2f}", f"Category: {bmi_status}")
        
    elif step_id == 'ap_hi':
        value = st.session_state.patient_data.get('ap_hi', 120)
        # Visual BP display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🩺</div>', unsafe_allow_html=True)
        ap_hi = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=250,
            value=value,
            key='ap_hi_input',
            help="Systolic blood pressure - the upper number in a BP reading"
        )
        st.session_state.patient_data['ap_hi'] = ap_hi
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #667eea; font-weight: bold; margin: 1rem 0;">{ap_hi} mmHg</div>', unsafe_allow_html=True)
        if ap_hi < 120:
            st.success("✅ Normal systolic pressure (< 120 mmHg)")
        elif ap_hi < 130:
            st.info("ℹ️ Elevated systolic pressure (120-129 mmHg)")
        else:
            st.warning("⚠️ High systolic pressure (≥ 130 mmHg)")
            
    elif step_id == 'ap_lo':
        value = st.session_state.patient_data.get('ap_lo', 80)
        # Visual BP display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🩺</div>', unsafe_allow_html=True)
        ap_lo = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=200,
            value=value,
            key='ap_lo_input',
            help="Diastolic blood pressure - the lower number in a BP reading"
        )
        st.session_state.patient_data['ap_lo'] = ap_lo
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #667eea; font-weight: bold; margin: 1rem 0;">{ap_lo} mmHg</div>', unsafe_allow_html=True)
        # Validate with systolic
        if 'ap_hi' in st.session_state.patient_data:
            ap_hi = st.session_state.patient_data['ap_hi']
            if ap_hi <= ap_lo:
                st.error("❌ Systolic BP must be higher than Diastolic BP!")
            else:
                st.success(f"✅ Valid BP Reading: {ap_hi}/{ap_lo} mmHg")
        
    elif step_id == 'cholesterol':
        value = st.session_state.patient_data.get('cholesterol', 1)
        # Visual cholesterol display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🧪</div>', unsafe_allow_html=True)
        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            index=value-1,
            format_func=lambda x: {
                1: "✅ Normal",
                2: "⚠️ Above Normal", 
                3: "🔴 Well Above Normal"
            }[x],
            key='cholesterol_input',
            help="Patient's cholesterol level"
        )
        st.session_state.patient_data['cholesterol'] = cholesterol
        cholesterol_text = {1: "✅ Normal", 2: "⚠️ Above Normal", 3: "🔴 Well Above Normal"}[cholesterol]
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {cholesterol_text}</div>', unsafe_allow_html=True)
        
    elif step_id == 'gluc':
        value = st.session_state.patient_data.get('gluc', 1)
        # Visual glucose display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🍬</div>', unsafe_allow_html=True)
        gluc = st.selectbox(
            "Glucose Level",
            options=[1, 2, 3],
            index=value-1,
            format_func=lambda x: {
                1: "✅ Normal",
                2: "⚠️ Above Normal",
                3: "🔴 Well Above Normal"
            }[x],
            key='gluc_input',
            help="Patient's glucose level"
        )
        st.session_state.patient_data['gluc'] = gluc
        gluc_text = {1: "✅ Normal", 2: "⚠️ Above Normal", 3: "🔴 Well Above Normal"}[gluc]
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {gluc_text}</div>', unsafe_allow_html=True)
        
    elif step_id == 'smoke':
        value = st.session_state.patient_data.get('smoke', 0)
        # Visual smoking display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🚬</div>', unsafe_allow_html=True)
        smoke = st.radio(
            "Does the patient smoke?",
            options=[0, 1],
            index=value,
            format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
            key='smoke_input',
            help="Smoking status"
        )
        st.session_state.patient_data['smoke'] = smoke
        smoke_text = "✅ Yes" if smoke == 1 else "❌ No"
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {smoke_text}</div>', unsafe_allow_html=True)
        if smoke == 1:
            st.warning("⚠️ Smoking significantly increases cardiovascular risk")
        else:
            st.success("✅ Great! Not smoking is beneficial for heart health")
        
    elif step_id == 'alco':
        value = st.session_state.patient_data.get('alco', 0)
        # Visual alcohol display
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🍷</div>', unsafe_allow_html=True)
        alco = st.radio(
            "Does the patient consume alcohol?",
            options=[0, 1],
            index=value,
            format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
            key='alco_input',
            help="Alcohol consumption status"
        )
        st.session_state.patient_data['alco'] = alco
        alco_text = "✅ Yes" if alco == 1 else "❌ No"
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {alco_text}</div>', unsafe_allow_html=True)
        
    elif step_id == 'active':
        value = st.session_state.patient_data.get('active', 1)
        # Visual activity display with animated emoji
        st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🏃</div>', unsafe_allow_html=True)
        active = st.radio(
            "Does the patient engage in physical activity?",
            options=[0, 1],
            index=value,
            format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
            key='active_input',
            help="Physical activity status"
        )
        st.session_state.patient_data['active'] = active
        active_text = "✅ Yes" if active == 1 else "❌ No"
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #667eea; font-weight: bold; margin: 1rem 0;">Selected: {active_text}</div>', unsafe_allow_html=True)
        if active == 1:
            st.success("✅ Regular physical activity is beneficial for heart health")
        else:
            st.info("💡 Consider adding physical activity to your routine for better heart health")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_summary():
    """Render summary of all entered data - This is now a separate page"""
    # Page header with visual design
    st.markdown("""
        <div class="review-page">
            <h1 style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">📋 Review Your Information</h1>
            <p style="text-align: center; font-size: 1.3rem; margin-bottom: 0;">Please review all the information before making a prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    data = st.session_state.patient_data
    
    col1, col2 = st.columns(2)
    

def render_prediction(model):
    """Render prediction results"""
    data = st.session_state.patient_data
    
    # Validate BP
    if data.get('ap_hi', 0) <= data.get('ap_lo', 0):
        st.error("❌ Invalid blood pressure values. Please go back and correct them.")
        return
    
    try:
        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]
        
        # Display results
        st.markdown("---")
        
        # Result box
        if prediction == 1:
            st.markdown("""
                <div class="prediction-box prediction-positive">
                    <h2 style="margin: 0; font-size: 3rem;">⚠️ HIGH RISK</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">Cardiovascular disease is predicted</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-box prediction-negative">
                    <h2 style="margin: 0; font-size: 3rem;">✅ LOW RISK</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">No cardiovascular disease predicted</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Probability metrics
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                "Risk Probability",
                f"{prediction_proba[1]*100:.2f}%",
                help="Probability of having cardiovascular disease"
            )
        with col4:
            st.metric(
                "Safe Probability",
                f"{prediction_proba[0]*100:.2f}%",
                help="Probability of not having cardiovascular disease"
            )
        
        # Visual probability bar
        st.markdown("### 📊 Probability Distribution")
        prob_df = pd.DataFrame({
            'Category': ['No Disease', 'Disease'],
            'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
        })
        st.bar_chart(prob_df.set_index('Category'))
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **Based on the prediction, consider the following:**
            - Consult with a healthcare professional for a comprehensive evaluation
            - Monitor blood pressure regularly
            - Maintain a healthy diet and exercise routine
            - Reduce or eliminate smoking and alcohol consumption if applicable
            - Follow up with regular medical check-ups
            """)
        else:
            st.info("""
            **Great! Keep maintaining a healthy lifestyle:**
            - Continue regular physical activity
            - Maintain a balanced diet
            - Monitor your health metrics regularly
            - Schedule routine medical check-ups
            """)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("### 📈 Feature Importance")
            feature_names = ['Age', 'Gender', 'Height', 'Weight', 'Systolic BP', 
                           'Diastolic BP', 'Cholesterol', 'Glucose', 'Smoking', 
                           'Alcohol', 'Activity']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature'))
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.exception(e)

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient information step by step to predict cardiovascular disease risk</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Model file not found. Please ensure 'model.pkl' exists in the project directory.")
        return
    
    # Show review page (separate page after clicking "Review & Predict")
    if st.session_state.show_summary and not st.session_state.prediction_made:
        # This is the REVIEW PAGE - separate from wizard steps
        render_summary()
        
        # Get patient data
        data = st.session_state.patient_data
        
        # Display all information in a nice layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Personal Information")
            st.markdown(f'<div class="summary-card"><strong>👤 Age:</strong> {data.get("age", "N/A")} years</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>⚧️ Gender:</strong> {"👩 Female" if data.get("gender") == 1 else "👨 Male"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>📏 Height:</strong> {data.get("height", "N/A")} cm</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>⚖️ Weight:</strong> {data.get("weight", "N/A")} kg</div>', unsafe_allow_html=True)
            if 'height' in data and 'weight' in data:
                height_m = data['height'] / 100
                bmi = data['weight'] / (height_m ** 2)
                bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                st.markdown(f'<div class="summary-card"><strong>📊 BMI:</strong> {bmi:.2f} ({bmi_status})</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🏥 Medical Information")
            st.markdown(f'<div class="summary-card"><strong>🩺 Systolic BP:</strong> {data.get("ap_hi", "N/A")} mmHg</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>🩺 Diastolic BP:</strong> {data.get("ap_lo", "N/A")} mmHg</div>', unsafe_allow_html=True)
            cholesterol_map = {1: "✅ Normal", 2: "⚠️ Above Normal", 3: "🔴 Well Above Normal"}
            st.markdown(f'<div class="summary-card"><strong>🧪 Cholesterol:</strong> {cholesterol_map.get(data.get("cholesterol"), "N/A")}</div>', unsafe_allow_html=True)
            gluc_map = {1: "✅ Normal", 2: "⚠️ Above Normal", 3: "🔴 Well Above Normal"}
            st.markdown(f'<div class="summary-card"><strong>🍬 Glucose:</strong> {gluc_map.get(data.get("gluc"), "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>🚬 Smoking:</strong> {"✅ Yes" if data.get("smoke") == 1 else "❌ No"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>🍷 Alcohol:</strong> {"✅ Yes" if data.get("alco") == 1 else "❌ No"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-card"><strong>🏃 Physical Activity:</strong> {"✅ Yes" if data.get("active") == 1 else "❌ No"}</div>', unsafe_allow_html=True)
        
        # Navigation buttons on review page
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Go Back to Edit", use_container_width=True):
                st.session_state.show_summary = False
                st.session_state.current_step = len(STEPS) - 1
                st.rerun()
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.current_step = 0
                st.session_state.patient_data = {}
                st.session_state.show_summary = False
                st.session_state.prediction_made = False
                st.rerun()
        with col3:
            if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                st.session_state.prediction_made = True
                st.rerun()
        
        return
    
    # Show prediction results page (after clicking "Make Prediction" on review page)
    if st.session_state.prediction_made:
        render_prediction(model)
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Review", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
        with col2:
            if st.button("🔄 Start New Prediction", type="primary", use_container_width=True):
                st.session_state.current_step = 0
                st.session_state.patient_data = {}
                st.session_state.show_summary = False
                st.session_state.prediction_made = False
                st.rerun()
        return
    
    # Render progress bar
    render_progress_bar()
    
    # Render current step
    render_step()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_step = 0
            st.session_state.patient_data = {}
            st.session_state.show_summary = False
            st.session_state.prediction_made = False
            st.rerun()
    
    with col3:
        if st.session_state.current_step < len(STEPS) - 1:
            if st.button("➡️ Next", type="primary", use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("✅ Review & Predict", type="primary", use_container_width=True):
                # Validate all data is collected
                if len(st.session_state.patient_data) == len(STEPS):
                    st.session_state.show_summary = True
                    st.rerun()
                else:
                    st.error("Please complete all steps before proceeding.")

if __name__ == "__main__":
    main()

