import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stSlider > div > div > div {
        color: #1f77b4;
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

@st.cache_data
def get_feature_ranges():
    """Get feature ranges from training data for validation"""
    try:
        df = pd.read_csv('cardio_train.csv', sep=';', nrows=10000)
        df['age'] = (df['age'] / 365).astype(int)
        ranges = {
            'age': (df['age'].min(), df['age'].max()),
            'height': (df['height'].min(), df['height'].max()),
            'weight': (df['weight'].min(), df['weight'].max()),
            'ap_hi': (df['ap_hi'].min(), df['ap_hi'].max()),
            'ap_lo': (df['ap_lo'].min(), df['ap_lo'].max()),
        }
        return ranges
    except:
        return None

def preprocess_input(data_dict):
    """Preprocess input data for prediction"""
    # Create DataFrame from input dictionary
    # Ensure correct column order (matching training data: excluding 'id' and 'cardio')
    # Age is already in years (as per model training preprocessing)
    feature_order = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df = pd.DataFrame([data_dict], columns=feature_order)
    
    # Return DataFrame to preserve feature names (model was trained with feature names)
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient information to predict cardiovascular disease risk</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Model file not found. Please ensure 'model.pkl' exists in the project directory.")
        return
    
    # Get feature ranges for validation
    ranges = get_feature_ranges()
    
    # VERTICAL LAYOUT - All fields stack one by one
    st.markdown("### 📋 Patient Information")
    
    # Age
    age = st.slider(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=50,
        help="Patient's age in years"
    )
    
    # Gender
    gender_option = st.selectbox(
        "Gender",
        options=[1, 2],
        format_func=lambda x: "Female" if x == 1 else "Male",
        help="Patient's gender"
    )
    
    # Height
    height = st.slider(
        "Height (cm)",
        min_value=120,
        max_value=220,
        value=170,
        help="Patient's height in centimeters"
    )
    
    # Weight
    weight = st.slider(
        "Weight (kg)",
        min_value=30,
        max_value=200,
        value=70,
        help="Patient's weight in kilograms"
    )
    
    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    st.markdown("---")
    st.metric("BMI (Body Mass Index)", f"{bmi:.2f}", 
             help="BMI = Weight (kg) / Height (m)²")
    
    # BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_category = "Normal"
        bmi_color = "green"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "orange"
    else:
        bmi_category = "Obese"
        bmi_color = "red"
    
    st.markdown(f"**Category:** <span style='color: {bmi_color}'>{bmi_category}</span>", unsafe_allow_html=True)
    
    # Blood Pressure
    st.markdown("---")
    st.markdown("### 🩺 Blood Pressure")
    ap_hi = st.number_input(
        "Systolic BP (ap_hi)",
        min_value=80,
        max_value=250,
        value=120,
        help="Systolic blood pressure (upper number)"
    )
    ap_lo = st.number_input(
        "Diastolic BP (ap_lo)",
        min_value=40,
        max_value=200,
        value=80,
        help="Diastolic blood pressure (lower number)"
    )
    
    # Validate BP
    if ap_hi <= ap_lo:
        st.warning("⚠️ Systolic BP should be higher than Diastolic BP")
    
    # Medical History & Lifestyle
    st.markdown("---")
    st.markdown("### 🏥 Medical History & Lifestyle")
    
    # Cholesterol
    cholesterol = st.selectbox(
        "Cholesterol Level",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        help="Cholesterol level"
    )
    
    # Glucose
    gluc = st.selectbox(
        "Glucose Level",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        help="Glucose level"
    )
    
    # Lifestyle factors
    st.markdown("---")
    st.markdown("#### Lifestyle Factors")
    
    smoke = st.radio(
        "Smoking",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Does the patient smoke?"
    )
    
    alco = st.radio(
        "Alcohol Intake",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Does the patient consume alcohol?"
    )
    
    active = st.radio(
        "Physical Activity",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Does the patient engage in physical activity?"
    )
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔮 Predict Cardiovascular Disease Risk", 
                              type="primary", 
                              use_container_width=True)
    
    if predict_button:
        # Prepare input data
        input_data = {
            'age': age,
            'gender': gender_option,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }
        
        # Validate BP
        if ap_hi <= ap_lo:
            st.error("❌ Invalid blood pressure values. Please correct and try again.")
            return
        
        # Preprocess and predict
        try:
            X_input = preprocess_input(input_data)
            prediction = model.predict(X_input)[0]
            prediction_proba = model.predict_proba(X_input)[0]
            
            # Display results
            st.markdown("---")
            
            # Result box
            if prediction == 1:
                st.markdown("""
                    <div class="prediction-box prediction-positive">
                        <h2 style="margin: 0; font-size: 2.5rem;">⚠️ HIGH RISK</h2>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">Cardiovascular disease is predicted</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box prediction-negative">
                        <h2 style="margin: 0; font-size: 2.5rem;">✅ LOW RISK</h2>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">No cardiovascular disease predicted</p>
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
            st.markdown("### Probability Distribution")
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
                st.markdown("### 📊 Feature Importance")
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
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses a **Decision Tree Classifier** 
        to predict cardiovascular disease risk based on patient 
        medical and lifestyle information.
        
        **Model Features:**
        - Age, Gender, Height, Weight
        - Blood Pressure (Systolic & Diastolic)
        - Cholesterol & Glucose Levels
        - Lifestyle Factors (Smoking, Alcohol, Activity)
        
        **Note:** This is a predictive tool and should not 
        replace professional medical advice.
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Fill in all patient information
        2. Ensure blood pressure values are valid
        3. Click the predict button
        4. Review the results and recommendations
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.markdown(f"""
        **Model Type:** Decision Tree Classifier
        
        **Status:** {'✅ Loaded' if model is not None else '❌ Not Loaded'}
        """)
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.markdown("""
        All data is processed locally. 
        No information is stored or transmitted.
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        st.code("""
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
        """, language="bash")

if __name__ == "__main__":
    main()
