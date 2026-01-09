import streamlit as st
import fitz  # PyMuPDF
import spacy
import cv2
import whisper
import numpy as np
from PIL import Image
from transformers import pipeline
from io import BytesIO
import json
from googletrans import Translator
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')    

# --- Authentication helpers ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_patient(username, password):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute("SELECT password_hash, patient_id FROM patients WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return hash_password(password) == row[0], row[1] if row else None
    return False, None

def create_patient_account(username, password, name, email, phone):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute('''INSERT INTO patients 
                 (username, password_hash, name, email, phone, created_date)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (username, hashed, name, email, phone, str(datetime.now())))
    conn.commit()
    patient_id = c.lastrowid
    conn.close()
    return patient_id

def init_db():
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    
    # Patients table with authentication
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (patient_id INTEGER PRIMARY KEY, 
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  name TEXT, 
                  age INTEGER, 
                  gender TEXT, 
                  phone TEXT, 
                  email TEXT, 
                  language TEXT, 
                  blood_type TEXT,
                  allergies TEXT, 
                  medications TEXT, 
                  conditions TEXT, 
                  emergency_contact TEXT, 
                  created_date TEXT)''')
    
    # Recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY, 
                  patient_id INTEGER, 
                  date TEXT,
                  recommendation_type TEXT,
                  title TEXT,
                  description TEXT, 
                  priority TEXT,
                  status TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, 
                  patient_id INTEGER, 
                  date TEXT,
                  alert_type TEXT,
                  message TEXT, 
                  priority TEXT,
                  status TEXT,
                  scheduled_time TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    conn.commit()
    conn.close()

init_db()

# --- Session state initialization ---
if 'patient_logged_in' not in st.session_state:
    st.session_state.patient_logged_in = False
if 'patient_username' not in st.session_state:
    st.session_state.patient_username = ""
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None

# Load Models
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")
    speech_model = whisper.load_model("base")
    translator = Translator()
    medical_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    
    return nlp, summarizer, sentiment_analyzer, speech_model, translator, medical_ner

nlp, summarizer, sentiment_analyzer, speech_model, translator, medical_ner = load_models()

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI Medical Assistant Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.alert-card {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin-bottom: 1rem;
}
.login-box {
    background: #ffffff;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    max-width: 450px;
    margin: 2rem auto;
}
.recommendation-card {
    background: #e8f5e9;
    padding: 1.2rem;
    border-radius: 8px;
    border-left: 4px solid #4caf50;
    margin-bottom: 1rem;
}
.high-priority {
    border-left: 4px solid #f44336;
    background: #ffebee;
}
.medium-priority {
    border-left: 4px solid #ff9800;
    background: #fff3e0;
}
.low-priority {
    border-left: 4px solid #2196f3;
    background: #e3f2fd;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏥 AI Medical Assistant Pro - Complete Healthcare Solution</h1></div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")

# Show patient info if logged in
if st.session_state.patient_logged_in:
    st.sidebar.success(f"👤 Patient: {st.session_state.patient_username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.patient_logged_in = False
        st.session_state.patient_username = ""
        st.session_state.patient_id = None
        st.rerun()

page = st.sidebar.selectbox("Choose a section:", [
    "📄 Document Analysis",
    "🖼️ Medical Imaging",
    "🎤 Voice Assistant",
    "🤖 AI Diagnosis Assistant",
    "⚙️ Settings"
])

# Helper functions
def get_patient_recommendations(patient_id):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM recommendations 
                 WHERE patient_id=? 
                 ORDER BY date DESC''', (patient_id,))
    results = c.fetchall()
    conn.close()
    return results

def get_patient_alerts(patient_id):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM alerts 
                 WHERE patient_id=? 
                 ORDER BY date DESC''', (patient_id,))
    results = c.fetchall()
    conn.close()
    return results

def add_sample_data(patient_id):
    """Add sample recommendations and alerts for demo purposes"""
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    
    # Sample recommendations
    recommendations = [
        (patient_id, str(datetime.now()), "Lifestyle", "Increase Physical Activity", 
         "Aim for at least 30 minutes of moderate exercise daily", "High", "Active"),
        (patient_id, str(datetime.now() - timedelta(days=1)), "Diet", "Reduce Sodium Intake", 
         "Limit sodium to 2000mg per day for better blood pressure control", "High", "Active"),
        (patient_id, str(datetime.now() - timedelta(days=2)), "Medication", "Medication Adherence", 
         "Take prescribed medications at the same time every day", "Medium", "Active"),
        (patient_id, str(datetime.now() - timedelta(days=3)), "Prevention", "Annual Health Checkup", 
         "Schedule your annual physical examination", "Low", "Pending")
    ]
    
    for rec in recommendations:
        try:
            c.execute('''INSERT INTO recommendations 
                        (patient_id, date, recommendation_type, title, description, priority, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''', rec)
        except:
            pass
    
    # Sample alerts
    alerts = [
        (patient_id, str(datetime.now()), "Medication", "Time to take Metformin 500mg", 
         "High", "Pending", "09:00 AM"),
        (patient_id, str(datetime.now()), "Appointment", "Doctor visit tomorrow at 2:00 PM", 
         "Medium", "Active", "02:00 PM"),
        (patient_id, str(datetime.now() - timedelta(days=1)), "Health Check", "Weekly blood pressure check due", 
         "Low", "Active", "10:00 AM"),
        (patient_id, str(datetime.now()), "Medication", "Evening medication reminder", 
         "High", "Pending", "09:00 PM")
    ]
    
    for alert in alerts:
        try:
            c.execute('''INSERT INTO alerts 
                        (patient_id, date, alert_type, message, priority, status, scheduled_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''', alert)
        except:
            pass
    
    conn.commit()
    conn.close()

# Medical knowledge base
MEDICAL_CONDITIONS = {
    "diabetes": {
        "symptoms": ["frequent urination", "excessive thirst", "fatigue", "blurred vision"],
        "treatments": ["insulin therapy", "dietary changes", "exercise", "blood sugar monitoring"],
        "lifestyle": ["low sugar diet", "regular exercise", "weight management"]
    },
    "hypertension": {
        "symptoms": ["headaches", "shortness of breath", "chest pain", "dizziness"],
        "treatments": ["ACE inhibitors", "beta blockers", "diuretics", "lifestyle changes"],
        "lifestyle": ["low sodium diet", "regular exercise", "stress management", "limit alcohol"]
    },
    "asthma": {
        "symptoms": ["wheezing", "shortness of breath", "chest tightness", "coughing"],
        "treatments": ["inhaled corticosteroids", "bronchodilators", "leukotriene modifiers"],
        "lifestyle": ["avoid triggers", "use air purifiers", "maintain healthy weight"]
    }
}

def check_symptoms(symptoms_text):
    entities = medical_ner(symptoms_text)
    possible_conditions = []
    
    for condition, details in MEDICAL_CONDITIONS.items():
        score = 0
        for symptom in details["symptoms"]:
            if symptom.lower() in symptoms_text.lower():
                score += 1
        if score > 0:
            possible_conditions.append({
                "condition": condition,
                "confidence": score / len(details["symptoms"]),
                "matching_symptoms": score
            })
    
    return sorted(possible_conditions, key=lambda x: x["confidence"], reverse=True)

# Patient Login Component
def patient_login_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.subheader("🔐 Patient Portal Login")
    st.info("Login to view your personalized health alerts and recommendations")
    
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔓 Login", use_container_width=True):
                is_valid, patient_id = verify_patient(username, password)
                if is_valid:
                    st.session_state.patient_logged_in = True
                    st.session_state.patient_username = username
                    st.session_state.patient_id = patient_id
                    
                    # Add sample data if this is first login
                    add_sample_data(patient_id)
                    
                    st.success(f"Welcome back, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password")
        
        with col2:
            if st.button("Back to Dashboard", use_container_width=True):
                st.rerun()
    
    with tab2:
        st.write("**Create Your Patient Account**")
        new_username = st.text_input("Choose Username", key="new_username")
        new_password = st.text_input("Choose Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
        with col2:
            phone = st.text_input("Phone Number")
        
        if st.button("✅ Create Patient Account", use_container_width=True):
            if not new_username or not new_password or not full_name:
                st.error("❌ Username, password, and name are required")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            elif len(new_password) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                try:
                    patient_id = create_patient_account(new_username, new_password, full_name, email, phone)
                    st.success("✅ Account created successfully! Please login.")
                except sqlite3.IntegrityError:
                    st.error("❌ Username already exists")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Dashboard Page
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", 247)
    
    with col2:
        st.metric("Today's Appointments", 15)
    
    with col3:
        st.metric("Active Prescriptions", 89)
    
    with col4:
        st.metric("Pending Reports", 7)
    
    st.subheader("📊 Healthcare Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        patient_counts = np.random.poisson(3, len(dates))
        
        fig = px.line(x=dates, y=patient_counts, title="Daily Consultations Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Other']
        counts = [25, 30, 15, 20]
        
        fig = px.pie(values=counts, names=conditions, title="Common Conditions Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Login to your patient account to view personalized health alerts and recommendations")

# My Recommendations Page (REQUIRES LOGIN)
elif page == "📈 My Recommendations":
    if not st.session_state.patient_logged_in:
        patient_login_page()
    else:
        st.subheader(f"📈 Health Recommendations for {st.session_state.patient_username}")
        
        recommendations = get_patient_recommendations(st.session_state.patient_id)
        
        if recommendations:
            # Filter options
            col1, col2 = st.columns([2, 1])
            with col1:
                filter_type = st.multiselect("Filter by Type:", 
                                            ["Lifestyle", "Diet", "Medication", "Prevention"],
                                            default=["Lifestyle", "Diet", "Medication", "Prevention"])
            with col2:
                filter_priority = st.selectbox("Priority:", ["All", "High", "Medium", "Low"])
            
            st.write("---")
            
            for rec in recommendations:
                rec_id, patient_id, date, rec_type, title, description, priority, status = rec
                
                # Apply filters
                if rec_type not in filter_type:
                    continue
                if filter_priority != "All" and priority != filter_priority:
                    continue
                
                priority_class = f"{priority.lower()}-priority"
                
                st.markdown(f'''
                <div class="recommendation-card {priority_class}">
                    <h4>🎯 {title}</h4>
                    <p><b>Type:</b> {rec_type} | <b>Priority:</b> {priority} | <b>Status:</b> {status}</p>
                    <p>{description}</p>
                    <small>Date: {date[:10]}</small>
                </div>
                ''', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button(f"✅ Complete", key=f"complete_{rec_id}"):
                        st.success("Recommendation marked as complete!")
                with col2:
                    if st.button(f"📝 Note", key=f"note_{rec_id}"):
                        st.info("Note feature coming soon!")
                
                st.write("")
        else:
            st.info("No recommendations yet. Check back after your next consultation.")

# My Health Alerts Page (REQUIRES LOGIN)
elif page == "🔔 My Health Alerts":
    if not st.session_state.patient_logged_in:
        patient_login_page()
    else:
        st.subheader(f"🔔 Health Alerts for {st.session_state.patient_username}")
        
        tab1, tab2 = st.tabs(["Active Alerts", "Alert History"])
        
        with tab1:
            alerts = get_patient_alerts(st.session_state.patient_id)
            
            if alerts:
                active_alerts = [a for a in alerts if a[6] in ['Pending', 'Active']]
                
                for alert in active_alerts:
                    alert_id, patient_id, date, alert_type, message, priority, status, scheduled_time = alert
                    
                    priority_color = "red" if priority == "High" else "orange" if priority == "Medium" else "blue"
                    
                    st.markdown(f'''
                    <div class="alert-card">
                        <b>{alert_type}</b> - <span style="color:{priority_color}">{priority} Priority</span><br>
                        {message}<br>
                        <small>⏰ Scheduled: {scheduled_time} | Status: {status}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("✓ Dismiss", key=f"dismiss_{alert_id}"):
                            st.success("Alert dismissed!")
                    
                    st.write("")
            else:
                st.info("No active alerts at this time.")
        
        with tab2:
            st.write("**Alert History:**")
            if alerts:
                history_data = []
                for alert in alerts:
                    history_data.append({
                        'Date': alert[2][:10],
                        'Type': alert[3],
                        'Message': alert[4],
                        'Priority': alert[5],
                        'Status': alert[6]
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No alert history available.")

# Document Analysis
elif page == "📄 Document Analysis":
    st.subheader("📄 Medical Document Analysis")
    
    uploaded_doc = st.file_uploader("Upload Medical Document (PDF)", type="pdf")
    
    if uploaded_doc:
        with fitz.open(stream=uploaded_doc.read(), filetype="pdf") as doc:
            doc_text = ""
            for page in doc:
                doc_text += page.get_text()
        
        st.success("📘 Document processed successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Document Preview:**")
            st.text_area("Document Content", doc_text[:1000] + "...", height=300)
        
        with col2:
            entities = medical_ner(doc_text)
            st.write("**Extracted Medical Information:**")
            
            for entity in entities[:10]:
                st.write(f"- **{entity['entity_group']}**: {entity['word']}")
            
            if st.button("Generate Summary"):
                summary = summarizer(doc_text[:1024], max_length=150, min_length=50)
                st.write("**Document Summary:**")
                st.info(summary[0]['summary_text'])

# Medical Imaging
elif page == "🖼️ Medical Imaging":
    st.subheader("🖼️ Medical Image Analysis")
    
    uploaded_img = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        image = Image.open(uploaded_img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Medical Image", use_container_width=True)
        
        with col2:
            img_array = np.array(image)
            
            st.write("**Image Properties:**")
            st.write(f"- Dimensions: {img_array.shape}")
            st.write(f"- Mean Intensity: {np.mean(img_array):.2f}")
            st.write(f"- Standard Deviation: {np.std(img_array):.2f}")
            
            mean_intensity = np.mean(img_array)
            
            if mean_intensity < 80:
                classification = "High Density Area Detected"
                confidence = "Medium"
            elif mean_intensity > 180:
                classification = "Low Density Area Detected" 
                confidence = "Medium"
            else:
                classification = "Normal Density Range"
                confidence = "High"
            
            st.markdown(f'<div class="alert-card"><b>Analysis Result:</b><br>{classification}<br><b>Confidence:</b> {confidence}</div>', unsafe_allow_html=True)
            
            if st.button("Enhance Image"):
                enhanced = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                st.image(enhanced, caption="Enhanced Image", use_container_width=True)

# Voice Assistant
elif page == "🎤 Voice Assistant":
    st.subheader("🎤 Voice Medical Assistant")
    
    voice_file = st.file_uploader("Upload Voice Recording", type=["mp3", "wav", "m4a"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if voice_file:
            st.audio(voice_file)
            
            if st.button("Transcribe Audio"):
                with open("temp_audio.mp3", "wb") as f:
                    f.write(voice_file.read())
                
                result = speech_model.transcribe("temp_audio.mp3")
                transcribed_text = result["text"]
                
                st.write("**Transcription:**")
                st.info(transcribed_text)
                
                sentiment = sentiment_analyzer(transcribed_text)[0]
                st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
                
                st.session_state['transcribed_query'] = transcribed_text
    
    with col2:
        if 'transcribed_query' in st.session_state:
            query = st.session_state['transcribed_query']
            
            st.write("**Analysis Options:**")
            
            if st.button("Check Symptoms"):
                symptoms_result = check_symptoms(query)
                
                if symptoms_result:
                    st.write("**Possible Conditions:**")
                    for condition in symptoms_result[:3]:
                        st.write(f"- {condition['condition'].title()}: {condition['confidence']:.0%} match")
                else:
                    st.write("No specific medical conditions identified.")

# AI Diagnosis Assistant
elif page == "🤖 AI Diagnosis Assistant":
    st.subheader("🤖 AI-Powered Diagnosis Assistant")
    
    st.markdown('<div class="alert-card"><b>Disclaimer:</b> This tool is for informational purposes only and should not replace professional medical consultation.</div>', unsafe_allow_html=True)
    
    symptoms = st.text_area("Describe your symptoms in detail:", height=100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.selectbox("How long have you had these symptoms?", 
                               ["Less than 1 day", "1-3 days", "1 week", "2+ weeks", "1+ months"])
        severity = st.slider("Rate symptom severity (1-10):", 1, 10, 5)
    
    with col2:
        age_group = st.selectbox("Age Group:", ["Child (0-12)", "Teen (13-17)", "Adult (18-64)", "Senior (65+)"])
        existing_conditions = st.text_input("Any existing medical conditions?")
    
    if st.button("Analyze Symptoms"):
        if symptoms:
            possible_conditions = check_symptoms(symptoms)
            
            st.write("**AI Analysis Results:**")
            
            if possible_conditions:
                for i, condition in enumerate(possible_conditions[:3]):
                    confidence_color = "green" if condition['confidence'] > 0.7 else "orange" if condition['confidence'] > 0.4 else "red"
                    
                    st.markdown(f'<div class="feature-card">'
                              f'<h4>{i+1}. {condition["condition"].title()}</h4>'
                              f'<p><b>Confidence:</b> <span style="color:{confidence_color}">{condition["confidence"]:.0%}</span></p>'
                              f'<p><b>Matching Symptoms:</b> {condition["matching_symptoms"]}</p>'
                              f'</div>', unsafe_allow_html=True)
                    
                    if condition["condition"] in MEDICAL_CONDITIONS:
                        condition_info = MEDICAL_CONDITIONS[condition["condition"]]
                        st.write(f"**Recommended Actions:**")
                        st.write("- " + "\n- ".join(condition_info["treatments"]))
            else:
                st.info("Unable to identify specific conditions. Please consult a healthcare provider.")
            
            st.write("**General Recommendations:**")
            if severity >= 8:
                st.error("⚠️ High severity - Seek immediate medical attention!")
            elif severity >= 5:
                st.warning("⚠️ Moderate symptoms - Consider consulting a healthcare provider.")
            else:
                st.info("💡 Mild symptoms - Monitor and maintain general health practices.")

# Other sections with placeholder content
elif page in ["📋 Consultations", "📊 Health Analytics", "💊 Medication Tracker", "📅 Appointments", "📚 Medical Knowledge Base", "⚙️ Settings"]:
    st.subheader(f"{page}")
    st.info("This section is available for all users. Content coming soon!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏥 AI Medical Assistant Pro - Advanced Healthcare Management System</p>
        <p>⚠️ This application is for informational purposes only. Always consult healthcare professionals for medical decisions.</p>
    </div>
    """, 
    unsafe_allow_html=True
)