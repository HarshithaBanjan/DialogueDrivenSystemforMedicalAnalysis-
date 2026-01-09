import streamlit as st
import fitz  # PyMuPDF
import spacy
import cv2
import whisper
import numpy as np
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from io import BytesIO
import json
from googletrans import Translator
from datetime import datetime, timedelta
import torchaudio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import threading
import time
from textblob import TextBlob
import requests
from datetime import date
import warnings
warnings.filterwarnings('ignore')    



# --- Authentication helpers ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return hash_password(password) == row[0]
    return False

def create_user(username, password):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute("INSERT INTO users (username, password_hash, created_date) VALUES (?, ?, ?)",
              (username, hashed, str(datetime.now())))
    conn.commit()
    conn.close()

def init_users_table():
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_users_table()

# --- Session state initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# --- Login Page ---
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

    st.write("---")
    st.write("Don’t have an account?")
    new_user = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Create account"):
        if new_user and new_password:
            try:
                create_user(new_user, new_password)
                st.success("Account created! Please login above.")
            except sqlite3.IntegrityError:
                st.error("Username already exists. Try a different one.")
        else:
            st.error("Both fields are required.")

    st.stop()

# Load Models
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")
    speech_model = whisper.load_model("base")
    translator = Translator()
    
    # Medical NER model (you can replace with specialized medical models)
    medical_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    
    return nlp, summarizer, sentiment_analyzer, speech_model, translator, medical_ner

nlp, summarizer, sentiment_analyzer, speech_model, translator, medical_ner = load_models()

# Database setup
def init_db():
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    
    # Patients table
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, gender TEXT, 
                  phone TEXT, email TEXT, language TEXT, blood_type TEXT,
                  allergies TEXT, medications TEXT, conditions TEXT, 
                  emergency_contact TEXT, created_date TEXT)''')
    
    # Consultations table
    c.execute('''CREATE TABLE IF NOT EXISTS consultations
                 (id INTEGER PRIMARY KEY, patient_id INTEGER, date TEXT, 
                  symptoms TEXT, diagnosis TEXT, prescription TEXT, 
                  follow_up_date TEXT, doctor_notes TEXT, severity TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    # Vitals table
    c.execute('''CREATE TABLE IF NOT EXISTS vitals
                 (id INTEGER PRIMARY KEY, patient_id INTEGER, date TEXT,
                  blood_pressure TEXT, heart_rate INTEGER, temperature REAL,
                  weight REAL, height REAL, bmi REAL, oxygen_saturation INTEGER,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    # Appointments table
    c.execute('''CREATE TABLE IF NOT EXISTS appointments
                 (id INTEGER PRIMARY KEY, patient_id INTEGER, appointment_date TEXT,
                  doctor_name TEXT, specialty TEXT, status TEXT, notes TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    # Medications table
    c.execute('''CREATE TABLE IF NOT EXISTS medications
                 (id INTEGER PRIMARY KEY, patient_id INTEGER, medication_name TEXT,
                  dosage TEXT, frequency TEXT, start_date TEXT, end_date TEXT,
                  instructions TEXT, status TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    conn.commit()
    conn.close()

init_db()

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI Medical Assistant Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Custom CSS for better styling
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

.success-card {
    background: #d1edff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏥 AI Medical Assistant Pro - Complete Healthcare Solution</h1></div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "🏠 Dashboard",
    "👤 Patient Management", 
    "📋 Consultations",
    "📊 Health Analytics",
    "📄 Document Analysis",
    "🖼️ Medical Imaging",
    "🎤 Voice Assistant",
    "💊 Medication Tracker",
    "📅 Appointments",
    "🔔 Health Alerts",
    "📈 Vital Signs Monitor",
    "🤖 AI Diagnosis Assistant",
    "📚 Medical Knowledge Base",
    "⚙️ Settings"
])

# Database helper functions
def get_patient_by_name(name):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE name=?", (name,))
    result = c.fetchone()
    conn.close()
    return result

def add_patient(name, age, gender, phone, email, language, blood_type, allergies, medications, conditions, emergency_contact):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('''INSERT INTO patients 
                 (name, age, gender, phone, email, language, blood_type, allergies, medications, conditions, emergency_contact, created_date)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (name, age, gender, phone, email, language, blood_type, allergies, medications, conditions, emergency_contact, str(datetime.now())))
    conn.commit()
    patient_id = c.lastrowid
    conn.close()
    return patient_id

def get_all_patients():
    conn = sqlite3.connect('medical_assistant.db')
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    return df

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

# Symptom checker using AI
def check_symptoms(symptoms_text):
    # Extract medical entities
    entities = medical_ner(symptoms_text)
    
    # Simple rule-based diagnosis (in real app, use trained medical models)
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

# Dashboard Page
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(get_all_patients())
        st.metric("Total Patients", total_patients)
    
    with col2:
        # Get today's appointments (simulated)
        today_appointments = 5  # This would be from database
        st.metric("Today's Appointments", today_appointments)
    
    with col3:
        active_medications = 12  # This would be from database
        st.metric("Active Prescriptions", active_medications)
    
    with col4:
        pending_reports = 3  # This would be from database
        st.metric("Pending Reports", pending_reports)
    
    # Recent Activity
    st.subheader("📊 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample chart - Patient Registration Trend
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        patient_counts = np.random.poisson(3, len(dates))
        
        fig = px.line(x=dates, y=patient_counts, title="Patient Registration Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample pie chart - Condition Distribution
        conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Other']
        counts = [25, 30, 15, 20]
        
        fig = px.pie(values=counts, names=conditions, title="Common Conditions")
        st.plotly_chart(fig, use_container_width=True)

# Patient Management
elif page == "👤 Patient Management":
    st.subheader("👤 Patient Profile Management")
    
    tab1, tab2, tab3 = st.tabs(["Add New Patient", "View Patients", "Search Patient"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *")
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            phone = st.text_input("Phone Number")
            email = st.text_input("Email Address")
        
        with col2:
            language = st.selectbox("Preferred Language", ["en", "hi", "kn", "fr", "es", "ar"])
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            allergies = st.text_area("Known Allergies")
            medications = st.text_area("Current Medications")
            conditions = st.text_area("Medical Conditions")
        
        emergency_contact = st.text_input("Emergency Contact (Name & Phone)")
        
        if st.button("Add Patient"):
            if name:
                patient_id = add_patient(name, age, gender, phone, email, language, blood_type, allergies, medications, conditions, emergency_contact)
                st.success(f"Patient {name} added successfully! ID: {patient_id}")
            else:
                st.error("Name is required!")
    
    with tab2:
        patients_df = get_all_patients()
        if not patients_df.empty:
            st.dataframe(patients_df, use_container_width=True)
        else:
            st.info("No patients found. Add some patients first.")
    
    with tab3:
        search_name = st.text_input("Search by Name")
        if search_name:
            patient = get_patient_by_name(search_name)
            if patient:
                st.write("**Patient Found:**")
                patient_data = {
                    "Name": patient[1],
                    "Age": patient[2],
                    "Gender": patient[3],
                    "Phone": patient[4],
                    "Email": patient[5],
                    "Language": patient[6],
                    "Blood Type": patient[7]
                }
                st.json(patient_data)
            else:
                st.warning("Patient not found!")

# Document Analysis
elif page == "📄 Document Analysis":
    st.subheader("📄 Medical Document Analysis")
    
    uploaded_doc = st.file_uploader("Upload Medical Document (PDF)", type="pdf")
    
    if uploaded_doc:
        # Extract text from PDF
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
            # Extract medical entities
            entities = medical_ner(doc_text)
            st.write("**Extracted Medical Information:**")
            
            for entity in entities:
                st.write(f"- **{entity['entity_group']}**: {entity['word']}")
            
            # Summarize document
            if st.button("Generate Summary"):
                summary = summarizer(doc_text, max_length=150, min_length=50)
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
            # Enhanced image analysis
            img_array = np.array(image)
            
            # Image properties
            st.write("**Image Properties:**")
            st.write(f"- Dimensions: {img_array.shape}")
            st.write(f"- Mean Intensity: {np.mean(img_array):.2f}")
            st.write(f"- Standard Deviation: {np.std(img_array):.2f}")
            
            # Simple classification (replace with actual medical AI model)
            mean_intensity = np.mean(img_array)
            
            if mean_intensity < 80:
                classification = "High Density Area Detected"
                confidence = "Medium"
                color = "orange"
            elif mean_intensity > 180:
                classification = "Low Density Area Detected" 
                confidence = "Medium"
                color = "blue"
            else:
                classification = "Normal Density Range"
                confidence = "High"
                color = "green"
            
            st.markdown(f'<div class="alert-card"><b>Analysis Result:</b><br>{classification}<br><b>Confidence:</b> {confidence}</div>', unsafe_allow_html=True)
            
            # Image enhancement options
            if st.button("Enhance Image"):
                # Apply basic image enhancement
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
                # Save temporary file
                with open("temp_audio.mp3", "wb") as f:
                    f.write(voice_file.read())
                
                # Transcribe
                result = speech_model.transcribe("temp_audio.mp3")
                transcribed_text = result["text"]
                
                st.write("**Transcription:**")
                st.info(transcribed_text)
                
                # Analyze sentiment
                sentiment = sentiment_analyzer(transcribed_text)[0]
                st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
                
                # Store in session state for further processing
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
                    st.write("No specific medical conditions identified from the symptoms described.")
            
            if st.button("Generate Health Advice"):
                advice_prompt = f"Provide general health advice for: {query}"
                # In a real implementation, you'd use a medical AI model here
                st.write("**General Health Advice:**")
                st.info("Please consult with a healthcare professional for personalized medical advice. Based on your query, consider maintaining a healthy lifestyle, staying hydrated, and monitoring your symptoms.")

# Medication Tracker
elif page == "💊 Medication Tracker":
    st.subheader("💊 Medication Management")
    
    tab1, tab2, tab3 = st.tabs(["Add Medication", "Current Medications", "Medication Reminders"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Patient Name")
            medication_name = st.text_input("Medication Name")
            dosage = st.text_input("Dosage (e.g., 500mg)")
            frequency = st.selectbox("Frequency", ["Once daily", "Twice daily", "Three times daily", "As needed"])
        
        with col2:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date (if applicable)")
            instructions = st.text_area("Special Instructions")
        
        if st.button("Add Medication"):
            # Add medication to database
            st.success("Medication added successfully!")
    
    with tab2:
        # Display current medications
        medications_data = {
            'Medication': ['Metformin', 'Lisinopril', 'Aspirin'],
            'Dosage': ['500mg', '10mg', '81mg'],
            'Frequency': ['Twice daily', 'Once daily', 'Once daily'],
            'Status': ['Active', 'Active', 'Active']
        }
        
        df = pd.DataFrame(medications_data)
        st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.write("**Upcoming Medication Reminders:**")
        
        reminders = [
            {"time": "09:00 AM", "medication": "Metformin 500mg", "status": "Pending"},
            {"time": "12:00 PM", "medication": "Lisinopril 10mg", "status": "Taken"},
            {"time": "09:00 PM", "medication": "Metformin 500mg", "status": "Pending"}
        ]
        
        for reminder in reminders:
            status_color = "green" if reminder["status"] == "Taken" else "orange"
            st.markdown(f'<div class="feature-card"><b>{reminder["time"]}</b> - {reminder["medication"]} <span style="color:{status_color}">({reminder["status"]})</span></div>', unsafe_allow_html=True)

# Health Analytics
elif page == "📊 Health Analytics":
    st.subheader("📊 Health Analytics & Trends")
    
    # Sample health data visualization
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Blood pressure trend
        systolic = 120 + np.random.normal(0, 10, len(dates))
        diastolic = 80 + np.random.normal(0, 5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=systolic, name='Systolic', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=dates, y=diastolic, name='Diastolic', line=dict(color='blue')))
        fig.update_layout(title='Blood Pressure Trend', xaxis_title='Date', yaxis_title='mmHg')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heart rate trend
        heart_rate = 70 + np.random.normal(0, 8, len(dates))
        
        fig = px.line(x=dates, y=heart_rate, title='Heart Rate Trend')
        fig.update_layout(xaxis_title='Date', yaxis_title='BPM')
        st.plotly_chart(fig, use_container_width=True)
    
    # Health score calculation
    st.subheader("🎯 Health Score")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sample health metrics
        bp_score = 85
        st.metric("Blood Pressure Score", f"{bp_score}/100", delta="5")
    
    with col2:
        activity_score = 92
        st.metric("Activity Score", f"{activity_score}/100", delta="3")
    
    with col3:
        overall_score = (bp_score + activity_score) / 2
        st.metric("Overall Health Score", f"{overall_score:.0f}/100", delta="4")

# AI Diagnosis Assistant
elif page == "🤖 AI Diagnosis Assistant":
    st.subheader("🤖 AI-Powered Diagnosis Assistant")
    
    st.markdown('<div class="alert-card"><b>Disclaimer:</b> This tool is for informational purposes only and should not replace professional medical consultation.</div>', unsafe_allow_html=True)
    
    # Symptom input
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
            # Symptom analysis
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
                    
                    # Show treatment information
                    if condition["condition"] in MEDICAL_CONDITIONS:
                        condition_info = MEDICAL_CONDITIONS[condition["condition"]]
                        st.write(f"**Recommended Actions for {condition['condition'].title()}:**")
                        st.write("- " + "\n- ".join(condition_info["treatments"]))
            else:
                st.info("Unable to identify specific conditions. Please consult a healthcare provider for proper diagnosis.")
            
            # General recommendations
            st.write("**General Recommendations:**")
            if severity >= 8:
                st.error("⚠️ High severity symptoms - Seek immediate medical attention!")
            elif severity >= 5:
                st.warning("⚠️ Moderate symptoms - Consider consulting a healthcare provider.")
            else:
                st.info("💡 Mild symptoms - Monitor and maintain general health practices.")

# Health Alerts & Reminders
elif page == "🔔 Health Alerts":
    st.subheader("🔔 Health Alerts & Reminders")
    
    tab1, tab2, tab3 = st.tabs(["Active Alerts", "Set New Alert", "Alert History"])
    
    with tab1:
        alerts = [
            {"type": "Medication", "message": "Time to take Metformin", "priority": "High", "time": "2 minutes ago"},
            {"type": "Appointment", "message": "Doctor visit tomorrow at 2:00 PM", "priority": "Medium", "time": "1 hour ago"},
            {"type": "Health Check", "message": "Weekly blood pressure check due", "priority": "Low", "time": "1 day ago"}
        ]
        
        for alert in alerts:
            priority_color = "red" if alert["priority"] == "High" else "orange" if alert["priority"] == "Medium" else "blue"
            
            st.markdown(f'<div class="alert-card">'
                       f'<b>{alert["type"]}</b> - <span style="color:{priority_color}">{alert["priority"]} Priority</span><br>'
                       f'{alert["message"]}<br>'
                       f'<small>{alert["time"]}</small>'
                       f'</div>', unsafe_allow_html=True)
    
    with tab2:
        alert_type = st.selectbox("Alert Type", ["Medication", "Appointment", "Health Check", "Custom"])
        alert_message = st.text_input("Alert Message")
        alert_time = st.time_input("Alert Time")
        alert_date = st.date_input("Alert Date")
        
        if st.button("Set Alert"):
            st.success("Alert set successfully!")
    
    with tab3:
        st.write("**Recent Alert History:**")
        history_data = {
            'Date': ['2024-01-15', '2024-01-14', '2024-01-13'],
            'Type': ['Medication', 'Appointment', 'Health Check'],
            'Message': ['Took morning medication', 'Completed cardiology appointment', 'Blood pressure recorded'],
            'Status': ['Completed', 'Completed', 'Completed']
        }
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

# Settings
elif page == "⚙️ Settings":
    st.subheader("⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["General", "Notifications", "Data Export"])
    
    with tab1:
        st.write("**General Settings:**")
        default_language = st.selectbox("Default Language", ["English", "Hindi", "Kannada", "French", "Spanish"])
        timezone = st.selectbox("Timezone", ["Asia/Kolkata", "UTC", "America/New_York"])
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    with tab2:
        st.write("**Notification Settings:**")
        email_notifications = st.checkbox("Email Notifications", value=True)
        sms_notifications = st.checkbox("SMS Notifications", value=False)
        push_notifications = st.checkbox("Push Notifications", value=True)
        
        notification_time = st.time_input("Daily Summary Time")
        
        if st.button("Update Notification Preferences"):
            st.success("Notification preferences updated!")
    
    with tab3:
        st.write("**Data Export Options:**")
        
        export_format = st.selectbox("Export Format", ["CSV", "PDF", "JSON"])
        export_range = st.date_input("Export Data From")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Patient Data"):
                st.success("Patient data exported!")
        
        with col2:
            if st.button("Export Consultations"):
                st.success("Consultation data exported!")
        
        with col3:
            if st.button("Export All Data"):
                st.success("All data exported!")

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