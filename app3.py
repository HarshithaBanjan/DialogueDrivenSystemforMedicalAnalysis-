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
    c.execute("SELECT password_hash, patient_id, language FROM patients WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        ok = hash_password(password) == row[0]
        lang = row[2] if len(row) > 2 else None
        return ok, row[1] if ok else None, lang
    return False, None, None

def create_patient_account(username, password, name, email, phone, language='en'):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute('''INSERT INTO patients 
                 (username, password_hash, name, email, phone, language, created_date)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (username, hashed, name, email, phone, language, str(datetime.now())))
    conn.commit()
    patient_id = c.lastrowid
    conn.close()
    return patient_id

def update_patient_language(patient_id, language):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('UPDATE patients SET language=? WHERE patient_id=?', (language, patient_id))
    conn.commit()
    conn.close()

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
    
    # Alerts table (patient_id nullable for anonymous SOS)
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, 
                  patient_id INTEGER, 
                  date TEXT,
                  alert_type TEXT,
                  message TEXT, 
                  priority TEXT,
                  status TEXT,
                  scheduled_time TEXT,
                  contact_name TEXT,
                  contact_phone TEXT,
                  location TEXT,
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
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'  # default language
if 'show_english' not in st.session_state:
    st.session_state.show_english = False

# Supported languages mapping (display name -> googletrans code)
LANGUAGE_OPTIONS = {
    'English': 'en',
    'Hindi': 'hi',
    'Kannada': 'kn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Malayalam': 'ml',
    'Marathi': 'mr',
    'Bengali': 'bn',
    'Gujarati': 'gu',
    'Urdu': 'ur',
    'Spanish': 'es',
    'French': 'fr'
}

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

# Translation helpers
def translate_text(text, dest_code=None):
    if not text:
        return text
    dest = dest_code if dest_code else st.session_state.get('lang', 'en')
    # If destination is English or translation not needed, return original
    if dest == 'en':
        return text
    try:
        # googletrans may struggle with very short tokens; skip translating short punctuation-only strings
        if len(text.strip()) <= 2:
            return text
        res = translator.translate(text, dest=dest)
        return res.text
    except Exception:
        return text

# Use `_()` as shorthand for translating UI strings; returns original if 'show_english' is True
def _(text):
    if st.session_state.get('show_english'):
        return text
    return translate_text(text)

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI Medical Assistant Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Sidebar: Language controls + SOS quick access
st.sidebar.title(_("🌐 Language & Quick Actions"))
selected_lang_name = st.sidebar.selectbox(_("Select your preferred language:"), list(LANGUAGE_OPTIONS.keys()), index=0)
st.session_state.lang = LANGUAGE_OPTIONS[selected_lang_name]
st.session_state.show_english = st.sidebar.checkbox(_("Show original English (no translation)"), value=False)

st.sidebar.markdown("---")
# SOS quick button (always available)
if st.sidebar.button(_("🚨 SOS — Emergency")):
    # If logged in, create SOS alert for patient; otherwise show a small modal-like form
    if st.session_state.patient_logged_in and st.session_state.patient_id:
        conn = sqlite3.connect('medical_assistant.db')
        c = conn.cursor()
        now = str(datetime.now())
        c.execute('''INSERT INTO alerts (patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (st.session_state.patient_id, now, 'Emergency', 'SOS: Immediate assistance required', 'High', 'Active', now, None, None, None))
        conn.commit()
        conn.close()
        st.sidebar.success(_("SOS alert sent. Emergency services notified in the app."))
    else:
        # Collect minimal contact info and optional location
        with st.sidebar.form("sos_form"):
            st.write(_("You're not logged in. Provide contact details to send an emergency alert."))
            contact_name = st.text_input(_("Contact Name"))
            contact_phone = st.text_input(_("Contact Phone"))
            location = st.text_input(_("Approximate Location (optional)"))
            submitted = st.form_submit_button(_("Send SOS"))
            if submitted:
                conn = sqlite3.connect('medical_assistant.db')
                c = conn.cursor()
                now = str(datetime.now())
                c.execute('''INSERT INTO alerts (patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (None, now, 'Emergency', 'Anonymous SOS — Immediate assistance required', 'High', 'Active', now, contact_name, contact_phone, location))
                conn.commit()
                conn.close()
                st.sidebar.success(_("SOS alert created. Provide this info to responders:"))
                st.sidebar.write(f"- { _('Name') }: {contact_name}")
                st.sidebar.write(f"- { _('Phone') }: {contact_phone}")
                if location:
                    st.sidebar.write(f"- { _('Location') }: {location}")

st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.title(_("🧭 Navigation"))
if st.session_state.patient_logged_in:
    st.sidebar.success(f"👤 {st.session_state.patient_username}")
    if st.sidebar.button(_("🚪 Logout")):
        st.session_state.patient_logged_in = False
        st.session_state.patient_username = ""
        st.session_state.patient_id = None
        st.rerun()

page = st.sidebar.selectbox(_("Choose a section:"), [
    _("📄 Document Analysis"),
    _("🖼️ Medical Imaging"),
    _("🎤 Voice Assistant"),
    _("🤖 AI Diagnosis Assistant"),
    _("🔔 My Health Alerts"),
    _("⚙️ Settings")
])

# Helper: safe translate wrapper for texts that may already be translated or are long
def safe_translate(text):
    try:
        return _(text)
    except Exception:
        return text

# Other helpers (unchanged)
def get_patient_recommendations(patient_id):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM recommendations 
                 WHERE patient_id=? 
                 ORDER BY date DESC''', (patient_id,))
    results = c.fetchall()
    conn.close()
    return results

def get_patient_alerts(patient_id=None):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    if patient_id:
        c.execute('''SELECT * FROM alerts 
                     WHERE patient_id=? 
                     ORDER BY date DESC''', (patient_id,))
    else:
        c.execute('''SELECT * FROM alerts ORDER BY date DESC''')
    results = c.fetchall()
    conn.close()
    return results

# Sample data adder remains the same
def add_sample_data(patient_id):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    recommendations = [
        (patient_id, str(datetime.now()), "Lifestyle", "Increase Physical Activity", 
         "Aim for at least 30 minutes of moderate exercise daily", "High", "Active"),
        (patient_id, str(datetime.now() - timedelta(days=1)), "Diet", "Reduce Sodium Intake", 
         "Limit sodium to 2000mg per day for better blood pressure control", "High", "Active"),
    ]
    for rec in recommendations:
        try:
            c.execute('''INSERT INTO recommendations 
                        (patient_id, date, recommendation_type, title, description, priority, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''', rec)
        except:
            pass
    alerts = [
        (patient_id, str(datetime.now()), "Medication", "Time to take Metformin 500mg", 
         "High", "Pending", "09:00 AM", None, None, None),
    ]
    for alert in alerts:
        try:
            c.execute('''INSERT INTO alerts 
                        (patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', alert)
        except:
            pass
    conn.commit()
    conn.close()

# Medical knowledge base (English canonical)
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

# Patient Login Component with full localization
def patient_login_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.subheader(safe_translate("🔐 Patient Portal Login"))
    st.info(safe_translate("Login to view your personalized health alerts and recommendations"))
    tab1, tab2 = st.tabs([safe_translate("Login"), safe_translate("Create Account")])
    with tab1:
        username = st.text_input(safe_translate("Username"), key="login_username")
        password = st.text_input(safe_translate("Password"), type="password", key="login_password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(safe_translate("🔓 Login"), use_container_width=True):
                is_valid, patient_id, lang = verify_patient(username, password)
                if is_valid:
                    st.session_state.patient_logged_in = True
                    st.session_state.patient_username = username
                    st.session_state.patient_id = patient_id
                    # if patient has stored language, use it
                    if lang:
                        st.session_state.lang = lang
                    #add_sample_data(patient_id)
                    st.success(safe_translate(f"Welcome back, {username}!"))
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(safe_translate("❌ Incorrect username or password"))
        with col2:
            if st.button(safe_translate("Back to Dashboard"), use_container_width=True):
                st.rerun()
    with col2:
        pass
    with tab2:
        st.write(safe_translate("**Create Your Patient Account**"))
        new_username = st.text_input(safe_translate("Choose Username"), key="new_username")
        new_password = st.text_input(safe_translate("Choose Password"), type="password", key="new_password")
        confirm_password = st.text_input(safe_translate("Confirm Password"), type="password", key="confirm_password")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input(safe_translate("Full Name"))
            email = st.text_input(safe_translate("Email"))
        with col2:
            phone = st.text_input(safe_translate("Phone Number"))
            create_lang = st.selectbox(safe_translate("Preferred Language"), list(LANGUAGE_OPTIONS.keys()))
        if st.button(safe_translate("✅ Create Patient Account"), use_container_width=True):
            if not new_username or not new_password or not full_name:
                st.error(safe_translate("❌ Username, password, and name are required"))
            elif new_password != confirm_password:
                st.error(safe_translate("❌ Passwords do not match"))
            elif len(new_password) < 6:
                st.error(safe_translate("❌ Password must be at least 6 characters"))
            else:
                try:
                    patient_id = create_patient_account(new_username, new_password, full_name, email, phone, LANGUAGE_OPTIONS[create_lang])
                    st.success(safe_translate("✅ Account created successfully! Please login."))
                except sqlite3.IntegrityError:
                    st.error(safe_translate("❌ Username already exists"))
    st.markdown('</div>', unsafe_allow_html=True)

# Pages with full localization
if page == safe_translate("📄 Document Analysis"):
    st.subheader(safe_translate("📄 Medical Document Analysis"))
    uploaded_doc = st.file_uploader(safe_translate("Upload Medical Document (PDF)"), type="pdf")
    if uploaded_doc:
        with fitz.open(stream=uploaded_doc.read(), filetype="pdf") as doc:
            doc_text = ""
            for p in doc:
                doc_text += p.get_text()
        st.success(safe_translate("📘 Document processed successfully!"))
        col1, col2 = st.columns(2)
        with col1:
            st.write(safe_translate("**Document Preview:**"))
            st.text_area(safe_translate("Document Content"), doc_text[:1000] + "...", height=300)
        with col2:
            entities = medical_ner(doc_text)
            st.write(safe_translate("**Extracted Medical Information:**"))
            for entity in entities[:10]:
                st.write(f"- **{entity['entity_group']}**: {entity['word']}")
            if st.button(safe_translate("Generate Summary")):
                summary = summarizer(doc_text[:1024], max_length=150, min_length=50)
                summary_text = summary[0]['summary_text']
                translated_summary = safe_translate(summary_text)
                st.write(safe_translate("**Document Summary:**"))
                st.info(translated_summary)

elif page == safe_translate("🖼️ Medical Imaging"):
    st.subheader(safe_translate("🖼️ Medical Image Analysis"))
    uploaded_img = st.file_uploader(safe_translate("Upload Medical Image"), type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=safe_translate("Uploaded Medical Image"), use_container_width=True)
        with col2:
            img_array = np.array(image)
            st.write(safe_translate("**Image Properties:**"))
            st.write(f"- {safe_translate('Dimensions')}: {img_array.shape}")
            st.write(f"- {safe_translate('Mean Intensity')}: {np.mean(img_array):.2f}")
            st.write(f"- {safe_translate('Standard Deviation')}: {np.std(img_array):.2f}")
            mean_intensity = np.mean(img_array)
            if mean_intensity < 80:
                classification = safe_translate("High Density Area Detected")
                confidence = safe_translate("Medium")
            elif mean_intensity > 180:
                classification = safe_translate("Low Density Area Detected") 
                confidence = safe_translate("Medium")
            else:
                classification = safe_translate("Normal Density Range")
                confidence = safe_translate("High")
            st.markdown(f'<div class="alert-card"><b>{safe_translate("Analysis Result:")}</b><br>{classification}<br><b>{safe_translate("Confidence:")}</b> {confidence}</div>', unsafe_allow_html=True)
            if st.button(safe_translate("Enhance Image")):
                enhanced = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                st.image(enhanced, caption=safe_translate("Enhanced Image"), use_container_width=True)

elif page == safe_translate("🎤 Voice Assistant"):
    st.subheader(safe_translate("🎤 Voice Medical Assistant"))
    voice_file = st.file_uploader(safe_translate("Upload Voice Recording"), type=["mp3", "wav", "m4a"]) 
    col1, col2 = st.columns(2)
    with col1:
        if voice_file:
            st.audio(voice_file)
            if st.button(safe_translate("Transcribe Audio")):
                with open("temp_audio.mp3", "wb") as f:
                    f.write(voice_file.read())
                result = speech_model.transcribe("temp_audio.mp3")
                transcribed_text = result["text"]
                st.write(safe_translate("**Transcription:**"))
                st.info(translated := safe_translate(transcribed_text))
                sentiment = sentiment_analyzer(transcribed_text)[0]
                st.write(f"{safe_translate('**Sentiment:**')} {sentiment['label']} ({sentiment['score']:.2f})")
                st.session_state['transcribed_query'] = transcribed_text
    with col2:
        if 'transcribed_query' in st.session_state:
            query = st.session_state['transcribed_query']
            st.write(safe_translate("**Analysis Options:**"))
            if st.button(safe_translate("Check Symptoms")):
                symptoms_result = check_symptoms(query)
                if symptoms_result:
                    st.write(safe_translate("**Possible Conditions:**"))
                    for condition in symptoms_result[:3]:
                        st.write(f"- {condition['condition'].title()}: {int(condition['confidence']*100)}% {safe_translate('match')}")
                else:
                    st.write(safe_translate("No specific medical conditions identified."))

elif page == safe_translate("🤖 AI Diagnosis Assistant"):
    st.subheader(safe_translate("🤖 AI-Powered Diagnosis Assistant"))
    st.markdown(f"<div class=\"alert-card\"><b>{safe_translate('Disclaimer:')}</b> {safe_translate('This tool is for informational purposes only and should not replace professional medical consultation.')}</div>", unsafe_allow_html=True)
    symptoms = st.text_area(safe_translate("Describe your symptoms in detail:"), height=100)
    col1, col2 = st.columns(2)
    with col1:
        duration = st.selectbox(safe_translate("How long have you had these symptoms?"), 
                               [safe_translate(x) for x in ["Less than 1 day", "1-3 days", "1 week", "2+ weeks", "1+ months"]])
        severity = st.slider(safe_translate("Rate symptom severity (1-10):"), 1, 10, 5)
    with col2:
        age_group = st.selectbox(safe_translate("Age Group:"), [safe_translate(x) for x in ["Child (0-12)", "Teen (13-17)", "Adult (18-64)", "Senior (65+)"]])
        existing_conditions = st.text_input(safe_translate("Any existing medical conditions?"))
    if st.button(safe_translate("Analyze Symptoms")):
        if symptoms:
            possible_conditions = check_symptoms(symptoms)
            st.write(safe_translate("**AI Analysis Results:**"))
            if possible_conditions:
                for i, condition in enumerate(possible_conditions[:3]):
                    confidence_color = "green" if condition['confidence'] > 0.7 else "orange" if condition['confidence'] > 0.4 else "red"
                    st.markdown(f'<div class="feature-card">'
                              f'<h4>{i+1}. {safe_translate(condition["condition"].title())}</h4>'
                              f'<p><b>{safe_translate("Confidence:")}</b> <span style="color:{confidence_color}">{int(condition["confidence"]*100)}%</span></p>'
                              f'<p><b>{safe_translate("Matching Symptoms:")}</b> {condition["matching_symptoms"]}</p>'
                              f'</div>', unsafe_allow_html=True)
                    if condition["condition"] in MEDICAL_CONDITIONS:
                        condition_info = MEDICAL_CONDITIONS[condition["condition"]]
                        st.write(safe_translate("**Recommended Actions:**"))
                        for tmt in condition_info["treatments"]:
                            st.write(f"- {safe_translate(tmt)}")
            else:
                st.info(safe_translate("Unable to identify specific conditions. Please consult a healthcare provider."))
            st.write(safe_translate("**General Recommendations:**"))
            if severity >= 8:
                st.error(safe_translate("⚠️ High severity - Seek immediate medical attention!"))
            elif severity >= 5:
                st.warning(safe_translate("⚠️ Moderate symptoms - Consider consulting a healthcare provider."))
            else:
                st.info(safe_translate("💡 Mild symptoms - Monitor and maintain general health practices."))

elif page == safe_translate("🔔 My Health Alerts"):
    # Requires login
    if not st.session_state.patient_logged_in:
        patient_login_page()
    else:
        st.subheader(safe_translate("🔔 Health Alerts"))
        alerts = get_patient_alerts(st.session_state.patient_id)
        if alerts:
            for alert in alerts:
                alert_id, patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location = alert
                color = 'red' if priority == 'High' else 'orange' if priority == 'Medium' else 'blue'
                st.markdown(f"<div class='alert-card'><b>{alert_type}</b> - <span style='color:{color}'>{priority} Priority</span><br>{message}<br><small>{safe_translate('Scheduled:')} {scheduled_time}</small></div>", unsafe_allow_html=True)
                cols = st.columns([1,4])
                if st.button(safe_translate('✓ Dismiss'), key=f'dismiss_{alert_id}'):
                    conn = sqlite3.connect('medical_assistant.db')
                    c = conn.cursor()
                    c.execute('UPDATE alerts SET status=? WHERE id=?', ('Dismissed', alert_id))
                    conn.commit()
                    conn.close()
                    st.success(safe_translate('Alert dismissed'))
        else:
            st.info(safe_translate('No alerts available.'))

elif page == safe_translate("⚙️ Settings"):
    st.subheader(safe_translate("⚙️ Settings"))
    st.write(safe_translate("Personalization & Language Settings"))
    if st.session_state.patient_logged_in and st.session_state.patient_id:
        chosen = st.selectbox(safe_translate("Set your account language:"), list(LANGUAGE_OPTIONS.keys()))
        if st.button(safe_translate("Save Language")):
            update_patient_language(st.session_state.patient_id, LANGUAGE_OPTIONS[chosen])
            st.success(safe_translate("Language preference updated."))

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏥 AI Medical Assistant Pro - Advanced Healthcare Management System</p>
        <p>⚠️ {safe_translate('This application is for informational purposes only. Always consult healthcare professionals for medical decisions.')}</p>
    </div>
    """, 
    unsafe_allow_html=True
)
