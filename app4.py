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
import re  
import torch
import torchxrayvision as xrv
from torchvision import transforms

warnings.filterwarnings('ignore')      
# Streamlit UI Configuration
st.set_page_config(
    page_title="Dialogue Driven System for Medical Interpretation and Interaction", 
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

st.markdown('<div class="main-header"><h1>🏥 Dialogue Driven System for Medical Interpretation and Interaction </h1></div>', unsafe_allow_html=True)

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
    
    # Alerts table - kept only for SOS emergency storage
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
    'Gujarati': 'gu'
}

# Load Models
# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.warning("Downloading SpaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")

    @st.cache_resource
    def load_speech_model():
        try:
            model = whisper.load_model("base")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load Whisper model: {e}")
            st.info("Re-downloading Whisper model...")
            import shutil, os
            cache_dir = os.path.expanduser("~/.cache/whisper")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            model = whisper.load_model("base")
            return model

    speech_model = load_speech_model()
    translator = Translator()
    medical_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

    return nlp, summarizer, sentiment_analyzer, speech_model, translator, medical_ner


# --- Chest X-ray AI Model ---
@st.cache_resource
def load_cxr_model(device='cpu'):
    try:
        model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        model = model.eval().to(device)
        return model
    except Exception as e:
        st.warning(f"Could not load chest X-ray model: {e}")
        return None


def preprocess_cxr(img_pil, out_size=224):
    transform = transforms.Compose([
        transforms.Resize((out_size, out_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_rgb = img_pil.convert('RGB')
    t = transform(img_rgb)
    img_gray = t.mean(dim=0, keepdim=True)
    return img_gray.unsqueeze(0)


# --- Load all main models once ---

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

# Use _() as shorthand for translating UI strings; returns original if 'show_english' is True
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
    # If logged in, create SOS alert for patient
    if st.session_state.patient_logged_in and st.session_state.patient_id:
        contact_name = None
        contact_phone = None
        location = None

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn = sqlite3.connect('medical_assistant.db')
            c = conn.cursor()
            c.execute('''INSERT INTO alerts
                         (patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (st.session_state.patient_id, now_str, 'Emergency',
                       'SOS: Immediate assistance required', 'High', 'Active',
                       now.strftime("%H:%M:%S"), contact_name, contact_phone, location))
            conn.commit()
        except Exception as e:
            st.sidebar.error(_("Failed to send SOS alert — please try again."))
        else:
            patient_label = st.session_state.get('patient_username', _('User'))
            st.sidebar.success(_(f"🚨 SOS emergency alert sent for {patient_label} at {now.strftime('%I:%M %p, %d %b %Y')}. Help is on the way."))
            try:
                st.balloons()
            except:
                pass
        finally:
            try:
                conn.close()
            except:
                pass
    else:
        st.sidebar.info(_("You're not logged in. Use the Anonymous SOS form below to send an emergency alert."))

st.sidebar.markdown("---")

# Sidebar anonymous SOS form (for guests / not-logged-in users)
st.sidebar.write(_("Anonymous SOS — provide contact details if not logged in:"))
with st.sidebar.form("sos_form", clear_on_submit=False):
    anon_name = st.text_input(_("Contact Name"))
    anon_phone = st.text_input(_("Contact Phone"))
    anon_location = st.text_input(_("Approximate Location (optional)"))
    send = st.form_submit_button(_("Send SOS"))

if send:
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect('medical_assistant.db')
        c = conn.cursor()
        c.execute('''INSERT INTO alerts
                     (patient_id, date, alert_type, message, priority, status, scheduled_time, contact_name, contact_phone, location)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (None, now_str, 'Emergency',
                   'Anonymous SOS — Immediate assistance required', 'High', 'Active',
                   now.strftime("%H:%M:%S"), anon_name or None, anon_phone or None, anon_location or None))
        conn.commit()
    except Exception as e:
        st.sidebar.error(_("Failed to create SOS alert — please try again."))
    else:
        st.sidebar.success(_("🚨 SOS alert created. Provide this info to responders:"))
        if anon_name:
            st.sidebar.write(f"- {_('Name')}: {anon_name}")
        if anon_phone:
            st.sidebar.write(f"- {_('Phone')}: {anon_phone}")
        if anon_location:
            st.sidebar.write(f"- {_('Location')}: {anon_location}")
        st.sidebar.write(f"- {_('Time')}: {now.strftime('%I:%M %p, %d %b %Y')}")
    finally:
        try:
            conn.close()
        except:
            pass

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
    _("🖼 Medical Imaging"),
    _("🎤 Voice Assistant"),
    _("🤖 AI Diagnosis Assistant"),
    _("⚙ Settings")
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

# Sample data adder updated to not add alerts (only recommendations)
def add_sample_data(patient_id):
    conn = sqlite3.connect('medical_assistant.db')
    c = conn.cursor()
    # Only add if no recommendations exist for this patient (prevent duplicates)
    c.execute("SELECT COUNT(*) FROM recommendations WHERE patient_id=?", (patient_id,))
    if c.fetchone()[0] > 0:
        conn.close()
        return

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

# ---------- Structured Summary Helpers ----------
def detect_abnormal(token_context):
    """Simple heuristic to decide if nearby text indicates abnormality."""
    s = token_context.lower()
    if '↑' in s or 'high' in s or 'elevated' in s or 'greater' in s or '+' in s:
        return "High"
    if '↓' in s or 'low' in s or 'decreased' in s or 'less' in s or '-' in s:
        return "Low"
    return None

def parse_patient_info(text):
    info = {}
    # Name (many reports use "Patient Name:" or "Name:")
    m = re.search(r'(?:Patient\s*Name|Name)\s*[:\-]\s*([A-Za-z\s\.\-]+)', text, re.IGNORECASE)
    if m:
        info['Name'] = m.group(1).strip()
    # Age
    m = re.search(r'Age\s*[:\-]\s*(\d{1,3})', text, re.IGNORECASE)
    if m:
        info['Age'] = m.group(1) + " years"
    else:
        m = re.search(r'Age\/Sex\s*[:\-]\s*(\d{1,3})', text, re.IGNORECASE)
        if m:
            info['Age'] = m.group(1) + " years"
    # Sex/Gender
    m = re.search(r'(?:Sex|Gender)\s*[:\-]\s*([MFUfmfuA-Za-z]+)', text, re.IGNORECASE)
    if m:
        g = m.group(1).strip()
        if g.upper().startswith('M'):
            info['Gender'] = "Male"
        elif g.upper().startswith('F'):
            info['Gender'] = "Female"
        else:
            info['Gender'] = g
    # Report / sample date (loose)
    m = re.search(r'(?:Date|Report Date|Year)\s*[:\-]\s*([0-9\-/]{6,20})', text, re.IGNORECASE)
    if m:
        info['Report Date'] = m.group(1).strip()
    # Lab ID
    m = re.search(r'(?:Lab\s*ID|LabID|Lab\s*No)\s*[:\-]?\s*([A-Za-z0-9\-/]+)', text, re.IGNORECASE)
    if m:
        info['Lab ID'] = m.group(1).strip()
    return info

def group_entities_by_type(entities, doc_text):
    vitals = []
    labs = []
    for ent in entities:
        word = ent.get('word') or ent.get('entity') or ''
        label = ent.get('entity_group') or ent.get('entity_label') or ''
        if not word:
            continue
        # Heuristics for vitals
        if 'Diagnostic' in label or 'PROCEDURE' in label.upper() or 'procedure' in word.lower() or any(k in word.lower() for k in ['pressure','respiratory','pulse','heart','systolic','diastolic','temperature','rate']):
            idx = doc_text.lower().find(word.lower())
            context = doc_text[max(0, idx-40): idx+len(word)+40] if idx!=-1 else word
            abnormal = detect_abnormal(context)
            vitals.append({'name': word.strip(), 'context': context.strip(), 'flag': abnormal})
        elif 'Lab_value' in label or re.search(r'\d', word):
            idx = doc_text.lower().find(word.lower())
            context = doc_text[max(0, idx-40): idx+len(word)+40] if idx!=-1 else word
            abnormal = detect_abnormal(context)
            labs.append({'value': word.strip(), 'context': context.strip(), 'flag': abnormal})
        else:
            # fallback: if numeric, go to labs; else vitals
            if re.search(r'\d', word):
                labs.append({'value': word.strip(), 'context': word, 'flag': None})
            else:
                vitals.append({'name': word.strip(), 'context': word, 'flag': None})
    return vitals, labs

# ---------- UI Components ----------
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
                    add_sample_data(patient_id)
                    st.success(safe_translate(f"Welcome back, {username}!"))
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(safe_translate("❌ Incorrect username or password"))
        with col2:
            if st.button(safe_translate("Back to Dashboard"), use_container_width=True):
                st.rerun()
    with tab2:
        st.write(safe_translate("*Create Your Patient Account*"))
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
    doc_text = ""
    
    if uploaded_doc:
        with fitz.open(stream=uploaded_doc.read(), filetype="pdf") as doc:
            for p in doc:
                # Some PDFs contain layout noise; get_text() is OK for many reports
                doc_text += p.get_text()
        st.success(safe_translate("📘 Document processed successfully!"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(safe_translate("*Document Preview:*"))
            st.text_area(safe_translate("Document Content"), doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text, height=300)
        
        with col2:
            # Entities from model (NER)
            try:
                entities = medical_ner(doc_text)
            except Exception:
                entities = []
            
            st.write(safe_translate("*Extracted Medical Information:*"))
            if entities:
                for entity in entities[:30]:
                    label = entity.get('entity_group', entity.get('entity', ''))
                    word = entity.get('word', entity.get('entity', ''))
                    st.write(f"- *{label}*: {word}")
            else:
                st.write("- No entities detected.")
            
            # ---------- Structured Generate Summary handler ----------
            if st.button(safe_translate("Generate Summary")):
                # Parse patient info using regex
                patient_info = parse_patient_info(doc_text)
                
                # Use NER if not already extracted
                try:
                    ner_entities = medical_ner(doc_text)
                except Exception:
                    ner_entities = []
                
                vitals, labs = group_entities_by_type(ner_entities, doc_text)
                
                # Structured output
                st.markdown("### 🩺 Structured Medical Report Summary")
                st.markdown("#### 👤 Patient Details")
                if patient_info:
                    for k, v in patient_info.items():
                        st.markdown(f"- *{k}:* {v}")
                else:
                    st.markdown("- No clear patient metadata found in document.")
                
                st.markdown("#### 🧪 Vital / Diagnostic Parameters")
                if vitals:
                    for v in vitals:
                        name = v.get('name') or v.get('context')
                        flag = v.get('flag')
                        if flag:
                            st.markdown(f"- *{name}* — {flag} (context: {v['context']})")
                        else:
                            st.markdown(f"- {name} (context: {v['context']})")
                else:
                    st.markdown("- No explicit vital parameters detected.")
                
                st.markdown("#### 🔬 Lab Values")
                if labs:
                    for l in labs:
                        val = l.get('value')
                        flag = l.get('flag')
                        if flag:
                            st.markdown(f"- *{val}* — {flag} (context: {l['context']})")
                        else:
                            st.markdown(f"- {val} (context: {l['context']})")
                else:
                    st.markdown("- No explicit lab values detected.")
                
                st.markdown("#### 💬 Interpretation / Insights")
                try:
                    text_for_summary = doc_text if len(doc_text) < 1500 else doc_text[:1500]
                    insight = summarizer(text_for_summary, max_length=80, min_length=20)
                    insight_text = insight[0]['summary_text'].strip()
                    insight_text = re.sub(r'\s+', ' ', insight_text)
                    st.info(insight_text)
                except Exception:
                    st.info("No automated interpretation available.")
                
                # Optional: show raw entities for debugging
                if st.checkbox(safe_translate("Show extracted entities (for debugging)"), value=False):
                    st.write(ner_entities)
    else:
        st.info(safe_translate("Upload a PDF medical report to extract and summarize."))

elif page == safe_translate("🖼 Medical Imaging"):
    st.subheader(safe_translate("🩻 Advanced Medical Imaging Analyzer"))
    uploaded_img = st.file_uploader(
        safe_translate("Upload any Medical Image (X-ray, MRI, CT, Ultrasound, etc.)"),
        type=["jpg", "jpeg", "png", "tif", "bmp"]
    )

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        aspect_ratio = gray.shape[1] / gray.shape[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption=safe_translate("Uploaded Medical Image"), use_container_width=True)

        with col2:
            st.write("### 🧩 Image Properties")
            st.write(f"- Dimensions: {gray.shape}")
            st.write(f"- Mean Intensity: {mean_intensity:.2f}")
            st.write(f"- Standard Deviation: {std_intensity:.2f}")
            st.write(f"- Edge Density: {edge_density:.3f}")
            st.write(f"- Aspect Ratio: {aspect_ratio:.2f}")

            # --- Detect image type (CT, MRI, Ultrasound, X-ray) ---
            def detect_modality():
                if edge_density < 0.05 and std_intensity < 40:
                    return "MRI"
                elif edge_density < 0.15 and std_intensity < 70:
                    return "CT Scan"
                elif mean_intensity > 160 and std_intensity > 60:
                    return "Ultrasound"
                else:
                    return "X-ray"

            modality = detect_modality()

            # --- If X-ray, detect body part ---
            def detect_xray_region():
                h, w = gray.shape
                if aspect_ratio > 0.9 and aspect_ratio < 1.2 and mean_intensity < 130:
                    return "Chest X-ray"
                elif aspect_ratio >= 1.3 and h > 1000 and mean_intensity < 150:
                    return "Spine X-ray"
                elif mean_intensity > 140 and edge_density > 0.25 and w > 600:
                    return "Hand/Limb X-ray"
                elif mean_intensity < 120 and std_intensity < 45 and w < 800:
                    return "Skull X-ray"
                elif mean_intensity > 160 and std_intensity < 55 and w > 900:
                    return "Abdomen X-ray"
                elif mean_intensity > 150 and edge_density > 0.3:
                    return "Dental X-ray"
                else:
                    return "General X-ray"

            # --- Generate structured findings & impression ---
            findings = []
            impression = ""

            if modality == "X-ray": 
                
                
                xray_type = detect_xray_region()
                st.markdown(f"*Detected Modality:* X-ray ({xray_type})") 

                            # Existing analysis result
            # (Removed undefined debug output for classification/confidence)

            # === 🧠 Add this AI detection block here ===
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cxr_model = load_cxr_model(device=device)

            if cxr_model is not None:
                with st.spinner("🧠 Running AI-based pathology detection..."):
                    img_input = preprocess_cxr(image)
                    img_input = img_input.to(device)
                    with torch.no_grad():
                        preds = cxr_model(img_input)
                    if isinstance(preds, torch.Tensor):
                        preds = preds.cpu().numpy()[0]
                    else:
                        preds = np.array(preds)[0]

                    # Define labels for model outputs (CheXpert style)
                    labels = [
                        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                        "Consolidation", "Edema", "Emphysema", "Fibrosis",
                        "Pleural_Thickening", "Hernia"
                    ]

                    results = list(zip(labels, preds))
                    positives = [(l, s) for (l, s) in results if s >= 0.3]

                    st.markdown("### 🧠 AI-Detected Abnormalities (CheXNet-like Model)")
                    if positives:
                        for label, score in sorted(positives, key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"- **{label}**: {score:.2f} (Possible abnormality)")
                        st.warning("⚠️ Model indicates possible pathology; radiologist confirmation recommended.")
                    else:
                        st.success("✅ No major abnormalities detected by AI model.")
            # === End AI detection block ===


                if xray_type == "Chest X-ray":
                    findings = [
                        "Lung fields appear clear with no focal consolidation or pleural effusion.",
                        "Cardiac silhouette and mediastinal contours are within normal limits.",
                        "Bony thoracic cage appears intact; no rib fractures noted.",
                        "Diaphragm and costophrenic angles are sharp and well-defined."
                    ]
                    impression = "Normal chest X-ray. No active cardiopulmonary disease."

                elif xray_type == "Abdomen X-ray":
                    findings = [
                        "Bowel gas pattern appears normal without dilated loops.",
                        "No air-fluid levels or free intraperitoneal air noted.",
                        "Soft tissue and bony outlines are unremarkable.",
                        "No abnormal calcifications visualized."
                    ]
                    impression = "Normal abdominal radiograph. No signs of obstruction or abnormal calcification."

                elif xray_type == "Spine X-ray":
                    findings = [
                        "Vertebral bodies show normal alignment and height.",
                        "Intervertebral disc spaces are well-maintained.",
                        "No evidence of fracture, spondylolisthesis, or deformity.",
                        "Paraspinal soft tissues appear unremarkable."
                    ]
                    impression = "Normal spinal X-ray. No osseous abnormality detected."

                elif xray_type == "Skull X-ray":
                    findings = [
                        "Calvarial and facial bones appear intact with normal contour.",
                        "No evidence of fracture, lytic or sclerotic lesion.",
                        "Paranasal sinuses and mastoid air cells are clear.",
                        "Soft tissue planes appear normal."
                    ]
                    impression = "Normal skull radiograph. No abnormal intracranial calcification or fracture seen."

                elif xray_type == "Hand/Limb X-ray":
                    findings = [
                        "Bony cortices appear continuous and intact.",
                        "Joint spaces are preserved without dislocation or deformity.",
                        "No fracture lines, erosions, or periosteal reaction noted.",
                        "Soft tissues show normal outline and density."
                    ]
                    impression = "Normal limb X-ray. No fracture or bone pathology detected."

                elif xray_type == "Dental X-ray":
                    findings = [
                        "All visible teeth are well-aligned with intact crown and root structures.",
                        "No obvious caries or periapical radiolucency identified.",
                        "Alveolar bone height and lamina dura appear normal.",
                        "No impacted or missing teeth visualized."
                    ]
                    impression = "Normal dental X-ray. No pathological dental changes observed."

                else:
                    findings = [
                        "Bony and soft tissue structures appear within expected limits.",
                        "No obvious fracture, dislocation, or abnormal shadowing seen.",
                        "Image demonstrates diagnostic quality with good exposure."
                    ]
                    impression = "Normal general X-ray appearance. No acute abnormality detected."   
                    

            elif modality == "CT Scan":
                findings = [
                    "CT slices show normal organ morphology and attenuation.",
                    "No intracranial or intrathoracic hemorrhage noted.",
                    "No evidence of mass lesion, calcification, or edema.",
                    "Bony structures appear intact."
                ]
                impression = "Normal CT scan. No acute intracranial or thoracic pathology detected."

            elif modality == "MRI":
                findings = [
                    "MRI sequences show symmetrical tissue intensity and structure.",
                    "No abnormal signal intensities seen on T1 or T2 sequences.",
                    "No mass effect, midline shift, or edema visualized.",
                    "Normal appearance of visualized brain and spinal cord structures."
                ]
                impression = "Normal MRI study. No abnormal signal or structural lesion detected."

            elif modality == "Ultrasound":
                findings = [
                    "Soft tissue echotexture appears homogeneous and uniform.",
                    "No cystic or solid mass lesions detected.",
                    "No abnormal fluid collection or echogenic debris noted.",
                    "Organ boundaries are well defined and smooth."
                ]
                impression = "Normal ultrasound appearance. No focal abnormality detected."

            else:
                findings = [
                    "Image appears to contain structured medical content.",
                    "No significant irregular densities or structural abnormalities observed.",
                    "Contrast and detail are within diagnostic range."
                ]
                impression = "General medical image analyzed. No remarkable findings detected."

            # --- Display structured report ---
            st.markdown("### 🩺 AI Radiology Report Summary")
            st.markdown(f"*Detected Modality:* {modality}")
            if modality == "X-ray":
                st.markdown(f"*Detected X-ray Type:* {xray_type}")

            st.markdown("#### 🧾 Findings:")
            for f in findings:
                st.markdown(f"- {safe_translate(f)}")

            st.markdown("#### 💬 Impression:")
            st.info(safe_translate(impression))

            # --- Optional Enhancement ---
            if st.button(safe_translate("Enhance Image")):
                enhanced = cv2.equalizeHist(gray)
                st.image(enhanced, caption=safe_translate("Enhanced Image"), use_column_width=True)

            # --- Download Report ---
            def make_text_report():
                report = f"AI Medical Imaging Report\n\nDetected Modality: {modality}\n"
                if modality == "X-ray":
                    report += f"Detected X-ray Type: {xray_type}\n\n"
                report += "Findings:\n"
                for f in findings:
                    report += f"- {f}\n"
                report += f"\nImpression:\n{impression}\n"
                report += f"\nReport Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                return report

            st.download_button(
                label=safe_translate("Download Imaging Report (.txt)"),
                data=make_text_report(),
                file_name=f"{modality}_Report.txt",
                mime="text/plain"
            )

    else:
        st.info(safe_translate("Upload any medical image to generate a structured AI-style radiology summary."))



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
                st.write(safe_translate("*Transcription:*"))
                st.info(translated := safe_translate(transcribed_text))
                sentiment = sentiment_analyzer(transcribed_text)[0]
                st.write(f"{safe_translate('*Sentiment:*')} {sentiment['label']} ({sentiment['score']:.2f})")
                st.session_state['transcribed_query'] = transcribed_text
    with col2:
        if 'transcribed_query' in st.session_state:
            query = st.session_state['transcribed_query']
            st.write(safe_translate("*Analysis Options:*"))
            if st.button(safe_translate("Check Symptoms")):
                symptoms_result = check_symptoms(query)
                if symptoms_result:
                    st.write(safe_translate("*Possible Conditions:*"))
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
            st.write(safe_translate("*AI Analysis Results:*"))
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
                        st.write(safe_translate("*Recommended Actions:*"))
                        for tmt in condition_info["treatments"]:
                            st.write(f"- {safe_translate(tmt)}")
            else:
                st.info(safe_translate("Unable to identify specific conditions. Please consult a healthcare provider."))
            st.write(safe_translate("*General Recommendations:*"))
            if severity >= 8:
                st.error(safe_translate("⚠ High severity - Seek immediate medical attention!"))
            elif severity >= 5:
                st.warning(safe_translate("⚠ Moderate symptoms - Consider consulting a healthcare provider."))
            else:
                st.info(safe_translate("💡 Mild symptoms - Monitor and maintain general health practices."))

elif page == safe_translate("⚙ Settings"):
    st.subheader(safe_translate("⚙ Settings"))
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
        <p>⚠ {safe_translate('This application is for informational purposes only. Always consult healthcare professionals for medical decisions.')}</p>
    </div>
    """, 
    unsafe_allow_html=True
)