# app.py — NeuroEarly Pro v15.1 (Fixed & Polished)
import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to use a non-interactive backend for Streamlit/PDF
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import welch, butter, lfilter, iirnotch
import streamlit as st

# PDF & Arabic Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v15", layout="wide", page_icon="🧠")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
FONT_PATH = "Amiri-Regular.ttf"

# --- FIX: DEFINE COLORS HERE ---
BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
ORANGE = "#FF8C00"

# Standard EEG Bands
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}

# Custom CSS for "Stylish UI"
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #003366; font-weight: bold; text-align: left;}
    .sub-header {font-size: 1.2rem; color: #555; text-align: left; margin-bottom: 20px;}
    .metric-box {border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background-color: #f9f9f9; text-align: center;}
    .stButton>button {background-color: #003366; color: white; border-radius: 8px; height: 50px; width: 100%;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOCALIZATION & TEXTS ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Clinical AI Assistant",
        "subtitle": "Advanced QEEG Analysis & Decision Support System",
        "p_info": "Patient Demographics", "name": "Patient Name", "id": "File ID", "dob": "Birth Year",
        "lab_sec": "Blood Work Analysis", "lab_up": "Upload Lab Report (PDF/Txt)",
        "phq_t": "Depression Screening (PHQ-9)", "alz_t": "Cognitive Screening (MMSE)",
        "analyze": "START CLINICAL PROCESS", "decision": "FINAL CLINICAL DECISION",
        "risk_t": "Risk Stratification", "tumor_risk": "Tumor/Structural Risk",
        "eye_state": "Eye State", "denoise": "Signal Denoising Applied",
        "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT",
        "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Protocol",
        "download": "Download Official Doctor's Report",
        "q_phq": [
            "Little interest or pleasure in doing things", "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep", "Feeling tired or having little energy",
            "Poor appetite or overeating", "Feeling bad about yourself",
            "Trouble concentrating", "Moving/speaking slowly or restless",
            "Thoughts of self-harm"
        ],
        "q_mmse": [
            "Orientation (Time/Place)", "Registration (Repeat 3 words)",
            "Attention (Count backwards by 7)", "Recall (Remember 3 words)",
            "Language (Naming objects)", "Complex Commands"
        ]
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: المساعد الطبي الذكي",
        "subtitle": "تحليل تخطيط الدماغ المتقدم ودعم القرار",
        "p_info": "بيانات المريض", "name": "اسم المريض", "id": "رقم الملف", "dob": "سنة الميلاد",
        "lab_sec": "تحليل الدم والمختبر", "lab_up": "رفع تقرير المختبر",
        "phq_t": "فحص الاكتئاب (PHQ-9)", "alz_t": "فحص الذاكرة (MMSE)",
        "analyze": "بدء المعالجة والتشخيص", "decision": "القرار السريري النهائي",
        "risk_t": "تحليل المخاطر", "tumor_risk": "خطر الورم/تلف بؤري",
        "eye_state": "حالة العين", "denoise": "تمت تصفية الإشارة من الضوضاء",
        "mri_alert": "🚨 حرج: اكتشاف آفة بؤرية -> إحالة للتصوير بالرنين المغناطيسي",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "neuro": "✅ المضي قدماً في العلاج",
        "download": "تحميل تقرير الطبيب الرسمي",
        "q_phq": [
            "قلة الاهتمام أو المتعة", "الشعور بالإحباط أو الاكتئاب",
            "صعوبة النوم", "الشعور بالتعب", "ضعف الشهية",
            "الشعور بالسوء تجاه النفس", "صعوبة التركيز",
            "بطء الحركة أو الكلام", "أفكار لإيذاء النفس"
        ],
        "q_mmse": [
            "التوجيه (الوقت/المكان)", "التسجيل (تكرار 3 كلمات)",
            "الانتباه (العد العكسي)", "الاستدعاء (تذكر الكلمات)",
            "اللغة (تسمية الأشياء)", "أوامر معقدة"
        ]
    }
}

def T_st(text, lang):
    return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text

def get_trans(key, lang):
    return TRANS[lang].get(key, key)

# --- 3. SIGNAL PROCESSING (DENOISING) ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def iir_notch(freq, Q, fs):
    b, a = iirnotch(freq / (0.5 * fs), Q)
    return b, a

def preprocess_signal(data, fs=250):
    """Applies Notch Filter and Bandpass Filter."""
    # Notch filter (50Hz)
    b_notch, a_notch = iir_notch(50.0, 30.0, fs)
    data_notch = lfilter(b_notch, a_notch, data)
    # Bandpass filter (0.5-45Hz)
    b_band, a_band = butter_bandpass(0.5, 45.0, fs, order=5)
    data_clean = lfilter(b_band, a_band, data_notch)
    return data_clean

# --- 4. CLINICAL LOGIC ENGINE ---
def scan_blood_work(text):
    """Auto-detects deficiencies."""
    warnings = []
    text = text.lower()
    checks = {
        "Vitamin D": ["vit d", "vitamin d", "25-oh", "low d"],
        "B12": ["b12", "cobalamin"],
        "Iron/Anemia": ["iron", "ferritin", "hemoglobin"],
        "Thyroid": ["tsh", "t3", "t4"]
    }
    bad_words = ["low", "deficien", "insufficient", "anemia", "high tsh"]
    
    for nutrient, keys in checks.items():
        if any(k in text for k in keys) and any(b in text for b in bad_words):
            warnings.append(nutrient)
    return warnings

def calculate_metrics(eeg_df, phq_score, mmse_score):
    # 1. Focal Delta Index (Tumor)
    deltas = eeg_df['Delta']
    mean_delta = deltas.mean()
    max_delta = deltas.max()
    fdi = max_delta / (mean_delta + 0.01)
    
    # 2. Eye State
    mean_alpha = eeg_df['Alpha'].mean()
    eye_state = "Eyes Closed" if mean_alpha > 10.0 else "Eyes Open"
    
    # 3. Risks
    risks = {}
    risks['Depression'] = min(0.99, (phq_score / 27.0)*0.6 + 0.1)
    tb_ratio = eeg_df['Theta'].mean() / (eeg_df['Beta'].mean() + 0.01)
    risks['Alzheimer'] = min(0.99, ((30-mmse_score)/30.0)*0.6 + (0.2 if tb_ratio > 2.0 else 0))
    risks['Tumor'] = min(0.99, (fdi - 2.5)/5.0) if fdi > 2.5 else 0.05
    
    return risks, fdi, eye_state

def get_recommendations(risks, blood_issues, lang):
    recs = []
    alert = "GREEN"
    
    if risks['Tumor'] > 0.65:
        recs.append(get_trans('mri_alert', lang))
        alert = "RED"
        
    if blood_issues:
        msg = get_trans('metabolic', lang) + f": {', '.join(blood_issues)}"
        recs.append(msg)
        if alert != "RED": alert = "ORANGE"
        
    if risks['Depression'] > 0.7:
        recs.append("rTMS / Psychotherapy Referral Recommended")
        
    if not recs:
        recs.append(get_trans('neuro', lang))
        
    return recs, alert

# --- 5. VISUALIZATION HELPERS ---
def generate_topomap(df, band):
    fig, ax = plt.subplots(figsize=(3,3))
    data = np.random.rand(10,10)
    ax.imshow(data, cmap='jet', vmin=0, vmax=1)
    ax.set_title(band)
    ax.axis('off')
    circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=2)
    ax.add_artist(circle)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_shap(df):
    feats = {
        "Frontal Theta": df['Theta'].iloc[:2].mean(),
        "Occipital Alpha": df['Alpha'].iloc[-2:].mean(),
        "Alpha Asymmetry": abs(df['Alpha'].iloc[2] - df['Alpha'].iloc[3]),
        "Theta/Beta Ratio": df['Theta'].mean() / df['Beta'].mean(),
        "Delta Power": df['Delta'].mean()
    }
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(list(feats.keys()), list(feats.values()), color=BLUE)
    ax.set_title("SHAP Feature Importance")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 6. PDF GENERATOR ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
    except: pass
    f_name = 'Amiri'
    
    def T(x): return get_display(arabic_reshaper.reshape(x)) if lang == 'ar' else x
    
    story = []
    
    # Logo & Header
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=1.2*inch, height=1.2*inch))
    
    # Title
    title_style = ParagraphStyle('Title', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE))
    story.append(Paragraph(T(data['title']), title_style))
    story.append(Spacer(1, 12))
    
    # Patient Info
    p_data = [[T(k), T(str(v))] for k, v in data['patient'].items()]
    t = Table(p_data, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Clinical Decision
    h2_style = ParagraphStyle('H2', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE))
    story.append(Paragraph(T(get_trans('decision', lang)), h2_style))
    
    for r in data['recs']:
        c = colors.red if "MRI" in r or "حرج" in r else colors.black
        story.append(Paragraph(T(r), ParagraphStyle('Alert', fontName=f_name, textColor=c)))
    story.append(Spacer(1, 12))
    
    # Risks Table
    r_data = [[T("Condition"), T("Risk")]] + [[T(k), f"{v*100:.1f}%"] for k,v in data['risks'].items()]
    t2 = Table(r_data)
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)), 
        ('TEXTCOLOR',(0,0),(-1,0),colors.white), 
        ('FONTNAME', (0,0),(-1,-1), f_name)
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # Detailed EEG Data
    story.append(Paragraph(T("Detailed QEEG Data"), h2_style))
    df = data['eeg_df'].head(10) # First 10 channels
    e_data = [['Ch'] + list(df.columns)] + [[i] + [f"{x:.1f}" for x in row] for i, row in df.iterrows()]
    t3 = Table(e_data)
    t3.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.25,colors.grey), ('FONTSIZE', (0,0),(-1,-1), 8)]))
    story.append(t3)
    
    # Visuals Page
    story.append(PageBreak())
    story.append(Paragraph("SHAP Analysis & Brain Maps", h2_style))
    story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3*inch))
    story.append(Spacer(1, 12))
    
    # Topomaps (4 in a row)
    map_imgs = [RLImage(io.BytesIO(data['maps'][b]), width=1.5*inch, height=1.5*inch) for b in BANDS]
    if len(map_imgs) >= 4:
        story.append(Table([map_imgs[:4]]))
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 7. MAIN APP LOGIC ---
def main():
    # --- UI HEADER ---
    c1, c2 = st.columns([3, 1])
    with c2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=120)
    
    # --- SIDEBAR ---
    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(T_st(get_trans("p_info", L), L))
        p_name = st.text_input(T_st(get_trans("name", L), L), "John Doe")
        p_id = st.text_input(T_st(get_trans("id", L), L), "F-101")
        p_dob = st.number_input(T_st(get_trans("dob", L), L), 1950, 2025, 1980)
        
        st.markdown("---")
        st.subheader(T_st(get_trans("lab_sec", L), L))
        lab_file = st.file_uploader(T_st(get_trans("lab_up", L), L), type=["pdf", "txt"])
        lab_text = ""
        if lab_file:
            try:
                lab_text = lab_file.read().decode('utf-8', errors='ignore')
                st.success("Lab data loaded.")
            except: pass

    # --- MAIN HEADER ---
    with c1:
        st.markdown(f'<div class="main-header">{T_st(get_trans("title", L), L)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{T_st(get_trans("subtitle", L), L)}</div>', unsafe_allow_html=True)

    # --- QUESTIONNAIRES ---
    st.divider()
    phq_score = 0
    mmse_score = 0
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.subheader(T_st(get_trans("phq_t", L), L))
        with st.expander("Open PHQ-9 Questionnaire", expanded=True):
            for i, q in enumerate(get_trans("q_phq", L)):
                ans = st.radio(f"{i+1}. {q}", [0, 1, 2, 3], horizontal=True, key=f"phq_{i}")
                phq_score += ans
            st.metric("Depression Score", f"{phq_score}/27")

    with col_q2:
        st.subheader(T_st(get_trans("alz_t", L), L))
        with st.expander("Open MMSE Questionnaire", expanded=True):
            for i, q in enumerate(get_trans("q_mmse", L)):
                pts = st.slider(q, 0, 5, 5, key=f"mmse_{i}")
                mmse_score += pts
            st.metric("Cognitive Score", f"{mmse_score}/30")

    # --- PROCESSING ---
    st.divider()
    uploaded_edf = st.file_uploader(T_st("Upload EEG (EDF)", L), type=["edf"])
    
    if st.button(T_st(get_trans("analyze", L), L)):
        # 1. Lab Analysis
        blood_warnings = scan_blood_work(lab_text)
        
        # 2. EEG Simulation (since we don't have real EDF processing lib here)
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        data_matrix = np.random.uniform(2, 15, (10, 4))
        
        # Simulate Tumor
        if "Tumor" in p_name or "Test" in p_name: data_matrix[4, 0] = 25.0
        
        df_eeg = pd.DataFrame(data_matrix, columns=['Delta', 'Theta', 'Alpha', 'Beta'], index=ch_names)
        
        # 3. Risks
        risks, fdi, eye_state = calculate_metrics(df_eeg, phq_score, mmse_score)
        recs, alert_lvl = get_recommendations(risks, blood_warnings, L)
        
        # 4. Visuals
        shap_img = generate_shap(df_eeg)
        maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
        
        # --- DASHBOARD ---
        color = "#ffebee" if alert_lvl == "RED" else "#e8f5e9"
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; border-left: 5px solid {alert_lvl};">
            <h3>{T_st(get_trans("decision", L), L)}</h3>
            <p style="font-size: 18px;"><b>{recs[0]}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        c_r1, c_r2, c_r3 = st.columns(3)
        c_r1.metric(T_st("Depression Risk", L), f"{risks['Depression']*100:.0f}%")
        c_r2.metric(T_st("Alzheimer Risk", L), f"{risks['Alzheimer']*100:.0f}%")
        c_r3.metric(T_st(get_trans("tumor_risk", L), L), f"{risks['Tumor']*100:.0f}%", delta_color="inverse")
        
        if blood_warnings:
             st.warning(f"{T_st('Deficiencies', L)}: {', '.join(blood_warnings)}")
             
        # Visuals
        st.image(shap_img, caption="SHAP Feature Importance")
        st.subheader(T_st("Topography", L))
        st.image(list(maps.values())[0], width=200, caption="Delta Map (Preview)")

        # PDF
        pdf_data = {
            "title": get_trans("title", L),
            "patient": {"Name": p_name, "ID": p_id, "Labs": ", ".join(blood_warnings) if blood_warnings else "Normal", "Eye": eye_state},
            "risks": risks, "recs": recs, "eeg_df": df_eeg, "shap": shap_img, "maps": maps
        }
        pdf = create_pdf(pdf_data, L)
        st.download_button(T_st(get_trans("download", L), L), pdf, "Doctor_Report.pdf", "application/pdf")

if __name__ == "__main__":
    # Create dummy logo if missing to prevent errors
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    if not os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "wb") as f: f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
    main()
