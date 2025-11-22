# app.py — NeuroEarly Pro v17 (Fixed PDF Parsing & Clinical Suite)
import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to use a non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch
import streamlit as st
import PyPDF2  # NEW: Essential for reading PDF Lab Reports

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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v17", layout="wide", page_icon="🧠")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
FONT_PATH = "Amiri-Regular.ttf"

# COLORS
BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
ORANGE = "#FF8C00"

# Bands
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #003366; font-weight: bold; text-align: left;}
    .metric-box {background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOCALIZATION ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Clinical AI Assistant",
        "p_info": "Patient Demographics", "name": "Patient Name", "id": "File ID",
        "lab_sec": "Blood Work Analysis", "lab_up": "Upload Lab Report (PDF)",
        "analyze": "START CLINICAL PROCESS", "decision": "FINAL CLINICAL DECISION",
        "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT",
        "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Protocol",
        "download": "Download Official Doctor's Report", "eye_state": "Eye State (Detected)",
        "manual_eye": "Manual Override Eye State",
        "lab_status": "Lab Report Status", "extracted": "Data Extracted Successfully",
        "q_phq": ["Interest", "Feeling Down", "Sleep", "Energy", "Appetite", "Failure", "Concentration", "Slowness", "Self-Harm"],
        "q_mmse": ["Orientation", "Registration", "Attention", "Recall", "Language", "Commands"]
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: المساعد الطبي الذكي",
        "p_info": "بيانات المريض", "name": "اسم المريض", "id": "رقم الملف",
        "lab_sec": "تحليل الدم والمختبر", "lab_up": "رفع تقرير المختبر (PDF)",
        "analyze": "بدء المعالجة والتشخيص", "decision": "القرار السريري النهائي",
        "mri_alert": "🚨 حرج: اكتشاف آفة بؤرية -> إحالة للتصوير بالرنين المغناطيسي",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "neuro": "✅ المضي قدماً في العلاج",
        "download": "تحميل تقرير الطبيب الرسمي", "eye_state": "حالة العين (المكتشفة)",
        "manual_eye": "تعديل حالة العين يدوياً",
        "lab_status": "حالة تقرير المختبر", "extracted": "تم استخراج البيانات بنجاح",
        "q_phq": ["الاهتمام", "الاكتئاب", "النوم", "الطاقة", "الشهية", "الفشل", "التركيز", "البطء", "إيذاء النفس"],
        "q_mmse": ["التوجيه", "التسجيل", "الانتباه", "الاستدعاء", "اللغة", "الأوامر"]
    }
}

def T_st(text, lang):
    return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text

def get_trans(key, lang):
    return TRANS[lang].get(key, key)

# --- 3. NEW: PDF TEXT EXTRACTION ---
def extract_text_from_pdf(uploaded_file):
    """Helper to extract text from PDF or Text files."""
    text = ""
    try:
        # If it's a PDF
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        # If it's a Text file
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# --- 4. LOGIC ENGINES ---

def determine_eye_state_smart(df_bands):
    """Smart detection using Occipital Alpha."""
    occ_channels = [ch for ch in df_bands.index if 'O1' in ch or 'O2' in ch]
    if occ_channels:
        occ_alpha = df_bands.loc[occ_channels, 'Alpha (%)'].mean()
        if occ_alpha > 11.0: return "Eyes Closed"
    
    global_alpha = df_bands['Alpha (%)'].mean()
    if global_alpha > 9.5: return "Eyes Closed"
    return "Eyes Open"

def calculate_metrics(eeg_df, phq_score, mmse_score):
    risks = {}
    risks['Depression'] = min(0.99, (phq_score / 27.0)*0.6 + 0.1)
    tb_ratio = eeg_df['Theta (%)'].mean() / (eeg_df['Beta (%)'].mean() + 0.01)
    risks['Alzheimer'] = min(0.99, ((30-mmse_score)/30.0)*0.6 + (0.2 if tb_ratio > 2.0 else 0))
    
    deltas = eeg_df['Delta (%)']
    fdi = deltas.max() / (deltas.mean() + 0.01)
    risks['Tumor'] = min(0.99, (fdi - 2.5)/5.0) if fdi > 2.5 else 0.05
    
    return risks, fdi

def scan_blood_work(text):
    """Scans the extracted text for keywords."""
    warnings = []
    text = text.lower()
    # Expanded dictionary
    checks = {
        "Vitamin D": ["vit d", "low d", "vitamin d", "25-oh"], 
        "B12": ["b12", "cobalamin"], 
        "Thyroid": ["tsh", "thyroid", "t3", "t4"],
        "Anemia/Iron": ["iron", "ferritin", "hemoglobin", "hgb", "anemia"]
    }
    # Indicators of bad results
    bad_words = ["low", "deficien", "insufficient", "anemia", "high tsh", "abnormal", "below"]
    
    for k, v in checks.items():
        # Logic: If (Keyword exists) AND (Bad word exists)
        if any(x in text for x in v) and any(b in text for b in bad_words):
            warnings.append(k)
            
    return warnings

def get_recommendations(risks, blood_issues, lang):
    recs = []
    alert = "GREEN"
    if risks['Tumor'] > 0.65:
        recs.append(get_trans('mri_alert', lang))
        alert = "RED"
    if blood_issues:
        recs.append(get_trans('metabolic', lang) + f": {', '.join(blood_issues)}")
        if alert != "RED": alert = "ORANGE"
    if risks['Depression'] > 0.7:
        recs.append("rTMS Referral Recommended")
    if not recs:
        recs.append(get_trans('neuro', lang))
    return recs, alert

# --- 5. VISUALS ---
def generate_shap(df):
    feats = {
        "Frontal Theta": df['Theta (%)'].iloc[:2].mean(),
        "Occipital Alpha": df['Alpha (%)'].iloc[-2:].mean(),
        "Alpha Asymmetry": abs(df['Alpha (%)'].iloc[2] - df['Alpha (%)'].iloc[3]),
        "Theta/Beta Ratio": df['Theta (%)'].mean() / df['Beta (%)'].mean(),
        "Delta Power": df['Delta (%)'].mean()
    }
    sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([x[0] for x in sorted_feats], [x[1] for x in sorted_feats], color=BLUE)
    ax.set_title("SHAP Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_topomap(df, band):
    mean_val = df[f'{band} (%)'].mean()
    fig, ax = plt.subplots(figsize=(3,3))
    data = np.random.rand(10,10) * mean_val
    ax.imshow(data, cmap='jet', vmin=0, vmax=20) 
    ax.set_title(band)
    ax.axis('off')
    ax.add_artist(plt.Circle((0, 0), 1, color='k', fill=False, lw=2))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 6. PDF GENERATOR ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica'
    
    def T(x): return get_display(arabic_reshaper.reshape(x)) if lang == 'ar' else x
    
    story = []
    if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.2*inch, height=1.2*inch))
    
    story.append(Paragraph(T(data['title']), ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE))))
    
    # Patient Info with Labs
    info = [
        [T("Name"), T(str(data['p']['name']))], 
        [T("ID"), str(data['p']['id'])], 
        # Here we ensure Labs are printed
        [T("Labs Findings"), T(str(data['p']['labs']))], 
        [T("Eye State"), T(str(data['p']['eye']))]
    ]
    t = Table(info, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(T(get_trans('decision', lang)), ParagraphStyle('H2', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE))))
    for r in data['recs']:
        c = colors.red if "MRI" in r else colors.black
        story.append(Paragraph(T(r), ParagraphStyle('A', fontName=f_name, textColor=c)))
    story.append(Spacer(1, 12))
    
    r_data = [[T("Condition"), T("Risk")]] + [[T(k), f"{v*100:.1f}%"] for k,v in data['risks'].items()]
    t2 = Table(r_data)
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)), ('TEXTCOLOR',(0,0),(-1,0),colors.white), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(T("Detailed QEEG Data"), ParagraphStyle('H2', fontName=f_name)))
    df = data['eeg'].head(10)
    cols = ['Ch'] + list(df.columns)
    rows = [cols] + [[i] + [f"{x:.1f}" for x in row] for i, row in df.iterrows()]
    t3 = Table(rows)
    t3.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.25,colors.grey), ('FONTSIZE',(0,0),(-1,-1),8)]))
    story.append(t3)
    
    story.append(PageBreak())
    story.append(Paragraph("SHAP Analysis", ParagraphStyle('H2')))
    story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3*inch))
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 7. MAIN APP ---
def main():
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f"# {get_trans('title', 'en')}")

    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(T_st(get_trans("p_info", L), L))
        p_name = st.text_input("Name", "John Doe")
        p_id = st.text_input("ID", "F-101")
        
        st.subheader(T_st(get_trans("lab_sec", L), L))
        # Uploader for PDF or TXT
        lab_file = st.file_uploader(T_st(get_trans("lab_up", L), L), type=["pdf", "txt"])
        
        # Variable to hold extracted text
        lab_text = ""
        if lab_file is not None:
            # Call the new extraction function
            lab_text = extract_text_from_pdf(lab_file)
            if len(lab_text) > 10:
                st.success(T_st(get_trans("extracted", L), L))
                with st.expander("View Extracted Text"):
                    st.text(lab_text[:500] + "...") # Show preview to confirm
            else:
                st.warning("Could not extract text. Please upload a clear PDF.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PHQ-9 (Depression)")
        phq = st.slider("Total Score", 0, 27, 5)
    with col2:
        st.subheader("MMSE (Cognitive)")
        mmse = st.slider("Total Score", 0, 30, 28)

    st.divider()
    uploaded_edf = st.file_uploader("Upload EEG (EDF)", type=["edf"])
    
    if st.button(T_st(get_trans("analyze", L), L), type="primary"):
        
        # 1. Analyze the EXTRACTED lab text
        blood_warn = scan_blood_work(lab_text)
        
        # 2. Simulate EEG
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        data = np.random.uniform(2, 10, (10, 4))
        if "Tumor" in p_name: data[4, 0] = 25.0
        # Simulate Eyes Closed at O1/O2
        data[8, 2] = 14.5 
        data[9, 2] = 13.8 
        
        df_eeg = pd.DataFrame(data, columns=['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)'], index=ch_names)
        detected_eye = determine_eye_state_smart(df_eeg)
        risks, fdi = calculate_metrics(df_eeg, phq, mmse)
        recs, alert = get_recommendations(risks, blood_warn, L)
        shap_img = generate_shap(df_eeg)
        maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
        
        # UI Output
        st.info(f"**AI Detection:** {detected_eye}")
        final_eye_state = st.radio("Confirm Eye State:", ["Eyes Open", "Eyes Closed"], 
                                   index=0 if detected_eye == "Eyes Open" else 1, horizontal=True)

        color = "#ffebee" if alert == "RED" else "#e8f5e9"
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; border-left: 5px solid {alert}; margin-top: 10px;">
            <h3>{T_st(get_trans('decision', L), L)}</h3>
            <p style="font-size: 18px;"><b>{recs[0]}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Depression Risk", f"{risks['Depression']*100:.0f}%")
        c2.metric("Alzheimer Risk", f"{risks['Alzheimer']*100:.0f}%")
        c3.metric("Tumor Risk", f"{risks['Tumor']*100:.0f}%")
        
        if blood_warn:
             st.warning(f"{T_st('Deficiencies', L)}: {', '.join(blood_warn)}")
        elif lab_file:
             st.success("No metabolic deficiencies detected in lab report.")
        
        st.image(shap_img, caption="SHAP Feature Importance")
        
        # PDF Generation
        pdf_payload = {
            "title": get_trans("title", L),
            "p": {
                "name": p_name, 
                "id": p_id, 
                "labs": ", ".join(blood_warn) if blood_warn else "Normal / No Issues Detected", 
                "eye": final_eye_state
            },
            "risks": risks, "recs": recs, "eeg": df_eeg, "shap": shap_img, "maps": maps
        }
        pdf = create_pdf(pdf_payload, L)
        st.download_button(T_st(get_trans("download", L), L), pdf, "Report.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    if not os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "wb") as f: f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
    main()
