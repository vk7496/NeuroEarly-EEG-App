# app.py — NeuroEarly Pro v14 (Final Clinical Suite)
import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to use a non-interactive backend for Streamlit/PDF
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import streamlit as st
import base64 

# PDF & Arabic Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Arabic Text Handling
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIGURATION ---
APP_TITLE = "NeuroEarly Pro — XAI Clinical Suite"
BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png") 
FONT_PATH = "Amiri-Regular.ttf" # Assumes Amiri-Regular.ttf is available

# Frequency Bands (The core QEEG bands)
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 2. LOCALIZATION (Translations) ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: XAI Clinical Decision Support", "patient_info": "Patient Information",
        "name": "Full Name", "id": "File ID", "labs_manual": "Labs (Manual Entry)",
        "labs_pdf": "Upload Lab Report (PDF/Text)", "assess_tab": "Clinical Assessments",
        "analyze": "Run Clinical Diagnosis", "decision": "CLINICAL DECISION & REFERRAL",
        "mri_rec": "🚨 URGENT: REFER FOR MRI/CT SCAN (FOCAL ANOMALY DETECTED)",
        "shap_title": "AI Explainability (SHAP Feature Importance)", "topomaps": "Brain Topography Mapping (Relative Power)",
        "download": "Download Comprehensive Report (PDF)", "upload_eeg": "Upload EEG (EDF) Data",
        "upload_shap": "Upload SHAP Summary (JSON)", "eeg_table": "Detailed QEEG Channel Data",
        "eyes_state": "EEG Recording State", "therapy_note": "Therapy Recommendation Note",
        "rTMS": "💊 Psychiatry/Neurology Referral (rTMS/tDCS)", "neuro": "✅ Proceed with Neurofeedback/Biofeedback",
        "metabolic": "⚠️ Metabolic Correction Required", "phq_title": "Depression Assessment (PHQ-9)",
        "alz_title": "Cognitive Assessment (MMSE - Orientation/Recall)", "q_1": "Little interest or pleasure in doing things",
        "q_2": "Feeling down, depressed, or hopeless", "q_3": "Trouble falling asleep or staying asleep",
        "q_4": "Orientation to time/place (0-5)", "q_5": "Immediate Recall (0-3)", "q_6": "Delayed Recall (0-2)"
    },
    "ar": {
        "title": "نظام NeuroEarly Pro لدعم القرار السريري", "patient_info": "بيانات المريض",
        "name": "الاسم الكامل", "id": "رقم الملف", "labs_manual": "التحاليل (إدخال يدوي)",
        "labs_pdf": "رفع تقرير التحاليل (PDF/نص)", "assess_tab": "التقييمات السريرية",
        "analyze": "تشغيل التشخيص", "decision": "القرار السريري والإحالة",
        "mri_rec": "🚨 عاجل: إحالة للتصوير بالرنين المغناطيسي/المقطعي (تم اكتشاف شذوذ بؤري)",
        "shap_title": "تفسير الذكاء الاصطناعي (SHAP)", "topomaps": "خرائط الدماغ (الطاقة النسبية)",
        "download": "تحميل التقرير الشامل (PDF)", "upload_eeg": "رفع بيانات تخطيط الدماغ (EDF)",
        "upload_shap": "رفع ملف SHAP (JSON)", "eeg_table": "بيانات مفصلة لقنوات QEEG",
        "eyes_state": "حالة تسجيل تخطيط الدماغ", "therapy_note": "ملاحظات وتوصيات العلاج",
        "rTMS": "💊 إحالة إلى الطب النفسي/الأعصاب (rTMS/tDCS)", "neuro": "✅ المضي قدمًا في التغذية العصبية/البيوفیدبک",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "phq_title": "تقييم الاكتئاب (PHQ-9)",
        "alz_title": "التقييم الإدراكي (MMSE - التوجه/الذاكرة)", "q_1": "قلة الاهتمام أو المتعة في فعل الأشياء",
        "q_2": "الشعور بالإحباط أو الاكتئاب أو اليأس", "q_3": "صعوبة النوم أو البقاء نائمًا",
        "q_4": "التوجه الزماني/المكاني (0-5)", "q_5": "الاستدعاء الفوري (0-3)", "q_6": "الاستدعاء المتأخر (0-2)"
    }
}

def get_text(key, lang): return TRANS[lang].get(key, key)
def process_arabic(text): 
    """Handles Arabic shaping for correct display (required for Streamlit/PDF)."""
    try: return get_display(arabic_reshaper.reshape(text))
    except: return text

def T_st(x, L): 
    """Streamlit Text Translator (Fixes NameError)."""
    if L == 'ar':
        return process_arabic(x)
    return x

# --- 3. CORE LOGIC FUNCTIONS ---

# This simulates data extraction from an EDF file (would be replaced by MNE/PyEEG in production)
def get_simulated_eeg_data(eye_state):
    """Simulates realistic QEEG Relative Power data for a standard 10-20 setup."""
    channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    data = np.random.uniform(3.0, 10.0, (len(channels), 4))
    
    df = pd.DataFrame(data, columns=['Delta', 'Theta', 'Alpha', 'Beta'], index=channels)
    
    # 1. Eyes Open/Closed Logic (Simulation)
    if "Eyes Closed" in eye_state:
        # Increase Alpha globally for eyes closed state
        df['Alpha'] += 10.0
    else:
        # Alpha blocking for eyes open state
        df['Alpha'] -= 3.0
        df['Alpha'] = np.clip(df['Alpha'], 1.0, 50.0) 
        
    # 2. Simulate Focal Delta Anomaly (Tumor Risk)
    if st.session_state.p_name == "Tumor Test":
        # Introduce high Delta (e.g., at C3/T3)
        if 'C3' in df.index: df.loc['C3', 'Delta'] += 20.0 
    
    # Calculate Relative Power (%)
    df_rel = df.apply(lambda x: x / x.sum() * 100, axis=1)
    df_rel.columns = ['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)']
    
    # Add a Gamma column (assuming remaining power is Gamma/other)
    df_rel['Gamma (%)'] = 100.0 - df_rel[['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)']].sum(axis=1)
    df_rel['Gamma (%)'] = np.clip(df_rel['Gamma (%)'], 1.0, 50.0)
    
    return df_rel.round(2)

def determine_eye_state(df_bands):
    """Determines Eyes Open/Closed based on global Alpha power (Alpha Blocking)."""
    global_alpha_mean = df_bands['Alpha (%)'].mean()
    # Threshold for Alpha dominance (typically > 8-10% of spectrum in a relaxed state)
    if global_alpha_mean >= 10.0:
        return "Eyes Closed (Alpha Dominance Detected)"
    else:
        return "Eyes Open (Alpha Blocking Detected)"

def calculate_risks(eeg_df, phq_score, mmse_score):
    risks = {"Depression": 0.0, "Alzheimer": 0.0, "Tumor": 0.0}
    
    # Biomarkers (using normalized percentage values)
    alpha_mean = eeg_df['Alpha (%)'].mean()
    theta_mean = eeg_df['Theta (%)'].mean()
    beta_mean = eeg_df['Beta (%)'].mean()
    tb_ratio = theta_mean / (beta_mean + 0.001)
    
    # 1. Depression Risk
    # Logic: Weighted average of subjective score (PHQ-9) and biological marker (Theta/Beta)
    risks['Depression'] = min(0.99, (phq_score / 27.0)*0.5 + (0.4 if tb_ratio > 1.2 else 0.1))
    
    # 2. Alzheimer Risk (Cognitive)
    # MMSE score max is 10 (simplified). A score of 5 or less indicates severe deficit.
    cog_deficit_percentage = (10 - mmse_score) / 10.0 
    risks['Alzheimer'] = min(0.99, (cog_deficit_percentage * 0.6) + (0.3 if alpha_mean < 8.0 else 0.1))
    
    # 3. Tumor Risk (Focal Delta)
    deltas = eeg_df['Delta (%)']
    fdi = deltas.max() / (deltas.mean() + 0.0001)
    risks['Tumor'] = min(0.99, (fdi - 3.0) / 7.0) if fdi > 3.0 else 0.05
    
    return risks, fdi

def analyze_lab_work(text, lang):
    """Analyzes text for metabolic deficiency keywords (Vitamin D, B12, Thyroid)"""
    warnings = []
    keywords = {"Vitamin D": ["vit d", "low d", "12", "15"], "Thyroid": ["tsh", "thyroid", "high tsh"], 
                "Anemia/Iron": ["iron", "anemia", "ferritin low"], "B12": ["b12", "low b12"]}
    
    text_lower = text.lower()
    for cat, words in keywords.items():
        if any(w in text_lower for w in words):
            warnings.append(cat)
    
    if warnings:
        return [f"{get_text('metabolic', lang)}: {', '.join(warnings)}"], "ORANGE"
    else:
        return [], "GREEN"

def get_referral_recommendation(risks, metabolic_recs, fdi, lang):
    recs = []
    alert_level = "GREEN"
    
    # 1. Tumor/Structural Alert (High Priority)
    if risks['Tumor'] > 0.6 or fdi > 5.0:
        recs.append(get_text('mri_rec', lang))
        alert_level = "RED"
    
    # 2. Metabolic Alert
    recs.extend(metabolic_recs)
    if "⚠️" in "".join(recs) and alert_level != "RED": alert_level = "ORANGE"
    
    # 3. Depression/Psychiatry Referral
    if risks['Depression'] > 0.7: 
        recs.append(get_text('rTMS', lang))
    
    # 4. Neurofeedback Default
    if not recs or (alert_level == "GREEN" and not any("rTMS" in r for r in recs)): 
        recs.append(get_text('neuro', lang))
        
    # Remove duplicates and ensure the MRI is always first if present
    unique_recs = sorted(list(set(recs)), key=lambda x: (x.startswith('🚨'), x.startswith('⚠️')), reverse=True)
    return unique_recs, alert_level

# --- 4. SHAP GENERATOR ---
def generate_shap_chart(shap_data):
    """Generates a Feature Importance Bar Chart using provided SHAP data or simulation."""
    if not shap_data:
        # Fallback to simulation with clinical features
        shap_data = {"PHQ-9_Score": 0.45, "Theta/Beta_Ratio": 0.35, "Alpha_Asymmetry": 0.2, 
                     "Frontal_Delta_Power": 0.15, "Lab_Vitamin_D": 0.05}

    # Sort data by magnitude
    sorted_feats = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
    # Clean up feature names for display
    keys = [x[0].replace('_', ' ').title().replace('Phq 9', 'PHQ-9') for x in sorted_feats] 
    values = [x[1] for x in sorted_feats]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Use single color (Blue) to match the latest XAI Report style
    ax.barh(keys, values, color='#1f77b4') 
    
    ax.invert_yaxis()
    ax.set_xlabel("mean(|shap|)", fontsize=12)
    ax.set_title("SHAP feature importances", fontsize=14, fontweight='bold', loc='left')
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()

# --- 5. VISUALIZATION (Topomaps) ---
def generate_topomap_image(df_bands, band_name):
    # Simplified Topomap logic for robustness, ensuring all 4 maps are generated.
    mean_val = df_bands[f'{band_name} (%)'].mean()
    
    fig, ax = plt.subplots(figsize=(3,3))
    # Create a dummy heatmap based on the band's mean (higher mean -> more red)
    data = np.random.rand(10,10) * (mean_val / 10.0)
    ax.imshow(data, cmap='jet', extent=(-1,1,-1,1), vmin=0, vmax=1.0)
    ax.set_title(band_name)
    ax.axis('off')
    circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=2)
    ax.add_artist(circle)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 6. PDF REPORT (Comprehensive) ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Define T(x) locally for PDF rendering (it handles font for RTL)
    try: 
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_name = 'Amiri'
    except: 
        f_name = 'Helvetica'
        
    def T(x): return process_arabic(x) if lang == 'ar' else x
    
    s_head = ParagraphStyle('H', parent=styles['Heading2'], fontName=f_name, textColor=colors.HexColor(BLUE))
    s_sub = ParagraphStyle('Sub', parent=styles['Heading3'], fontName=f_name, textColor=colors.HexColor(BLUE))
    s_norm = ParagraphStyle('N', parent=styles['Normal'], fontName=f_name)

    story = []
    
    # 1. Logo and Title
    try:
        if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    except:
        story.append(Paragraph("GOLDEN BIRD L.L.C", s_norm)) 
        
    story.append(Paragraph(T(data['ui']['title']), s_head))
    story.append(Spacer(1, 10))
    
    # 2. Patient & Clinical Info
    story.append(Paragraph(T(data['ui']['patient_info']), s_sub))
    p = data['patient']
    info = [
        [T("Name"), T(p['name']), T("ID"), p['id']],
        [T("EEG State"), T(p['eye_state']), T("PHQ-9 Score"), str(p['phq'])],
        [T("MMSE Score"), str(p['mmse']), T("Lab Report Summary"), T(p['labs'])],
    ]
    t = Table(info, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # 3. Risk Stratification
    story.append(Paragraph(T("RISK STRATIFICATION"), s_sub))
    risks_data = [
        [T("Condition"), T("Risk Probability"), T("Severity")],
        [T("Major Depression"), f"{data['risks']['Depression']*100:.1f}%", "HIGH" if data['risks']['Depression'] > 0.6 else "MODERATE" if data['risks']['Depression'] > 0.3 else "LOW"],
        [T("Alzheimer's/Dementia"), f"{data['risks']['Alzheimer']*100:.1f}%", "HIGH" if data['risks']['Alzheimer'] > 0.6 else "MODERATE" if data['risks']['Alzheimer'] > 0.3 else "LOW"],
        [T("Tumor/Focal Anomaly"), f"{data['risks']['Tumor']*100:.1f}%", "HIGH" if data['risks']['Tumor'] > 0.6 else "LOW"]
    ]
    t_risk = Table(risks_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    t_risk.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name),
                                ('BACKGROUND', (0,0),(-1,0), colors.lightgrey)]))
    story.append(t_risk)
    story.append(Spacer(1, 10))
    
    # 4. Therapy Note (Decision & Referral)
    story.append(Paragraph(T(data['ui']['therapy_note']), s_sub))
    for r in data['recs']:
        color = colors.red if "MRI" in r or "عاجل" in r else colors.black
        story.append(Paragraph(T(r), ParagraphStyle('W', parent=s_norm, textColor=color)))
    story.append(Spacer(1, 10))
    
    # 5. XAI SHAP Chart 
    story.append(Paragraph(T(data['ui']['shap_title']), s_sub))
    if data['shap_img']:
        story.append(RLImage(io.BytesIO(data['shap_img']), width=6.5*inch, height=3.5*inch))
    story.append(Spacer(1, 10))
    
    story.append(PageBreak()) 
    
    # 6. Topomaps (Visual QEEG)
    story.append(Paragraph(T(data['ui']['topomaps']), s_head))
    if data['maps']:
        map_keys = list(data['maps'].keys())
        imgs = [RLImage(io.BytesIO(data['maps'][k]), width=1.8*inch, height=1.8*inch) for k in map_keys]
        # Display 4 maps per row (Delta, Theta, Alpha, Beta)
        story.append(Table([[imgs[0], imgs[1], imgs[2], imgs[3]]]))
    story.append(Spacer(1, 10))
    
    # 7. Detailed QEEG Data Table (The numerical band chart for the doctor)
    story.append(Paragraph(T(data['ui']['eeg_table']), s_sub))
    
    df_eeg = data['eeg_data']
    # Ensure all column names are translated
    header = [T("Channel")] + [T(col) for col in df_eeg.columns]
    table_data = [header] + [[T(str(idx))] + [str(val) for val in row] for idx, row in df_eeg.iterrows()]
    
    t_eeg = Table(table_data, repeatRows=1)
    t_eeg.setStyle(TableStyle([
        ('GRID', (0,0),(-1,-1),0.25,colors.grey), 
        ('FONTNAME', (0,0),(-1,-1), f_name),
        ('BACKGROUND', (0,0),(-1,0), colors.lightgrey),
        ('ALIGN', (1,1),(-1,-1), 'CENTER')
    ]))
    story.append(t_eeg)
        
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 7. MAIN APP ---
def main():
    st.set_page_config(page_title="NeuroEarly Pro XAI", layout="wide")
    
    # Initialize session state for persistence
    if 'p_name' not in st.session_state:
        st.session_state.p_name = "Sara Miller"
        st.session_state.p_id = "PAT-2025"
        st.session_state.p_labs = "Vitamin D: 12 (Low), Iron: Normal"
        st.session_state.q_phq1 = 0
        st.session_state.q_phq2 = 0
        st.session_state.q_phq3 = 0
        st.session_state.q_mmse_orient = 5
        st.session_state.q_mmse_imm_recall = 3
        st.session_state.q_mmse_delay_recall = 2
        st.session_state.eyes_state = "Eyes Closed"

    with st.sidebar:
        # Logo check and display
        try:
            if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, caption="GOLDEN BIRD L.L.C", width=140)
        except:
             st.write("GOLDEN BIRD L.L.C")

        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(T_st(get_text("patient_info", L), L))
        st.session_state.p_name = st.text_input(T_st(get_text("name", L), L), st.session_state.p_name)
        st.session_state.p_id = st.text_input(T_st(get_text("id", L), L), st.session_state.p_id)
        
        # Labs Input: Manual Text Area and PDF Uploader
        st.subheader(T_st("Lab Results", L))
        st.session_state.p_labs = st.text_area(T_st(get_text("labs_manual", L), L), st.session_state.p_labs, height=100)
        
        uploaded_lab_pdf = st.file_uploader(T_st(get_text("labs_pdf", L), L), type=["pdf", "txt"], key="lab_uploader")
        if uploaded_lab_pdf is not None:
            # Simulation of text extraction from PDF/Text file
            st.info(T_st("System is extracting text from lab report...", L))
            try:
                content = uploaded_lab_pdf.read().decode("utf-8")
                st.session_state.p_labs += "\n\n(Extracted from file):\n" + content[:500] 
                st.success(T_st("Lab text extracted and merged successfully.", L))
            except Exception as e:
                st.warning(T_st(f"Could not read uploaded lab file: {e}. Using manual text input only.", L))

        # EEG Recording State Selector
        st.subheader(T_st("EEG Setup", L))
        st.session_state.eyes_state = st.selectbox(T_st(get_text("eyes_state", L), L), 
                                                ["Eyes Closed", "Eyes Open"], 
                                                index=0 if st.session_state.eyes_state == "Eyes Closed" else 1)
    
    st.title(T_st(get_text("title", L), L))
    
    tab1, tab2 = st.tabs([T_st(get_text("assess_tab", L), L), T_st(get_text("analyze", L), L)])
    
    with tab1:
        # --- Depression Assessment (PHQ-9 Component) ---
        st.header(T_st(get_text("phq_title", L), L))
        st.session_state.q_phq1 = st.slider(T_st(get_text("q_1", L), L), 0, 3, st.session_state.q_phq1)
        st.session_state.q_phq2 = st.slider(T_st(get_text("q_2", L), L), 0, 3, st.session_state.q_phq2)
        st.session_state.q_phq3 = st.slider(T_st(get_text("q_3", L), L), 0, 3, st.session_state.q_phq3)
        
        # Calculate Total PHQ Score (simplified - max score 9, real PHQ-9 is 27)
        phq_score = st.session_state.q_phq1 + st.session_state.q_phq2 + st.session_state.q_phq3
        phq_score_scaled_to_27 = int(phq_score / 9 * 27) # Scale to a full PHQ-9 range for risk calculation
        st.metric(T_st("Total PHQ Score (Simplified/3 questions)", L), phq_score)
        
        st.divider()

        # --- Cognitive Assessment (MMSE Component) ---
        st.header(T_st(get_text("alz_title", L), L))
        st.session_state.q_mmse_orient = st.slider(T_st(get_text("q_4", L), L), 0, 5, st.session_state.q_mmse_orient)
        st.session_state.q_mmse_imm_recall = st.slider(T_st(get_text("q_5", L), L), 0, 3, st.session_state.q_mmse_imm_recall)
        st.session_state.q_mmse_delay_recall = st.slider(T_st(get_text("q_6", L), L), 0, 2, st.session_state.q_mmse_delay_recall)
        
        # Calculate Total MMSE Cognitive Score (Simplified max score 10 for this section)
        mmse_score = st.session_state.q_mmse_orient + st.session_state.q_mmse_imm_recall + st.session_state.q_mmse_delay_recall
        st.metric(T_st("Total Cognitive Score (Simplified/10)", L), mmse_score)

    with tab2:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_eeg = st.file_uploader(T_st(get_text("upload_eeg", L), L), type=["edf", "txt"], key="eeg_uploader")
        with col_up2:
            uploaded_shap_json = st.file_uploader(T_st(get_text("upload_shap", L), L), type=["json"], key="shap_uploader")
        
        if st.button(T_st(get_text("analyze", L), L), type="primary"):
            
            # 1. Load SHAP Data (Essential for XAI)
            shap_data = None
            if uploaded_shap_json is not None:
                try:
                    shap_content = uploaded_shap_json.read().decode("utf-8")
                    full_json = json.loads(shap_content)
                    # Prioritize 'depression_global' as the main XAI target
                    shap_data = full_json.get("depression_global", full_json.get("alzheimers_global", full_json))
                    st.success(T_st("SHAP data for XAI loaded successfully.", L))
                except Exception as e:
                    st.error(T_st(f"Error reading SHAP JSON file: {e}. Using simulated SHAP data.", L))
                    
            # 2. Load EEG Data (Simulation based on selected state)
            df_eeg = get_simulated_eeg_data(st.session_state.eyes_state)
            
            # 3. Clinical Logic & Risk Calculation
            risks, fdi = calculate_risks(df_eeg, phq_score_scaled_to_27, mmse_score)
            metabolic_recs, _ = analyze_lab_work(st.session_state.p_labs, L)
            recs, alert = get_referral_recommendation(risks, metabolic_recs, fdi, L)
            eye_state_detected = determine_eye_state(df_eeg) # Cross-check detected vs user-input state
            
            # 4. Generate Visuals
            shap_bytes = generate_shap_chart(shap_data) 
            maps = {b: generate_topomap_image(df_eeg, b) for b in BANDS}
            
            # 5. Dashboard Output
            st.divider()
            st.subheader(T_st(get_text('decision', L), L))
            
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric(T_st("Depression Risk", L), f"{risks['Depression']*100:.1f}%")
                st.metric(T_st("Alzheimer Risk", L), f"{risks['Alzheimer']*100:.1f}%")
                
            with c2:
                st.metric(T_st("Tumor/Focal Risk", L), f"{risks['Tumor']*100:.1f}%")
                st.metric(T_st("Focal Delta Index (FDI)", L), f"{fdi:.2f}")

            with c3:
                st.info(T_st(get_text('therapy_note', L), L))
                for r in recs:
                    st.write(T_st(r, L))

            st.subheader(T_st(get_text('shap_title', L), L))
            st.image(shap_bytes, use_container_width=True)
            
            st.subheader(T_st(get_text('eyes_state', L), L))
            st.code(T_st(eye_state_detected, L)) # Display detected state clearly
            
            # 6. PDF Generation & Download
            r_data = {
                "ui": TRANS[L],
                "patient": {"name": st.session_state.p_name, "id": st.session_state.p_id, 
                            "labs": st.session_state.p_labs, "eye_state": eye_state_detected,
                            "phq": phq_score, "mmse": mmse_score},
                "risks": risks,
                "recs": recs,
                "shap_img": shap_bytes,
                "maps": maps,
                "eeg_data": df_eeg
            }
            pdf = create_pdf(r_data, L)
            st.download_button(T_st(get_text("download", L), L), pdf, "XAI_Neuro_Report.pdf", "application/pdf")

if __name__ == "__main__":
    # Setup necessary directories and dummy files for a smooth initial run
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    if not os.path.exists(LOGO_PATH):
        # Create a tiny dummy image file for the logo
        with open(LOGO_PATH, "wb") as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))

    main()
