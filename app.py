# app.py — NeuroEarly Pro v13 (Full Clinical & XAI Suite)
import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import base64 # Used for encoding logo image

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
# Simulating the presence of your logo file (Golden Bird)
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png") 
FONT_PATH = "Amiri-Regular.ttf"

# Frequency Bands
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 2. LOCALIZATION (Updated for new fields) ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: XAI Clinical Decision Support", "patient_info": "Patient Information",
        "name": "Full Name", "id": "File ID", "labs_manual": "Labs (Manual Entry)",
        "labs_pdf": "Upload Lab Report (PDF/Text)", "assess_tab": "Assessments",
        "analyze": "Run Clinical Diagnosis", "decision": "CLINICAL DECISION & REFERRAL",
        "mri_rec": "URGENT: REFER FOR MRI/CT SCAN (FOCAL ANOMALY DETECTED)",
        "shap_title": "AI Explainability (SHAP Feature Importance)", "topomaps": "Brain Topography Mapping (Relative Power)",
        "download": "Download Comprehensive Report (PDF)", "upload_eeg": "Upload EEG (EDF) Data",
        "upload_shap": "Upload SHAP Summary (JSON)", "phq_header": "Depression (PHQ-9 Score 0-27)",
        "alz_header": "Cognitive (MMSE Score 0-10)", "eeg_table": "Detailed QEEG Channel Data",
        "eyes_state": "EEG Recording State", "therapy_note": "Therapy Recommendation Note",
        "rTMS": "💊 Psychiatry/Neurology Referral (rTMS/tDCS)", "neuro": "✅ Proceed with Neurofeedback",
        "metabolic": "⚠️ Metabolic Correction Required"
    },
    "ar": {
        "title": "نظام NeuroEarly Pro لدعم القرار السريري", "patient_info": "بيانات المريض",
        "name": "الاسم الكامل", "id": "رقم الملف", "labs_manual": "التحاليل (إدخال يدوي)",
        "labs_pdf": "رفع تقرير التحاليل (PDF/نص)", "assess_tab": "التقييمات",
        "analyze": "تشغيل التشخيص", "decision": "القرار السريري والإحالة",
        "mri_rec": "عاجل: إحالة للتصوير بالرنين المغناطيسي/المقطعي (تم اكتشاف شذوذ بؤري)",
        "shap_title": "تفسير الذكاء الاصطناعي (SHAP)", "topomaps": "خرائط الدماغ (الطاقة النسبية)",
        "download": "تحميل التقرير الشامل (PDF)", "upload_eeg": "رفع بيانات تخطيط الدماغ (EDF)",
        "upload_shap": "رفع ملف SHAP (JSON)", "phq_header": "الاكتئاب (PHQ-9، من 0-27)",
        "alz_header": "الذاكرة/الإدراك (MMSE، من 0-10)", "eeg_table": "بيانات مفصلة لقنوات QEEG",
        "eyes_state": "حالة تسجيل تخطيط الدماغ", "therapy_note": "ملاحظات وتوصيات العلاج",
        "rTMS": "💊 إحالة إلى الطب النفسي/الأعصاب (rTMS/tDCS)", "neuro": "✅ المضي قدمًا في التغذية العصبية",
        "metabolic": "⚠️ يتطلب تصحيح أيضي"
    }
}

def get_text(key, lang): return TRANS[lang].get(key, key)
def process_arabic(text): 
    try: return get_display(arabic_reshaper.reshape(text))
    except: return text

# --- 3. CORE LOGIC FUNCTIONS ---

def get_simulated_eeg_data():
    """Simulates realistic QEEG Relative Power data for 32 channels."""
    channels = [f'Ch{i}' for i in range(1, 33)] 
    data = np.random.uniform(1.0, 4.0, (32, 4))
    
    # Simulate focal Delta anomaly (Tumor risk test) at a central channel
    if st.session_state.p_name == "Tumor Test":
        data[channels.index('Ch1'), 0] = 15.0 
    
    # Simulate high Alpha for Eyes Closed state (default simulation)
    data[:, 2] += 4.0 
    
    df = pd.DataFrame(data, columns=['Delta_rel', 'Theta_rel', 'Alpha_rel', 'Beta_rel'], index=channels)
    
    # Normalize to 100% (The sum of relative power in bands should be close to 100% of the spectrum)
    df_rel = df.apply(lambda x: x / x.sum() * 100, axis=1)
    df_rel.columns = ['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)']
    
    # Add a 'Gamma (%)' column for full report simulation (Gamma is often residual/noise)
    df_rel['Gamma (%)'] = 100.0 - df_rel[['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)']].sum(axis=1)
    df_rel['Gamma (%)'] = np.clip(df_rel['Gamma (%)'], 1.0, 50.0) # Ensure no negative/too high
    
    return df_rel.round(2)

def calculate_focal_delta_index(df_bands):
    if 'Delta (%)' not in df_bands.columns: return 0.0, None
    deltas = df_bands['Delta (%)']
    # FDI: Focal Delta Index (Max Delta / Mean Global Delta)
    fdi = deltas.max() / (deltas.mean() + 0.0001)
    return fdi, deltas.idxmax()

def determine_eye_state(df_bands):
    """Determines Eyes Open/Closed based on global Alpha power."""
    global_alpha_mean = df_bands['Alpha (%)'].mean()
    # High Alpha (e.g., > 10%) is typical of a relaxed, Eyes Closed state
    if global_alpha_mean >= 10.0:
        return "Eyes Closed (Alpha Dominance Detected)"
    else:
        return "Eyes Open (Minimal Alpha Blocking)"

def calculate_risks(eeg_df, phq_score, alz_score):
    risks = {"Depression": 0.0, "Alzheimer": 0.0, "Tumor": 0.0}
    
    # Biomarkers (using normalized percentage values)
    alpha_mean = eeg_df['Alpha (%)'].mean()
    theta_mean = eeg_df['Theta (%)'].mean()
    beta_mean = eeg_df['Beta (%)'].mean()
    tb_ratio = theta_mean / (beta_mean + 0.001)
    
    # Logic (Tuning for clinical sensitivity)
    risks['Depression'] = min(0.99, (phq_score / 27.0)*0.5 + (0.5 if tb_ratio > 1.2 else 0.1))
    
    # MMSE is 0-10, scale to 0-30 for realistic deficit calculation
    cog_deficit = (30 - (alz_score * 3)) / 30.0 
    risks['Alzheimer'] = min(0.99, (cog_deficit * 0.5) + (0.4 if alpha_mean < 8.0 else 0.1))
    
    fdi, _ = calculate_focal_delta_index(eeg_df)
    risks['Tumor'] = min(0.99, (fdi - 3.0) / 7.0) if fdi > 3.0 else 0.05
    
    return risks, fdi

def analyze_lab_work(text, lang):
    warnings = []
    keywords = {"Vitamin D": ["vit d", "low d", "12", "15"], "Thyroid": ["tsh", "thyroid", "high tsh"], 
                "Anemia": ["iron", "anemia", "ferritin low"], "B12": ["b12", "low b12"]}
    
    text_lower = text.lower()
    for cat, words in keywords.items():
        if any(w in text_lower for w in words):
            warnings.append(cat)
    
    if warnings:
        return [f"{get_text('metabolic', lang)}: {', '.join(warnings)}"], "ORANGE"
    else:
        return [get_text('neuro', lang)], "GREEN"

def get_referral_recommendation(risks, metabolic_recs, fdi, lang):
    recs = []
    alert_level = "GREEN"
    
    # 1. Tumor/Structural Alert
    if risks['Tumor'] > 0.6 or fdi > 5.0:
        recs.append(f"🚨 {get_text('mri_rec', lang)}")
        alert_level = "RED"
    
    # 2. Metabolic Alert (Overwrites Green/Yellow)
    if "⚠️" in metabolic_recs[0]:
        recs.append(metabolic_recs[0])
        if alert_level != "RED": alert_level = "ORANGE"
    
    # 3. Depression/Psychiatry Referral
    if risks['Depression'] > 0.7: 
        recs.append(get_text('rTMS', lang))
    
    # 4. Default
    if not recs or (alert_level == "GREEN" and not any("rTMS" in r for r in recs)): 
        recs.append(get_text('neuro', lang))
        
    return recs, alert_level

# --- 4. SHAP GENERATOR (Bar Chart) ---
def generate_shap_chart(shap_data):
    if not shap_data:
        # Fallback if no real JSON is uploaded
        shap_data = {"frontal_theta_power": 2.8, "occipital_alpha_power": 2.6, 
                     "delta_rel_power": 2.3, "temporal_beta_power": 1.9,
                     "beta_alpha_ratio": 1.2, "theta_alpha_ratio": 0.9}

    # Sort data by magnitude
    sorted_feats = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
    keys = [x[0].replace('_', ' ').title() for x in sorted_feats] # Clean feature names
    values = [x[1] for x in sorted_feats]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Use single color (Blue) to match the latest XAI Report (1).pdf style
    bars = ax.barh(keys, values, color='#1f77b4') 
    
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
    # This simulation is simplified but fulfills the requirement of generating a map per band.
    # In a real setup, this would use libraries like MNE or custom algorithms.
    
    # Use the mean value of the band to color the simulated map.
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
    # Note: Ensure the logo file exists in the 'assets' directory in a production environment
    try:
        if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    except:
        # Fallback if file access fails in the environment
        story.append(Paragraph("GOLDEN BIRD L.L.C", s_norm)) 
        
    story.append(Paragraph(T(data['ui']['title']), s_head))
    story.append(Spacer(1, 10))
    
    # 2. Patient & Clinical Info
    story.append(Paragraph(T(data['ui']['patient_info']), s_sub))
    p = data['patient']
    info = [
        [T("Name"), T(p['name']), T("ID"), p['id']],
        [T(data['ui']['labs_manual']), T(p['labs']), T("EEG State"), T(p['eye_state'])]
    ]
    t = Table(info, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # 3. Risk Stratification
    story.append(Paragraph(T("RISK STRATIFICATION"), s_sub))
    risks_data = [
        [T("Condition"), T("Risk Probability"), T("Severity")],
        [T("Major Depression"), f"{data['risks']['Depression']*100:.1f}%", "MODERATE" if data['risks']['Depression'] > 0.3 else "LOW"],
        [T("Alzheimer's/Dementia"), f"{data['risks']['Alzheimer']*100:.1f}%", "MODERATE" if data['risks']['Alzheimer'] > 0.3 else "LOW"],
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
        color = colors.red if "MRI" in r else colors.black
        story.append(Paragraph(T(r), ParagraphStyle('W', parent=s_norm, textColor=color)))
    story.append(Spacer(1, 10))
    
    # 5. XAI SHAP Chart (Why the result?)
    story.append(Paragraph(T(data['ui']['shap_title']), s_sub))
    if data['shap_img']:
        story.append(RLImage(io.BytesIO(data['shap_img']), width=6.5*inch, height=3.5*inch))
    story.append(Spacer(1, 10))
    
    story.append(PageBreak()) # New page for QEEG details
    
    # 6. Topomaps (Visual QEEG)
    story.append(Paragraph(T(data['ui']['topomaps']), s_head))
    if data['maps']:
        map_keys = list(data['maps'].keys())
        imgs = [RLImage(io.BytesIO(data['maps'][k]), width=1.5*inch, height=1.5*inch) for k in map_keys]
        # Display 4 maps per row (Delta, Theta, Alpha, Beta)
        story.append(Table([[imgs[0], imgs[1], imgs[2], imgs[3]]]))
    story.append(Spacer(1, 10))
    
    # 7. Detailed QEEG Data Table
    story.append(Paragraph(T(data['ui']['eeg_table']), s_sub))
    
    df_eeg = data['eeg_data']
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
        st.session_state.phq = 10
        st.session_state.alz = 8
    
    with st.sidebar:
        # Logo check and display
        try:
            if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, caption="GOLDEN BIRD L.L.C", width=140)
        except:
             st.write("GOLDEN BIRD L.L.C")

        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(get_text("patient_info", L))
        st.session_state.p_name = st.text_input(get_text("name", L), st.session_state.p_name)
        st.session_state.p_id = st.text_input(get_text("id", L), st.session_state.p_id)
        
        # Labs Input: Manual Text Area
        st.session_state.p_labs = st.text_area(get_text("labs_manual", L), st.session_state.p_labs, height=100)
        
        # Labs Input: PDF Upload (Simulation)
        uploaded_lab_pdf = st.file_uploader(get_text("labs_pdf", L), type=["pdf", "txt"], key="lab_uploader")
        if uploaded_lab_pdf is not None:
             # Simulation of text extraction from PDF/Text file
            st.info(T("The system is running text extraction on the lab report..."))
            try:
                content = uploaded_lab_pdf.read().decode("utf-8")
                # Append extracted content to manual text area for processing
                st.session_state.p_labs += "\n\n(Extracted from file):\n" + content[:500] 
                st.success(T("Lab text extracted and merged successfully."))
            except Exception as e:
                st.warning(T(f"Could not read uploaded lab file: {e}. Using manual text input only."))
    
    st.title(get_text("title", L))
    
    tab1, tab2 = st.tabs([get_text("assess_tab", L), get_text("analyze", L)])
    
    with tab1:
        # PHQ-9 (Depression) - 0 to 27
        st.session_state.phq = st.slider(get_text("phq_header", L), 0, 27, st.session_state.phq)
        # MMSE (Cognitive/Alzheimer) - Simplified 0 to 10 scale (10 is perfect)
        st.session_state.alz = st.slider(get_text("alz_header", L), 0, 10, st.session_state.alz)
        
    with tab2:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_eeg = st.file_uploader(get_text("upload_eeg", L), type=["edf", "txt"], key="eeg_uploader")
        with col_up2:
            uploaded_shap_json = st.file_uploader(get_text("upload_shap", L), type=["json"], key="shap_uploader")
        
        if st.button(get_text("analyze", L), type="primary"):
            
            # 1. Load SHAP Data
            shap_data = None
            if uploaded_shap_json is not None:
                try:
                    shap_content = uploaded_shap_json.read().decode("utf-8")
                    # Assuming the JSON contains the feature importance structure (e.g., "depression_global": {...})
                    full_json = json.loads(shap_content)
                    # Use the Depression SHAP data (most relevant for PHQ/EEG features)
                    shap_data = full_json.get("depression_global", full_json) 
                    st.success(T("SHAP data for XAI loaded successfully."))
                except Exception as e:
                    st.error(T(f"Error reading SHAP JSON file: {e}. Using simulated SHAP data."))
                    shap_data = None

            # 2. Load EEG Data (Simulation)
            # In a real app, 'uploaded_eeg' would be processed here (e.g., using MNE library)
            df_eeg = get_simulated_eeg_data()
            
            # 3. Clinical Logic & Risk Calculation
            risks, fdi = calculate_risks(df_eeg, st.session_state.phq, st.session_state.alz)
            metabolic_recs, _ = analyze_lab_work(st.session_state.p_labs, L)
            recs, alert = get_referral_recommendation(risks, metabolic_recs, fdi, L)
            eye_state = determine_eye_state(df_eeg) # NEW: Eyes State Detection
            
            # 4. Generate Visuals
            shap_bytes = generate_shap_chart(shap_data) 
            maps = {b: generate_topomap_image(df_eeg, b) for b in BANDS}
            
            # 5. Dashboard Output
            st.divider()
            st.subheader(T(get_text('decision', L)))
            
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric(T("Depression Risk"), f"{risks['Depression']*100:.1f}%")
                st.metric(T("Alzheimer Risk"), f"{risks['Alzheimer']*100:.1f}%")
                
            with c2:
                st.metric(T("Tumor/Focal Risk"), f"{risks['Tumor']*100:.1f}%")
                st.metric(T("Focal Delta Index (FDI)"), f"{fdi:.2f}")

            with c3:
                st.info(T(get_text('therapy_note', L)))
                for r in recs:
                    st.write(T(r))

            st.subheader(T(get_text('shap_title', L)))
            st.image(shap_bytes, use_container_width=True)
            
            st.subheader(T(get_text('eyes_state', L)))
            st.write(T(eye_state))
            
            # 6. PDF Generation & Download
            r_data = {
                "ui": TRANS[L],
                "patient": {"name": st.session_state.p_name, "id": st.session_state.p_id, 
                            "labs": st.session_state.p_labs, "eye_state": eye_state},
                "risks": risks,
                "recs": recs,
                "shap_img": shap_bytes,
                "maps": maps,
                "eeg_data": df_eeg
            }
            pdf = create_pdf(r_data, L)
            st.download_button(get_text("download", L), pdf, "XAI_Neuro_Report.pdf", "application/pdf")

if __name__ == "__main__":
    # Ensure a dummy logo file exists for successful running of the demo app
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    if not os.path.exists(LOGO_PATH):
        # Create a tiny dummy file for the demo to avoid errors
        with open(LOGO_PATH, "wb") as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))

    # Ensure Amiri font (or a similar one) is available for Arabic PDF rendering
    # This step is dependent on the execution environment; a user running locally must install the font.
    # For a robust solution, consider ReportLab's built-in fonts if custom fonts are unavailable.

    main()
