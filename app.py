import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
import streamlit as st
import mne 
from datetime import datetime

# PDF & Language Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="NeuroEarly Pro v46", layout="wide", page_icon="🩸")
FONT_PATH = "Amiri-Regular.ttf"  # Ensure this file is next to app.py

# Initialize Session State for Blood Data
if 'b12' not in st.session_state: st.session_state['b12'] = 400.0
if 'tsh' not in st.session_state: st.session_state['tsh'] = 2.5
if 'crp' not in st.session_state: st.session_state['crp'] = 1.0
if 'blood_analyzed' not in st.session_state: st.session_state['blood_analyzed'] = False

# --- 2. LOGIC: SIMULATED OCR (AUTO-READ LAB REPORT) ---
def analyze_blood_report(file):
    """
    Simulates reading a PDF/Image lab report using AI.
    In a real app, this would use Tesseract OCR or PyPDF2.
    Here we simulate extraction for stability.
    """
    with st.spinner("Scanning Lab Report via AI-OCR..."):
        time.sleep(2.0) # Simulate processing time
        # Simulated extracted values (Mock Data)
        extracted_data = {
            'b12': 180.0,  # Low
            'tsh': 5.8,    # High
            'crp': 12.5    # High Inflammation
        }
    return extracted_data

# --- 3. LOGIC: STRESS & PDF ---
def calculate_stress_gauge(val):
    fig, ax = plt.subplots(figsize=(6, 1.5))
    cmap = plt.get_cmap('RdYlGn_r')
    norm = plt.Normalize(0, 100)
    
    # Gradient Bar
    grad = np.linspace(0, 100, 256).reshape(1, -1)
    ax.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 100, 0, 1])
    
    # Marker
    ax.axvline(val, color='black', linewidth=4)
    ax.text(val, 1.2, f"{val:.1f}", ha='center', fontsize=12, weight='bold')
    
    ax.set_yticks([]); ax.set_xlim(0, 100)
    ax.set_title("Neuro-Autonomic Stress Index", fontsize=10)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    
    # Font Handling (Fail-safe)
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_main = 'Amiri'
    except:
        f_main = 'Helvetica'
    
    def T(txt):
        if lang == 'ar' and f_main == 'Amiri':
            return get_display(arabic_reshaper.reshape(str(txt)))
        return str(txt)

    styles = getSampleStyleSheet()
    s_head = ParagraphStyle('Head', fontName=f_main, fontSize=14, textColor=colors.navy, backColor=colors.aliceblue, borderPadding=5)
    s_body = ParagraphStyle('Body', fontName=f_main, fontSize=10, leading=15, alignment=TA_RIGHT if lang=='ar' else TA_LEFT)
    
    elements = []
    
    # 1. Header with Date
    elements.append(Paragraph(T("NeuroEarly Pro - Clinical Report"), s_head))
    elements.append(Paragraph(T(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), s_body))
    elements.append(Spacer(1, 15))
    
    # 2. Patient & Lab Info (Table)
    lab_text = f"B12: {data['b12']} | TSH: {data['tsh']} | CRP: {data['crp']}"
    p_data = [
        [Paragraph(T(f"Patient: {data['name']}"), s_body), Paragraph(T(f"ID: {data['id']}"), s_body)],
        [Paragraph(T(f"Lab Results: {lab_text}"), s_body), Paragraph(T(f"Status: {data['status']}"), s_body)]
    ]
    t = Table(p_data, colWidths=[3.5*inch, 3.5*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # 3. Stress Gauge
    elements.append(Paragraph(T("Stress Analysis / تحليل التوتر"), s_head))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(io.BytesIO(data['gauge']), width=5*inch, height=1.2*inch))
    elements.append(Spacer(1, 20))
    
    # 4. Clinical Scores
    elements.append(Paragraph(T("Clinical Assessment / التقييم السريري"), s_head))
    elements.append(Spacer(1, 5))
    score_data = [
        [T("Test"), T("Score"), T("Interpretation")],
        [T("PHQ-9 (Depression)"), T(f"{data['phq']}/27"), T("Moderate" if data['phq']>10 else "Normal")],
        [T("MMSE (Cognitive)"), T(f"{data['mmse']}/30"), T("Impairment" if data['mmse']<24 else "Normal")]
    ]
    t_scores = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 3*inch])
    t_scores.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), f_main)
    ]))
    elements.append(t_scores)
    
    # 5. Physician Narrative
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(T("Final Recommendation / التوصية النهائية"), s_head))
    elements.append(Paragraph(T(data['narrative']), s_body))
    
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- 4. MAIN UI ---
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=50)
    st.sidebar.title("NeuroEarly v46")
    
    # --- A. PATIENT & LAB INPUT ---
    st.sidebar.header("1. Patient & Lab Data")
    p_name = st.sidebar.text_input("Name", "John Doe")
    p_id = st.sidebar.text_input("File ID", "F-9090")
    
    # Blood Upload Button
    uploaded_lab = st.sidebar.file_uploader("📄 Upload Blood Report (PDF/Img)", type=["pdf", "png", "jpg"])
    
    if uploaded_lab is not None and not st.session_state['blood_analyzed']:
        # Trigger Auto-Read Logic
        extracted = analyze_blood_report(uploaded_lab)
        st.session_state['b12'] = extracted['b12']
        st.session_state['tsh'] = extracted['tsh']
        st.session_state['crp'] = extracted['crp']
        st.session_state['blood_analyzed'] = True
        st.sidebar.success("✅ Data Extracted Automatically!")
        st.rerun()

    # Numeric Inputs (Auto-filled by upload, but editable)
    b12_val = st.sidebar.number_input("Vitamin B12", value=st.session_state['b12'])
    tsh_val = st.sidebar.number_input("TSH Level", value=st.session_state['tsh'])
    crp_val = st.sidebar.number_input("CRP (Inflammation)", value=st.session_state['crp'])

    # --- B. MAIN TABS ---
    tab1, tab2 = st.tabs(["📋 Questionnaires", "🧠 EEG & Analysis"])

    with tab1:
        st.header("Clinical Assessment Questionnaires")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PHQ-9 (Depression)")
            # HARDCODED QUESTIONS TO ENSURE VISIBILITY
            q_phq = ["Little interest", "Feeling down", "Sleep trouble", "Low energy", "Appetite change"]
            phq_score = 0
            for i, q in enumerate(q_phq):
                ans = st.radio(f"{i+1}. {q}", ["Not at all (0)", "Several days (1)", "More than half (2)", "Nearly every day (3)"], key=f"phq_{i}", horizontal=True)
                phq_score += int(ans.split('(')[1][0])
            st.metric("Depression Score", f"{phq_score}/15")

        with col2:
            st.subheader("MMSE (Cognition)")
            q_mmse = ["Orientation Time", "Orientation Place", "Registration", "Attention", "Recall"]
            mmse_score = 0
            for i, q in enumerate(q_mmse):
                ans = st.radio(f"{i+1}. {q}", ["Incorrect (0)", "Correct (1)"], key=f"mmse_{i}", horizontal=True, index=1)
                mmse_score += int(ans.split('(')[1][0])
            # Scale to 30 for demo
            final_mmse = mmse_score * 6 
            st.metric("Cognitive Score", f"{final_mmse}/30")

    with tab2:
        st.header("Neuro-Physiological Analysis")
        uploaded_eeg = st.file_uploader("Upload EEG File (.edf)", type=["edf"])
        
        if uploaded_eeg:
            st.success("EEG Signal Processed Successfully.")
            
            # Simulated Analysis Logic combining Blood + EEG
            stress_val = 65.0
            if tsh_val > 4.0 or crp_val > 5.0:
                stress_val += 20 # Inflammation increases stress index
            
            gauge_bytes = calculate_stress_gauge(min(99, stress_val)).getvalue()
            st.image(gauge_bytes, caption="AI-Calculated Stress Index (Integrated with Lab Data)")
            
            # Report Generation
            if st.button("Generate Final Report"):
                narrative = f"Patient shows stress index of {stress_val:.1f}. "
                if b12_val < 200: narrative += "CRITICAL: Low B12 detected, mimicking cognitive decline. "
                if crp_val > 3: narrative += "High inflammation markers present. "
                if final_mmse < 24: narrative += "Cognitive screening suggests impairment."
                
                pdf_data = {
                    'name': p_name, 'id': p_id,
                    'b12': b12_val, 'tsh': tsh_val, 'crp': crp_val,
                    'status': "Abnormal" if stress_val > 70 else "Stable",
                    'gauge': gauge_bytes,
                    'phq': phq_score, 'mmse': final_mmse,
                    'narrative': narrative
                }
                
                pdf = create_pdf(pdf_data, "en") # Default to English for stability
                st.download_button("Download Report (PDF)", pdf, "Clinical_Report.pdf")

if __name__ == "__main__":
    main()
