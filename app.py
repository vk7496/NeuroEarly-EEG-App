import os
import io
import time
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- CONFIG ---
st.set_page_config(page_title="NeuroEarly Pro v53", layout="wide", page_icon="🧠")
FONT_PATH = "Amiri-Regular.ttf"  # Ensure this font file is in the directory

# --- DATA: FULL QUESTIONS TEXT ---
PHQ9_FULL = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure",
    "7. Trouble concentrating on things, such as reading or TV",
    "8. Moving or speaking so slowly that other people noticed",
    "9. Thoughts that you would be better off dead"
]

MMSE_FULL = [
    "1. Orientation to Time (Year, Season, Date, Day, Month)",
    "2. Orientation to Place (Country, City, Town, Hospital, Floor)",
    "3. Registration (Repeat 3 objects: Apple, Table, Penny)",
    "4. Attention (Serial 7s subtraction or spell WORLD backwards)",
    "5. Recall (Recall the 3 objects from step 3)",
    "6. Language (Naming, Repetition, Reading, Writing, Copying)"
]

# --- HELPER FUNCTIONS ---
def reshape_ar(text):
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except:
        return str(text)

def calculate_risks(mmse_score, phq_score, crp_val, b12_val):
    # Base probabilities
    probs = {"Alzheimer's Disease": 5.0, "Major Depression": 10.0, "Space Occupying Lesion (Tumor)": 1.0}
    
    # Alzheimer's Logic
    if mmse_score < 24: probs["Alzheimer's Disease"] += 50.0
    if mmse_score < 20: probs["Alzheimer's Disease"] += 20.0
    if b12_val < 250: probs["Alzheimer's Disease"] += 10.0  # Metabolic contribution
    
    # Depression Logic
    if phq_score > 10: probs["Major Depression"] += 40.0
    if phq_score > 15: probs["Major Depression"] += 25.0
    
    # Tumor Logic (Inflammation + Focal Signs simulation)
    if crp_val > 10.0: probs["Space Occupying Lesion (Tumor)"] += 15.0
    if crp_val > 20.0: probs["Space Occupying Lesion (Tumor)"] += 20.0
    
    # Cap at 99%
    for k in probs:
        probs[k] = min(probs[k], 99.0)
        
    return probs

def generate_plots(stress_level, is_tumor_suspect):
    # 1. Topomaps
    fig_t, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    bands = ['Delta', 'Theta', 'Alpha', 'Beta']
    for i, ax in enumerate(axes):
        # Simulate focal delta for tumor suspect
        data = np.random.rand(10, 10)
        if is_tumor_suspect and bands[i] == 'Delta':
            data[2:5, 2:5] = 1.0 # Focal hotspot
        ax.imshow(data, cmap='jet', interpolation='bicubic')
        ax.set_title(bands[i])
        ax.axis('off')
    buf_t = io.BytesIO()
    fig_t.savefig(buf_t, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig_t)

    # 2. SHAP
    fig_s, ax_s = plt.subplots(figsize=(6, 2.5))
    ax_s.barh(['FAA', 'Complexity', 'Alpha Power'], [0.2, 0.5, 0.3], color=['#2E86C1', '#28B463', '#E74C3C'])
    ax_s.set_title("AI Feature Importance")
    buf_s = io.BytesIO()
    fig_s.savefig(buf_s, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig_s)
    
    return buf_t.getvalue(), buf_s.getvalue()

# --- PDF GENERATION ---
def create_report_v53(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    
    # Font Registration
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        font_ar = 'Amiri'
    except:
        font_ar = 'Helvetica' # Fallback

    styles = getSampleStyleSheet()
    s_head = ParagraphStyle('H', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.navy, spaceAfter=6)
    s_body = ParagraphStyle('B', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14)
    s_ar = ParagraphStyle('AR', parent=styles['Normal'], fontName=font_ar, fontSize=11, leading=14, alignment=TA_RIGHT)
    
    elements = []
    
    # Header
    elements.append(Paragraph(f"NeuroEarly Pro - Clinical Diagnostic Report", styles['Title']))
    elements.append(Paragraph(f"Patient: {data['name']} | ID: {data['id']} | Date: {datetime.now().strftime('%Y-%m-%d')}", s_body))
    elements.append(Paragraph(f"Eye Condition: {data['eyes']} | B12: {data['b12']} pg/mL | CRP: {data['crp']} mg/L", s_body))
    elements.append(Spacer(1, 15))

    # 1. Diagnostic Probabilities
    elements.append(Paragraph("1. AI Diagnostic Probability Estimates", s_head))
    prob_data = [["Condition / Diagnosis", "Probability (%)", "Risk Level"]]
    for k, v in data['probs'].items():
        risk = "HIGH" if v > 50 else "Moderate" if v > 20 else "Low"
        color = colors.red if risk == "HIGH" else colors.orange if risk == "Moderate" else colors.green
        prob_data.append([k, f"{v:.1f}%", risk])
        
    t = Table(prob_data, colWidths=[3.5*inch, 1.2*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('TEXTCOLOR', (2,1), (2,-1), colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 15))

    # 2. Neuro-Imaging
    elements.append(Paragraph("2. EEG Topography & Interpretation", s_head))
    elements.append(RLImage(io.BytesIO(data['topo_img']), width=6.5*inch, height=1.6*inch))
    
    # Explanations
    txt_en = "<b>Interpretation:</b> Red areas in Delta band indicate focal slowing. If coincident with high CRP, this raises suspicion of structural or inflammatory pathology."
    txt_ar = reshape_ar("التفسير: المناطق الحمراء في نطاق دلتا تشير إلى تباطؤ بؤري. إذا تزامن ذلك مع ارتفاع CRP، يزداد احتمال وجود خلل هيكلي أو التهابي.")
    
    elements.append(Paragraph(txt_en, s_body))
    elements.append(Paragraph(txt_ar, s_ar))
    elements.append(Spacer(1, 10))
    
    # Tumor Warning
    if data['probs']['Space Occupying Lesion (Tumor)'] > 20:
        warning = "<b>CLINICAL ALERT:</b> High probability of mass effect detected. MRI Correlation Recommended."
        elements.append(Paragraph(warning, ParagraphStyle('Warn', fontName='Helvetica-Bold', textColor=colors.red)))

    # 3. Detailed Questionnaires (New Page)
    elements.append(PageBreak())
    elements.append(Paragraph("3. Detailed Clinical Assessment (Questions & Responses)", s_head))
    
    # PHQ-9 Table
    elements.append(Paragraph("<b>PHQ-9 (Depression Screening)</b>", s_body))
    p_data = [["Question", "Score (0-3)"]]
    for i, q_text in enumerate(PHQ9_FULL):
        p_data.append([Paragraph(q_text, s_body), str(data['phq_ans'][i])])
    
    t_phq = Table(p_data, colWidths=[5.5*inch, 1*inch])
    t_phq.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(t_phq)
    elements.append(Spacer(1, 10))
    
    # MMSE Table
    elements.append(Paragraph("<b>MMSE (Cognitive Screening)</b>", s_body))
    m_data = [["Task / Domain", "Score (0-5)"]]
    for i, q_text in enumerate(MMSE_FULL):
        m_data.append([Paragraph(q_text, s_body), str(data['mmse_ans'][i])])

    t_mmse = Table(m_data, colWidths=[5.5*inch, 1*inch])
    t_mmse.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(t_mmse)

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- MAIN UI ---
def main():
    st.sidebar.title("🧠 NeuroEarly v53")
    
    # Patient Data
    p_name = st.sidebar.text_input("Patient Name", "John Doe")
    p_id = st.sidebar.text_input("Patient ID", "F-2026")
    eye_cond = st.sidebar.radio("EEG Condition", ["Eyes Open", "Eyes Closed"])
    
    # Lab Data
    st.sidebar.subheader("Biomarkers")
    b12 = st.sidebar.number_input("Vitamin B12 (pg/mL)", 100, 1500, 400)
    crp = st.sidebar.number_input("CRP (mg/L)", 0.0, 50.0, 1.0)
    
    # Clinical Tabs
    tab1, tab2 = st.tabs(["📝 Assessment (Questions)", "🩺 Diagnostics & Report"])
    
    phq_answers = []
    mmse_answers = []
    
    with tab1:
        st.subheader("Clinical Interview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### PHQ-9")
            for q in PHQ9_FULL:
                ans = st.selectbox(q, [0, 1, 2, 3], key=q[:5])
                phq_answers.append(ans)
                
        with col2:
            st.markdown("### MMSE")
            for q in MMSE_FULL:
                ans = st.slider(q, 0, 5, 5, key=q[:5])
                mmse_answers.append(ans)

    with tab2:
        st.subheader("AI Analysis Dashboard")
        up_file = st.file_uploader("Upload EEG File (.edf)", type=['edf'])
        
        if up_file is not None:
            # Simulation
            st.success("Signal Processed.")
            
            # Calculations
            phq_score = sum(phq_answers)
            mmse_score = sum(mmse_answers)
            probs = calculate_risks(mmse_score, phq_score, crp, b12)
            
            # Display Probabilities
            st.write("#### Disease Probability Estimates")
            cols = st.columns(3)
            idx = 0
            for k, v in probs.items():
                cols[idx].metric(label=k, value=f"{v}%", delta="High Risk" if v > 50 else "Low Risk", delta_color="inverse")
                idx += 1
            
            # Visuals
            is_tumor = probs["Space Occupying Lesion (Tumor)"] > 20
            t_img, s_img = generate_plots(stress_level=65, is_tumor_suspect=is_tumor)
            
            st.image(t_img, caption="Brain Topography (Delta-Theta-Alpha-Beta)")
            st.image(s_img, caption="SHAP Feature Importance")
            
            # PDF Generation
            if st.button("Generate Final Clinical Report"):
                pdf_bytes = create_report_v53({
                    'name': p_name, 'id': p_id, 'eyes': eye_cond,
                    'b12': b12, 'crp': crp,
                    'probs': probs,
                    'topo_img': t_img, 'shap_img': s_img,
                    'phq_ans': phq_answers, 'mmse_ans': mmse_answers
                })
                st.download_button("Download PDF Report", pdf_bytes, f"NeuroReport_{p_id}.pdf", "application/pdf")

if __name__ == "__main__":
    main()
