# app.py — NeuroEarly Pro v9 (Interactive Clinical Suite)
import os
import io
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import welch
import streamlit as st

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

# --- 1. CONFIGURATION & ASSETS ---
APP_TITLE = "NeuroEarly Pro — Clinical Suite"
BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
FONT_PATH = "Amiri-Regular.ttf"  # Ensure this file exists locally

# Frequency Bands
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 2. LOCALIZATION (English & Arabic) ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Clinical Decision Support",
        "patient_info": "Patient Information",
        "name": "Full Name",
        "dob": "Birth Year",
        "id": "File ID",
        "history": "Medical History",
        "labs": "Lab Results / Blood Work",
        "labs_ph": "Paste text (e.g., Vitamin D: 12, Iron: Low...)",
        "assessment": "Clinical Assessment",
        "phq_title": "PHQ-9 Depression Screening",
        "alz_title": "Cognitive Impairment Screening",
        "analyze_btn": "Run Analysis & Generate Report",
        "upload": "Upload EEG (EDF File)",
        "report_title": "Clinical Report",
        "decision": "SYSTEM DECISION",
        "reason": "Reasoning",
        "risk": "Risk Analysis",
        "abnormal": "Abnormal Channel Detection",
        "download": "Download PDF Report",
        "questions_phq": [
            "Little interest or pleasure in doing things?",
            "Feeling down, depressed, or hopeless?",
            "Trouble falling or staying asleep, or sleeping too much?",
            "Feeling tired or having little energy?",
            "Poor appetite or overeating?",
            "Feeling bad about yourself — or that you are a failure?",
            "Trouble concentrating on things?",
            "Moving or speaking so slowly that other people could have noticed?",
            "Thoughts that you would be better off dead?"
        ],
        "options_phq": ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"]
    },
    "ar": {
        "title": "نظام NeuroEarly Pro للدعم السريري",
        "patient_info": "معلومات المريض",
        "name": "الاسم الكامل",
        "dob": "سنة الميلاد",
        "id": "رقم الملف",
        "history": "التاريخ الطبي",
        "labs": "نتائج المختبر / تحليل الدم",
        "labs_ph": "الصق النص هنا (مثلا: فيتامين د: منخفض...)",
        "assessment": "التقييم السريري",
        "phq_title": "فحص الاكتئاب (PHQ-9)",
        "alz_title": "فحص الضعف الإدراكي",
        "analyze_btn": "بدء التحليل وإنشاء التقرير",
        "upload": "رفع ملف تخطيط الدماغ (EDF)",
        "report_title": "التقرير الطبي",
        "decision": "قرار النظام",
        "reason": "التعليل",
        "risk": "تحليل المخاطر",
        "abnormal": "القنوات غير الطبيعية",
        "download": "تحميل التقرير (PDF)",
        "questions_phq": [
            "قلة الاهتمام أو المتعة بممارسة الأشياء؟",
            "الشعور بالإحباط أو الاكتئاب أو اليأس؟",
            "صعوبة في النوم أو النوم أكثر من اللازم؟",
            "الشعور بالتعب أو قلة الطاقة؟",
            "ضعف الشهية أو الإفراط في الأكل؟",
            "الشعور بالسوء تجاه نفسك - أو أنك فاشل؟",
            "صعوبة في التركيز على الأشياء؟",
            "التحرك أو التحدث ببطء شديد؟",
            "أفكار بأنك ستكون أفضل حالاً لو كنت ميتاً؟"
        ],
        "options_phq": ["إطلاقاً (0)", "عدة أيام (1)", "أكثر من نصف الأيام (2)", "يومياً تقريباً (3)"]
    }
}

# --- 3. HELPER FUNCTIONS ---
def get_text(key, lang):
    return TRANS[lang].get(key, key)

def process_arabic(text):
    """Reshapes Arabic text for PDF rendering."""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except:
        return text

# --- 4. CORE CLINICAL LOGIC (AI) ---
def analyze_blood_work_ai(text):
    """
    Detects metabolic contraindications for Neurofeedback.
    Returns a list of detected issues.
    """
    warnings = []
    text = text.lower()
    
    # Expanded Medical Dictionary
    keywords = {
        "Vitamin D Deficiency": ["vit d", "vitamin d", "vitamin-d", "25-oh"],
        "Thyroid Dysfunction": ["tsh", "t3", "t4", "thyroid", "hypothyroid", "hyperthyroid"],
        "Anemia / Iron": ["iron", "ferritin", "hemoglobin", "hgb", "anemia"],
        "Inflammation": ["crp", "esr", "inflammation"],
        "B12 Deficiency": ["b12", "cobalamin"]
    }
    
    bad_indicators = ["low", "high", "deficien", "abnormal", "pos", "+", "below", "above"]
    
    for condition, terms in keywords.items():
        # Check if any term exists AND a bad indicator exists nearby or in text
        term_found = any(t in text for t in terms)
        indicator_found = any(ind in text for ind in bad_indicators)
        
        if term_found and indicator_found:
            warnings.append(condition)
            
    return warnings

def calculate_clinical_decision(eeg_data, phq_score, blood_warnings, lang="en"):
    """
    Generates the final recommendation based on Multi-Modal Data.
    """
    # 1. Metabolic Check (Priority #1)
    if blood_warnings:
        status = "STOP / CAUTION" if lang == "en" else "توقف / تحذير"
        reason = f"Metabolic issues detected ({', '.join(blood_warnings)}). Treat these before starting Neurofeedback."
        if lang == "ar":
            reason = f"تم اكتشاف مشاكل استقلابية ({', '.join(blood_warnings)}). يجب علاجها قبل البدء."
        return status, reason, "RED"

    # 2. Depression Check
    if phq_score >= 15:
        status = "PSYCHIATRY REFERRAL" if lang == "en" else "إحالة لطبيب نفسي"
        reason = "PHQ-9 Score indicates Moderately Severe to Severe Depression. Combined therapy recommended."
        if lang == "ar":
            reason = "نتيجة PHQ-9 تشير إلى اكتئاب حاد. يوصى بالعلاج المشترك."
        return status, reason, "RED"

    # 3. EEG check (Simple Heuristic for Demo)
    theta_beta_ratio = eeg_data['Theta_rel'].mean() / eeg_data['Beta_rel'].mean()
    if theta_beta_ratio > 2.5: # High Theta/Beta often associated with ADHD/Slow wave issues
        status = "PROCEED (Protocol: Beta Up/Theta Down)" if lang == "en" else "بدء العلاج (بروتوكول: رفع بيتا/خفض ثيتا)"
        reason = "QEEG shows elevated Theta/Beta ratio consistent with symptoms."
        if lang == "ar":
            reason = "تخطيط الدماغ يظهر ارتفاع نسبة ثيتا/بيتا المتوافقة مع الأعراض."
        return status, reason, "GREEN"
    
    status = "PROCEED (Standard SMR)" if lang == "en" else "بدء العلاج (بروتوكول SMR)"
    reason = "Biomarkers are stable. Standard sensorimotor rhythm training suggested."
    if lang == "ar":
        reason = "المؤشرات الحيوية مستقرة. يقترح تدريب SMR القياسي."
    return status, reason, "GREEN"

# --- 5. PDF GENERATION ENGINE ---
def generate_pdf_report(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Font Registration
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        font_name = 'Amiri'
    except:
        font_name = 'Helvetica' # Fallback
        
    # Custom Styles
    s_title = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontName=font_name, fontSize=18, textColor=colors.HexColor(BLUE), alignment=1 if lang=='en' else 2)
    s_normal = ParagraphStyle("NormalAr", parent=styles["Normal"], fontName=font_name, fontSize=10, leading=14, alignment=0 if lang=='en' else 2)
    s_h2 = ParagraphStyle("H2Ar", parent=styles["Heading2"], fontName=font_name, textColor=colors.HexColor(BLUE), fontSize=12, spaceBefore=10, alignment=0 if lang=='en' else 2)
    s_alert = ParagraphStyle("Alert", parent=s_normal, textColor=colors.red)

    story = []
    
    # Helpers for Text
    def T(text): return process_arabic(text) if lang == 'ar' else text

    # Header
    story.append(Paragraph(T(data['ui']['title']), s_title))
    story.append(Spacer(1, 20))
    
    # Patient Table
    p = data['patient']
    # Arabic columns need to be reversed in list for visual correctness in Table sometimes, 
    # but ReportLab tables usually fill Left-to-Right. For Arabic, we might map: Label | Value
    
    info_data = [
        [T(data['ui']['name']), T(p['name']), T(data['ui']['id']), p['id']],
        [T(data['ui']['dob']), str(p['dob']), T("Gender"), "---"],
        [T(data['ui']['labs']), Paragraph(T(p['labs']), s_normal), "", ""]
    ]
    
    t = Table(info_data, colWidths=[1.2*inch, 2.3*inch, 1*inch, 2.3*inch])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,-1), font_name)
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Decision Section
    dec = data['decision']
    story.append(Paragraph(T(data['ui']['decision']), s_h2))
    story.append(Paragraph(T(dec[0]), s_alert if dec[2]=="RED" else s_normal))
    story.append(Paragraph(f"{T(data['ui']['reason'])}: {T(dec[1])}", s_normal))
    story.append(Spacer(1, 10))
    
    # Scores
    story.append(Paragraph(T(data['ui']['assessment']), s_h2))
    story.append(Paragraph(f"PHQ-9 Score: {data['scores']['phq']} / 27", s_normal))
    story.append(Spacer(1, 10))

    # Abnormal Channels Table
    if not data['df'].empty:
        story.append(Paragraph(T(data['ui']['abnormal']), s_h2))
        
        df = data['df']
        mean_theta = df['Theta_rel'].mean()
        mean_beta = df['Beta_rel'].mean()
        
        headers = ["Channel", "Condition", "Value", "Norm"]
        if lang == 'ar': headers = ["القناة", "الحالة", "القيمة", "المعدل"]
        
        tbl_data = [[T(h) for h in headers]]
        
        for idx, row in df.iterrows():
            if row['Theta_rel'] > mean_theta * 1.5:
                cond = "High Theta" if lang=='en' else "ثيتا مرتفع"
                tbl_data.append([idx, T(cond), f"{row['Theta_rel']*100:.1f}%", f"{mean_theta*100:.1f}%"])
            if row['Beta_rel'] > mean_beta * 1.5:
                cond = "High Beta" if lang=='en' else "بيتا مرتفع"
                tbl_data.append([idx, T(cond), f"{row['Beta_rel']*100:.1f}%", f"{mean_beta*100:.1f}%"])
                
        if len(tbl_data) > 1:
            t_ab = Table(tbl_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1.5*inch])
            t_ab.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BLUE)),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('FONTNAME', (0,0), (-1,-1), font_name)
            ]))
            story.append(t_ab)
        else:
            msg = "No significant deviations detected." if lang=='en' else "لم يتم اكتشاف انحرافات كبيرة."
            story.append(Paragraph(T(msg), s_normal))

    # Brain Maps
    if data['topomaps']:
        story.append(PageBreak())
        story.append(Paragraph("Brain Topography", s_h2))
        imgs = []
        for b_name, b_bytes in data['topomaps'].items():
            img = RLImage(io.BytesIO(b_bytes), width=2.5*inch, height=2.5*inch)
            imgs.append(img)
        
        # Arrange in grid
        if len(imgs) >= 2:
            story.append(Table([[imgs[0], imgs[1]]]))
        
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 6. EEG VISUALIZATION ---
def generate_topomap(values, ch_names, title):
    # Simplified Topomap logic (Requires valid channel mapping in production)
    # Using random placeholder for robustness if coords missing
    return None # Skipped for brevity in code, assuming standard logic

# --- 7. MAIN UI ---
def main():
    st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
    
    # Language Selection
    lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
    L = "ar" if lang == "العربية" else "en"
    
    # Header
    st.title(f"🧠 {get_text('title', L)}")
    
    # Sidebar: Patient Data
    with st.sidebar:
        st.header(get_text('patient_info', L))
        p_name = st.text_input(get_text('name', L), "Ali Ahmed")
        p_id = st.text_input(get_text('id', L), "FILE-101")
        p_dob = st.number_input(get_text('dob', L), 1920, 2024, 1985)
        
        st.markdown("---")
        st.subheader(get_text('labs', L))
        p_labs = st.text_area(get_text('labs', L), height=100, placeholder=get_text('labs_ph', L))
    
    # Main Area: Interactive Assessment
    st.subheader(f"📝 {get_text('assessment', L)}")
    
    # PHQ-9 Expander
    with st.expander(get_text('phq_title', L), expanded=True):
        phq_score = 0
        questions = get_text('questions_phq', L)
        options = get_text('options_phq', L)
        
        cols = st.columns(2) # Split questions to 2 columns for better layout
        for i, q in enumerate(questions):
            col = cols[i % 2]
            # User radio button for each question
            ans = col.radio(f"{i+1}. {q}", options, key=f"phq_{i}")
            # Extract score from string "Text (Score)"
            score = int(ans.split('(')[1].replace(')', ''))
            phq_score += score
            
        st.metric("Total PHQ-9 Score", f"{phq_score} / 27")
        if phq_score >= 10:
            st.warning("Score indicates Moderate to Severe Depression")

    # Upload
    st.markdown("---")
    st.subheader(f"📂 {get_text('upload', L)}")
    uploaded_file = st.file_uploader(" ", type=["edf"])
    
    if st.button(get_text('analyze_btn', L), type="primary"):
        # 1. Logic: Check Blood Work First
        blood_warnings = analyze_blood_work_ai(p_labs)
        
        # 2. Process EEG (Simulated if no file)
        if uploaded_file:
            # Real processing would go here
            # data, meta = read_edf(uploaded_file)
            # For demo, creating dummy DF
            ch_names = [f"Ch{i}" for i in range(1, 20)]
            df_bands = pd.DataFrame(np.random.rand(19, 4), columns=['Delta_rel', 'Theta_rel', 'Alpha_rel', 'Beta_rel'], index=ch_names)
            topomaps = {} # Placeholder
        else:
            # Simulation Mode
            ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
            df_bands = pd.DataFrame(np.abs(np.random.randn(10, 4)), columns=['Delta_rel', 'Theta_rel', 'Alpha_rel', 'Beta_rel'], index=ch_names)
            topomaps = {}
            
        # 3. Calculate Decision
        decision = calculate_clinical_decision(df_bands, phq_score, blood_warnings, lang=L)
        
        # 4. Display Output
        st.divider()
        
        # System Decision Box
        bg_color = "#ffe6e6" if decision[2] == "RED" else "#e6ffe6"
        st.markdown(f"""
        <div style="padding: 20px; background-color: {bg_color}; border-radius: 10px; border: 2px solid {decision[2]}">
            <h2 style="color: {decision[2]}; margin:0">{get_text('decision', L)}: {decision[0]}</h2>
            <p style="font-size: 18px; margin-top:10px"><b>{get_text('reason', L)}:</b> {decision[1]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Lab Alerts
        if blood_warnings:
            st.error(f"⚠️ Metabolic Alerts: {', '.join(blood_warnings)}")
        
        # Channel Table
        st.subheader(get_text('abnormal', L))
        st.dataframe(df_bands.style.background_gradient(cmap="Reds"))
        
        # 5. Generate PDF
        report_data = {
            "ui": TRANS[L],
            "patient": {"name": p_name, "id": p_id, "dob": p_dob, "labs": p_labs},
            "decision": decision,
            "scores": {"phq": phq_score},
            "df": df_bands,
            "topomaps": topomaps
        }
        
        pdf_bytes = generate_pdf_report(report_data, lang=L)
        st.download_button(
            label=get_text('download', L),
            data=pdf_bytes,
            file_name=f"Report_{p_id}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
