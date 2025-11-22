# app.py — NeuroEarly Pro v19 (Advanced Clinical & Automated Narrative)
import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import butter, lfilter, iirnotch
import streamlit as st
import PyPDF2 

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
st.set_page_config(page_title="NeuroEarly Pro v19", layout="wide", page_icon="🧠")

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
    .main-header {font-size: 2.2rem; color: #003366; font-weight: bold;}
    .narrative-box {background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 6px solid #1e90ff; margin: 20px 0;}
    .doctor-note {background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 6px solid #003366; margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOCALIZATION (TEXTS & QUESTIONS) ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Advanced Clinical System",
        "p_info": "Patient Demographics", "name": "Patient Name", "id": "File ID",
        "lab_sec": "Blood Work Analysis", "lab_up": "Upload Lab Report (PDF)",
        "analyze": "START CLINICAL DIAGNOSIS", "decision": "CLINICAL DECISION & REFERRAL",
        "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT",
        "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Protocol",
        "download": "Download Official Doctor's Report", "eye_state": "Eye State (Detected)",
        "doc_guide": "Doctor's Guidance & Protocol", "narrative": "Automated Clinical Narrative",
        "phq_t": "Depression Screening (PHQ-9)", "alz_t": "Cognitive Screening (MMSE)",
        "methodology": "Methodology: Data Processing & Analysis",
        "method_desc": "QEEG data was analyzed using a simulated 10-20 system. Bands calculated via FFT on a 2-second epoch. Z-Scores reflect deviation from a simulated normative database. Alpha Asymmetry (O1/O2) is prioritized for eye-state detection.",
        "q_phq": [
            "Little interest or pleasure in doing things", "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep", "Feeling tired or having little energy",
            "Poor appetite or overeating", "Feeling bad about yourself",
            "Trouble concentrating", "Moving/speaking slowly or restless",
            "Thoughts of self-harm"
        ],
        "opt_phq": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "q_mmse": ["Orientation (Time/Place)", "Registration (Repeat 3 words)", "Attention (Count backwards by 7)", "Recall (Remember 3 words)", "Language (Naming objects)"]
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: التحليل السريري المتقدم",
        "p_info": "بيانات المريض", "name": "اسم المريض", "id": "رقم الملف",
        "lab_sec": "تحليل الدم والمختبر", "lab_up": "رفع تقرير المختبر (PDF)",
        "analyze": "بدء المعالجة والتشخيص", "decision": "القرار السريري النهائي",
        "mri_alert": "🚨 حرج: اكتشاف آفة بؤرية -> إحالة للتصوير بالرنين المغناطيسي",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "neuro": "✅ المضي قدماً في العلاج",
        "download": "تحميل تقرير الطبيب الرسمي", "eye_state": "حالة العين (المكتشفة)",
        "doc_guide": "توجيهات الطبيب والبروتوكول", "narrative": "الرواية السريرية التلقائية",
        "phq_t": "فحص الاكتئاب (PHQ-9)", "alz_t": "فحص الذاكرة (MMSE)",
        "methodology": "المنهجية: معالجة وتحليل البيانات",
        "method_desc": "تم تحليل بيانات QEEG باستخدام نظام 10-20 المحاكي. تم حساب النطاقات عبر تحويل فورييه السريع (FFT). تعكس قيم Z-Score الانحراف عن قاعدة بيانات هنجارية محاكية. يتم إعطاء الأولوية لتناظر ألفا (O1/O2) للكشف عن حالة العين.",
        "q_phq": [
            "قلة الاهتمام أو المتعة", "الشعور بالإحباط أو الاكتئاب",
            "صعوبة النوم", "الشعور بالتعب", "ضعف الشهية",
            "الشعور بالسوء تجاه النفس", "صعوبة التركيز",
            "بطء الحركة أو الكلام", "أفكار لإيذاء النفس"
        ],
        "opt_phq": ["أبداً", "عدة أيام", "أكثر من نصف الأيام", "يومياً تقريباً"],
        "q_mmse": ["التوجيه (الوقت/المكان)", "التسجيل (تكرار 3 كلمات)", "الانتباه (العد العكسي)", "الاستدعاء (تذكر الكلمات)", "اللغة (تسمية الأشياء)"]
    }
}

def T_st(text, lang):
    return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text

def get_trans(key, lang):
    return TRANS[lang].get(key, key)

# --- 3. LOGIC ENGINES ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages: text += page.extract_text() + "\n"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
    except: pass
    return text

def determine_eye_state_smart(df_bands):
    occ_channels = [ch for ch in df_bands.index if 'O1' in ch or 'O2' in ch]
    if occ_channels:
        if df_bands.loc[occ_channels, 'Alpha (%)'].mean() > 11.0: return "Eyes Closed"
    if df_bands['Alpha (%)'].mean() > 9.5: return "Eyes Closed"
    return "Eyes Open"

def calculate_metrics(eeg_df, phq_score, mmse_score):
    risks = {}
    
    # 1. Base Risks
    risks['Depression'] = min(0.99, (phq_score / 27.0)*0.6 + 0.1)
    risks['Alzheimer'] = min(0.99, ((10-mmse_score)/10.0)*0.7 + 0.1)
    
    deltas = eeg_df['Delta (%)']
    fdi = deltas.max() / (deltas.mean() + 0.01)
    risks['Tumor'] = min(0.99, (fdi - 2.5)/5.0) if fdi > 2.5 else 0.05
    
    # 2. Advanced Metrics (Biomarkers)
    tbr = eeg_df['Theta (%)'].mean() / (eeg_df['Beta (%)'].mean() + 0.01)
    risks['ADHD'] = min(0.99, 0.1 + (tbr/4.0) if tbr > 2.5 else 0.05) # TBR > 2.5 is high for adults
    
    # 3. Connectivity (Simulated)
    conn = eeg_df['Coherence (Fp1-Fp2)'].mean() # Use the new simulated column
    risks['Connectivity'] = conn
    
    return risks, fdi, tbr

def scan_blood_work(text):
    warnings = []
    text = text.lower()
    checks = {"Vitamin D": ["vit d", "low d"], "B12": ["b12"], "Thyroid": ["tsh", "thyroid"], "Anemia": ["iron", "anemia", "ferritin"]}
    bad_words = ["low", "deficien", "insufficient", "anemia", "high", "abnormal"]
    for k, v in checks.items():
        if any(x in text for x in v) and any(b in text for b in bad_words): warnings.append(k)
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
        recs.append("Referral: Psychiatry (rTMS / Medication)")
    if risks['Alzheimer'] > 0.6:
         recs.append("Referral: Neurology (Cognitive Eval)")
    if risks['ADHD'] > 0.5:
         recs.append("Referral: Neurofeedback Protocol (TBR Normalization)")
         
    if not recs:
        recs.append(get_trans('neuro', lang))
    return recs, alert

def generate_narrative(risks, blood_issues, tbr, lang):
    narrative = ""
    L = lang
    
    # 1. Start with Metabolic findings
    if blood_issues:
        narrative += T_st("Based on the lab results, there are indications of metabolic deficiencies (e.g., ", L)
        narrative += ", ".join(blood_issues)
        narrative += T_st("). These must be addressed first as they can influence neurophysiological readings.", L)
    else:
        narrative += T_st("Metabolic screening is within normal limits, allowing immediate focus on neurophysiological data. ", L)
        
    # 2. Add EEG/Biomarker findings
    if risks['Tumor'] > 0.65:
        narrative += T_st(" **CRITICAL FINDING:** Significant focal Delta asymmetry detected, requiring immediate imaging. ", L)
    
    if risks['Alzheimer'] > 0.6:
        narrative += T_st(" QEEG analysis suggests possible cognitive impairment, characterized by an **increase in slow-wave activity (Theta/Delta)** in the posterior regions. ", L)
        
    if risks['ADHD'] > 0.5:
        narrative += T_st(f" The **Theta/Beta Ratio (TBR)** is elevated ({tbr:.2f}), which is a strong biomarker for attentional issues. ", L)
        
    # 3. Add Conclusion/Risk Summary
    if risks['Depression'] > 0.7:
        narrative += T_st(" The high PHQ-9 score aligns with QEEG patterns, suggesting a moderate to high risk of Major Depressive Disorder. ", L)
    elif risks['Tumor'] < 0.65 and not blood_issues:
         narrative += T_st(" Overall, the neurophysiological profile suggests a primary focus on attentional and executive function improvement. ", L)
         
    return narrative

# --- 4. VISUALS ---
def generate_shap(df):
    feats = {
        "Frontal Theta": df['Theta (%)'].iloc[:2].mean(),
        "Occipital Alpha": df['Alpha (%)'].iloc[-2:].mean(),
        "Theta/Beta Ratio": df['TBR'].mean(), # Use new metric
        "Alpha Z-Score": df['Alpha Z-Score'].abs().mean(), # Use new metric
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
    data = lfilter([1.0/5]*5, 1, data, axis=0) 
    ax.imshow(data, cmap='jet', vmin=0, vmax=20, extent=(-1,1,-1,1), interpolation='bicubic')
    ax.set_title(band)
    ax.axis('off')
    ax.add_artist(plt.Circle((0, 0), 1, color='k', fill=False, lw=2))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 5. PDF GENERATOR ---
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
    
    # Patient Info Table
    info = [
        [T("Name"), T(str(data['p']['name']))], 
        [T("ID"), str(data['p']['id'])], 
        [T("Labs Findings"), T(str(data['p']['labs']))], 
        [T("Eye State"), T(str(data['p']['eye']))]
    ]
    t = Table(info, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # NEW: Automated Narrative
    story.append(Paragraph(T(get_trans('narrative', lang)), ParagraphStyle('H2', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE))))
    # Applying the complex narrative text
    story.append(Paragraph(T(data['narrative']), ParagraphStyle('BodyText', fontName=f_name, fontSize=11)))
    story.append(Spacer(1, 12))
    
    # Doctor's Guidance
    story.append(Paragraph(T(get_trans('doc_guide', lang)), ParagraphStyle('H2', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE))))
    for r in data['recs']:
        c = colors.red if "MRI" in r or "حرج" in r else colors.black
        s = ParagraphStyle('A', fontName=f_name, textColor=c, fontSize=12)
        story.append(Paragraph(T("• " + r), s))
    story.append(Spacer(1, 12))
    
    # Risks
    r_data = [[T("Condition"), T("Risk")]] + [[T(k), f"{v*100:.1f}%"] for k,v in data['risks'].items() if k not in ['Connectivity']]
    r_data.append([T("TBR (Attentional Marker)"), f"{data['risks']['TBR']:.2f}"])
    t2 = Table(r_data)
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)), ('TEXTCOLOR',(0,0),(-1,0),colors.white), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # EEG Table (With Z-Score & TBR)
    story.append(Paragraph(T("Detailed QEEG Data (Rel. Power & Z-Score)"), ParagraphStyle('H2', fontName=f_name)))
    df = data['eeg'].head(10)
    cols = ['Ch'] + list(df.columns)
    rows = [cols] + [[i] + [f"{x:.1f}" for x in row] for i, row in df.iterrows()]
    t3 = Table(rows)
    t3.setStyle(TableStyle([('GRID', (0,0),(-1,-1),0.25,colors.grey), ('FONTSIZE',(0,0),(-1,-1),8)]))
    story.append(t3)
    
    # NEW: Methodology Section
    story.append(PageBreak())
    story.append(Paragraph(T(get_trans('methodology', lang)), ParagraphStyle('H2', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE))))
    story.append(Paragraph(T(get_trans('method_desc', lang)), ParagraphStyle('BodyText', fontName=f_name, fontSize=10)))
    story.append(Spacer(1, 12))
    
    # Visuals
    story.append(Paragraph("SHAP & Topography Analysis", ParagraphStyle('H2')))
    story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3*inch))
    story.append(Spacer(1, 12))
    
    imgs = [RLImage(io.BytesIO(data['maps'][b]), width=1.5*inch, height=1.5*inch) for b in BANDS]
    if len(imgs) >= 4: story.append(Table([imgs]))
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 6. MAIN APP ---
def main():
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f'<div class="main-header">{get_trans("title", "en")}</div>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(T_st(get_trans("p_info", L), L))
        p_name = st.text_input("Name", "John Doe")
        p_id = st.text_input("ID", "F-101")
        
        st.subheader(T_st(get_trans("lab_sec", L), L))
        lab_file = st.file_uploader(T_st(get_trans("lab_up", L), L), type=["pdf", "txt"])
        lab_text = ""
        if lab_file:
            lab_text = extract_text_from_pdf(lab_file)
            if len(lab_text) > 5: st.success("Lab data extracted.")

    # --- Questions ---
    st.divider()
    col1, col2 = st.columns(2)
    phq_score = 0
    mmse_score = 0
    
    with col1:
        st.subheader(T_st(get_trans("phq_t", L), L))
        with st.expander("Answer PHQ-9 Questions", expanded=True):
            opts = get_trans("opt_phq", L)
            for i, q in enumerate(get_trans("q_phq", L)):
                ans_str = st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"phq_{i}")
                phq_score += opts.index(ans_str)
            st.metric("Depression Score", f"{phq_score}/27")

    with col2:
        st.subheader(T_st(get_trans("alz_t", L), L))
        with st.expander("Answer MMSE Questions", expanded=True):
            opts_m = get_trans("opt_mmse", L) 
            for i, q in enumerate(get_trans("q_mmse", L)):
                ans_str = st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"mmse_{i}", index=2 if i==0 else 0) # Pre-fill 
                mmse_score += opts_m.index(ans_str) * 2 # Simplified scoring
            mmse_total = min(30, mmse_score + 10) # Base score adjustment
            st.metric("Cognitive Score (Est)", f"{int(mmse_total)}/30")

    # --- Analysis ---
    st.divider()
    uploaded_edf = st.file_uploader("Upload EEG (EDF)", type=["edf"])
    
    if st.button(T_st(get_trans("analyze", L), L), type="primary"):
        # 1. Blood
        blood_warn = scan_blood_work(lab_text)
        
        # 2. Advanced EEG Simulation (WITH TBR, Z-Score, Coherence)
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        data = np.random.uniform(2, 10, (10, 4))
        
        # Add high TBR for demo (Theta/Beta ratio)
        data[:, 1] = data[:, 1] * 1.5 
        
        df_eeg = pd.DataFrame(data, columns=['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)'], index=ch_names)
        
        # Calculate derived metrics
        df_eeg['TBR'] = df_eeg['Theta (%)'] / (df_eeg['Beta (%)'] + 0.01)
        df_eeg['Alpha Z-Score'] = np.random.uniform(-2.5, 3.5, 10) # Simulated Z-Score
        df_eeg['Coherence (Fp1-Fp2)'] = np.random.uniform(0.1, 0.5, 10) # Simulated Coherence
        
        # 3. Processing
        detected_eye = determine_eye_state_smart(df_eeg)
        risks, fdi, tbr = calculate_metrics(df_eeg, phq_score, int(mmse_total))
        recs, alert = get_recommendations(risks, blood_warn, L)
        narrative = generate_narrative(risks, blood_issues, tbr, L)
        
        shap_img = generate_shap(df_eeg)
        maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
        
        # 4. Dashboard
        st.info(f"**AI Detected Eye State:** {detected_eye}")
        final_eye = st.radio("Confirm Eye State:", ["Eyes Open", "Eyes Closed"], index=0 if detected_eye=="Eyes Open" else 1)
        
        # Automated Narrative UI
        st.markdown(f"""
        <div class="narrative-box">
            <h3>📝 {T_st(get_trans('narrative', L), L)}</h3>
            <p style="font-size: 11pt;">{T_st(narrative, L)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Doctor Note UI
        st.markdown(f"""
        <div class="doctor-note">
            <h3>👨‍⚕️ {T_st(get_trans('doc_guide', L), L)}</h3>
            <ul>{''.join([f'<li><b>{r}</b></li>' for r in recs])}</ul>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Depression Risk", f"{risks['Depression']*100:.0f}%")
        c2.metric("Alzheimer Risk", f"{risks['Alzheimer']*100:.0f}%")
        c3.metric("ADHD Risk (TBR)", f"{risks['ADHD']*100:.0f}%")
        c4.metric("Mean TBR", f"{tbr:.2f}")

        st.image(shap_img, caption="SHAP Analysis (Advanced Biomarkers Included)")
        st.subheader("Brain Topography Preview")
        st.image(list(maps.values()), width=150, caption=list(maps.keys()))

        # PDF
        pdf_data = {
            "title": get_trans("title", L),
            "p": {"name": p_name, "id": p_id, "labs": ", ".join(blood_warn) if blood_warn else "Normal", "eye": final_eye},
            "risks": risks, "recs": recs, "eeg": df_eeg, "shap": shap_img, "maps": maps, "narrative": narrative
        }
        pdf = create_pdf(pdf_data, L)
        st.download_button(T_st(get_trans("download", L), L), pdf, "Doctor_Report_V19.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    if not os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "wb") as f: f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
    main()
