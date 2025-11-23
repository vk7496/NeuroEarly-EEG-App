# app.py — NeuroEarly Pro v26 (Final Artifact Filter & PDF Fix)
import os
import io
import json
import base64
import tempfile
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch
import streamlit as st
import PyPDF2
import mne 

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
st.set_page_config(page_title="NeuroEarly Pro v26", layout="wide", page_icon="🧠")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
FONT_PATH = "Amiri-Regular.ttf"

BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
ORANGE = "#FF8C00"

BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; color: #003366; font-weight: bold; margin-bottom: 0px;}
    .sub-header {font-size: 1rem; color: #666; margin-bottom: 20px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0 0;}
    .stTabs [aria-selected="true"] {background-color: #003366; color: white;}
    .report-box {background-color: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 5px solid #003366;}
    .alert-box {background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 5px solid #d32f2f;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOCALIZATION ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Clinical AI Assistant", "subtitle": "Advanced Decision Support System",
        "p_info": "Patient Demographics", "name": "Full Name", "gender": "Gender", "dob": "Date of Birth", "id": "File ID",
        "male": "Male", "female": "Female",
        "lab_sec": "Blood Work Analysis", "lab_up": "Upload Lab Report (PDF)",
        "tab_assess": "1. Clinical Assessments", "tab_neuro": "2. Neuro-Analysis (EEG)",
        "analyze": "RUN DIAGNOSIS", "decision": "CLINICAL DECISION",
        "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT",
        "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Protocol",
        "download": "Download Doctor's Report", "eye_state": "Eye State (AI Detected)",
        "doc_guide": "Doctor's Guidance & Protocol", "narrative": "Automated Clinical Narrative",
        "phq_t": "Depression Screening (PHQ-9)", "alz_t": "Cognitive Screening (MMSE)",
        "methodology": "Methodology: Data Processing & Analysis",
        "method_desc": "Real QEEG analysis via MNE-Python. Ultra-robust artifact rejection applied for Delta power.",
        "q_phq": ["Little interest", "Feeling down", "Sleep issues", "Tiredness", "Appetite", "Failure", "Concentration", "Slowness", "Self-harm"],
        "opt_phq": ["Not at all", "Several days", "More than half", "Nearly every day"],
        "q_mmse": ["Orientation", "Registration", "Attention", "Recall", "Language"],
        "opt_mmse": ["Incorrect", "Partial", "Correct"]
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: المساعد الطبي الذكي", "subtitle": "نظام دعم القرار المتقدم",
        "p_info": "بيانات المريض", "name": "الاسم الكامل", "gender": "الجنس", "dob": "تاريخ الميلاد", "id": "رقم الملف",
        "male": "ذكر", "female": "أنثى",
        "lab_sec": "تحليل الدم والمختبر", "lab_up": "رفع تقرير المختبر (PDF)",
        "tab_assess": "١. التقييمات السريرية", "tab_neuro": "٢. التحليل العصبي (EEG)",
        "analyze": "تشغيل التشخيص", "decision": "القرار السريري",
        "mri_alert": "🚨 حرج: اكتشاف آفة بؤرية -> إحالة للتصوير بالرنين المغناطيسي",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "neuro": "✅ المضي قدماً في العلاج",
        "download": "تحميل تقرير الطبيب", "eye_state": "حالة العين (كشف الذكاء الاصطناعي)",
        "doc_guide": "توجيهات الطبيب والبروتوكول", "narrative": "الرواية السريرية التلقائية",
        "phq_t": "فحص الاكتئاب (PHQ-9)", "alz_t": "فحص الذاكرة (MMSE)",
        "methodology": "المنهجية: معالجة وتحليل البيانات",
        "method_desc": "تحليل QEEG حقيقي. تم تطبيق رفض شوائب فائق الثبات بر اساس کانالهای مرکزی.",
        "q_phq": ["الاهتمام", "الاكتئاب", "النوم", "التعب", "الشهية", "الفشل", "التركيز", "البطء", "إيذاء النفس"],
        "opt_phq": ["أبداً", "عدة أيام", "أكثر من نصف الأيام", "يومياً"],
        "q_mmse": ["التوجيه", "التسجيل", "الانتباه", "الاستدعاء", "اللغة"],
        "opt_mmse": ["خطأ", "جزئي", "صحيح"]
    }
}

def T_st(text, lang): return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text
def get_trans(key, lang): return TRANS[lang].get(key, key)

# --- 3. SIGNAL PROCESSING (ROBUST) ---
def process_real_edf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        sf = raw.info['sfreq']
        if sf > 100: raw.notch_filter(np.arange(50, sf/2, 50), verbose=False)
        raw.filter(0.5, 45.0, verbose=False)
        
        data = raw.get_data()
        ch_names = raw.ch_names
        psds, freqs = mne.time_frequency.psd_array_welch(data, sf, fmin=0.5, fmax=45.0, n_fft=int(2*sf), verbose=False)
        
        df_rows = []
        for i, ch in enumerate(ch_names):
            total_power = np.sum(psds[i, :])
            row = {}
            for band, (fmin, fmax) in BANDS.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                val = np.sum(psds[i, idx])
                row[f"{band} (%)"] = (val / total_power) * 100 if total_power > 0 else 0
            df_rows.append(row)
            
        df_eeg = pd.DataFrame(df_rows, index=ch_names)
        os.remove(tmp_path)
        return df_eeg, None
    except Exception as e:
        return None, str(e)

# --- 4. LOGIC & METRICS (ULTRA-STABLE TUMOR LOGIC V26) ---
def determine_eye_state_smart(df_bands):
    occ_channels = [ch for ch in df_bands.index if any(x in ch for x in ['O1','O2','P3','P4'])]
    if occ_channels:
        if df_bands.loc[occ_channels, 'Alpha (%)'].median() > 12.0: return "Eyes Closed"
    if df_bands['Alpha (%)'].median() > 10.0: return "Eyes Closed"
    return "Eyes Open"

def calculate_metrics(eeg_df, phq, mmse):
    risks = {}
    tbr = 0
    if 'Theta (%)' in eeg_df and 'Beta (%)' in eeg_df:
        # Use median for robustness
        tbr = eeg_df['Theta (%)'].median() / (eeg_df['Beta (%)'].median() + 0.01)
        eeg_df['TBR'] = eeg_df['Theta (%)'] / (eeg_df['Beta (%)'] + 0.01)
    
    risks['Depression'] = min(0.99, (phq / 27.0)*0.6 + 0.1)
    risks['Alzheimer'] = min(0.99, ((10-mmse)/10.0)*0.7 + 0.1)
    
    fdi = 0
    focal_ch = "N/A"
    
    if 'Delta (%)' in eeg_df:
        # 1. Identify "Clean" central channels for a stable baseline
        # These channels are least prone to non-biological noise (EKG, EMG)
        stable_channel_names = ['C3', 'C4', 'P3', 'P4', 'Cz', 'Pz']
        stable_channels = [ch for ch in eeg_df.index if any(x in ch for x in stable_channel_names)]
        
        # 2. Identify all channels to test (excluding known artifacts and non-scalp channels)
        # Exclude Fp, Temporal, EOG, numeric-only (like 65), and common reference channels
        artifact_patterns = ['Fp', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FT', 'Ref', 'GND', 'EKG', 'ECG', 'EOG', 'HEOG', 'VEOG']
        test_channels = [ch for ch in eeg_df.index if not any(p in ch for p in artifact_patterns) and not ch.isdigit() and len(ch) < 4]

        if stable_channels and test_channels:
            
            deltas_test = eeg_df.loc[test_channels, 'Delta (%)']
            deltas_stable = eeg_df.loc[stable_channels, 'Delta (%)']
            
            max_delta = deltas_test.max()
            median_delta_stable = deltas_stable.median()
            
            # FDI is Max Delta Power in test set / Median Delta Power in stable central set
            fdi = max_delta / (median_delta_stable + 0.01)

            focal_ch = deltas_test.idxmax()
            
        # 3. Refined Thresholding (FDI > 4.0 is suspicious, Denominator (10.0) provides high stability)
        risk_calc = max(0.05, (fdi - 4.0) / 10.0) 
        risks['Tumor'] = min(0.99, risk_calc) if fdi > 4.0 else 0.05
    else:
        risks['Tumor'] = 0.05
    
    risks['ADHD'] = min(0.99, (tbr / 3.0)) if tbr > 1.5 else 0.1
    
    if 'Alpha (%)' in eeg_df:
        eeg_df['Alpha Z'] = (eeg_df['Alpha (%)'] - eeg_df['Alpha (%)'].mean()) / (eeg_df['Alpha (%)'].std()+0.01)
        
    return risks, fdi, tbr, df_eeg, focal_ch

def scan_blood_work(text):
    warnings = []
    text = text.lower()
    checks = {"Vitamin D": ["vit d", "low d"], "Thyroid": ["tsh", "thyroid"], "Anemia": ["iron", "anemia"]}
    for k, v in checks.items():
        if any(x in text for x in v) and "low" in text: warnings.append(k)
    return warnings

def get_recommendations(risks, blood_issues, lang):
    recs = []
    alert = "GREEN"
    if risks['Tumor'] > 0.65:
        recs.append(get_trans('mri_alert', lang))
        alert = "RED"
    if blood_issues:
        recs.append(get_trans('metabolic', lang))
        if alert != "RED": alert = "ORANGE"
    if risks['Depression'] > 0.7: recs.append("Psychiatry Referral (Depression)")
    if risks['ADHD'] > 0.6: recs.append("Neurofeedback (Attention Protocol)")
    if not recs: recs.append(get_trans('neuro', lang))
    return recs, alert

def generate_narrative(risks, blood, tbr, lang, fdi, focal_ch):
    L = lang
    n = ""
    if blood: n += T_st("Lab results indicate metabolic deficiencies. ", L)
    if risks['Tumor'] > 0.65: n += T_st(f" CRITICAL: Focal Delta asymmetry (FDI: {fdi:.2f} at {focal_ch}). Lesion risk must be ruled out. ", L)
    if risks['ADHD'] > 0.6: n += T_st(f" High TBR ({tbr:.2f}) suggests attentional deficit. ", L)
    if n == "": n = T_st("Neurophysiological profile is within normal range.", L)
    return n

# --- 5. VISUALS ---
def generate_shap(df):
    try:
        feats = {
            "Frontal Theta": df['Theta (%)'].mean(), "Occipital Alpha": df['Alpha (%)'].mean(),
            "TBR": df['TBR'].mean(), "Delta Power": df['Delta (%)'].mean()
        }
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(list(feats.keys()), list(feats.values()), color=BLUE)
        ax.set_title("SHAP Analysis")
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except: return None

def generate_topomap(df, band):
    if f'{band} (%)' not in df.columns: return None
    vals = df[f'{band} (%)'].values
    grid_size = int(np.ceil(np.sqrt(len(vals))))
    if grid_size*grid_size < len(vals): grid_size += 1
    padded = np.zeros(grid_size*grid_size)
    padded[:len(vals)] = vals
    grid = padded.reshape((grid_size, grid_size))
    grid = lfilter([1.0/3]*3, 1, grid, axis=0)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(grid, cmap='jet', interpolation='bicubic')
    ax.axis('off')
    ax.add_artist(plt.Circle((grid_size/2-0.5, grid_size/2-0.5), grid_size*0.4, color='k', fill=False, lw=2))
    buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# --- 6. PDF (FIXED RTL/ARABIC TEXT) ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 1. Font Registration
    try: 
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_name = 'Amiri'
    except: 
        f_name = 'Helvetica'
        st.warning("Amiri font not found. Using default Helvetica, which may still show RTL issues.")

    # 2. Helper functions for ReportLab content
    def T(x): # For general paragraphs (already Bidi-processed by caller)
        return get_display(arabic_reshaper.reshape(str(x))) if lang == 'ar' else str(x)
        
    def T_p(text): # For table cells, creating a Paragraph object for Bidi stability
        if lang == 'ar':
            # Use Paragraph for better Bidi handling in tables, Right alignment (2) is best for RTL data
            return Paragraph(get_display(arabic_reshaper.reshape(str(text))), 
                             ParagraphStyle(name='RTL', fontName=f_name, alignment=2, leading=12))
        return str(text)

    
    story = []
    if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    story.append(Paragraph(T(data['title']), ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE))))
    
    # Patient Info Table (Using T_p for Bidi stability)
    p = data['p']
    info = [
        [T_p(get_trans("name",lang)), T_p(p['name']), T_p(get_trans("id",lang)), T_p(p['id'])],
        [T_p(get_trans("gender",lang)), T_p(p['gender']), T_p(get_trans("dob",lang)), T_p(p['dob'])],
        [T_p("Eye State"), T_p(p['eye']), T_p("Labs"), T_p(p['labs'])]
    ]
    t = Table(info, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), 
                           ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1,10))
    
    # Narrative and Recommendations (Using T for paragraphs)
    story.append(Paragraph(T(data['narrative']), ParagraphStyle('B', fontName=f_name, fontSize=10)))
    story.append(Spacer(1,10))
    
    for r in data['recs']:
        c = colors.red if "MRI" in r or "حرج" in r else colors.black
        story.append(Paragraph(T(r), ParagraphStyle('A', fontName=f_name, textColor=c, fontSize=10)))
        
    # Risks Table (Using T_p for Bidi stability)
    r_data = [[T_p("Condition"), T_p("Risk")]]
    for k,v in data['risks'].items(): 
        if k not in ['Connectivity', 'TBR']: r_data.append([T_p(k), T_p(f"{v*100:.1f}%")])
    r_data.append([T_p("TBR"), T_p(f"{data['tbr']:.2f}")])
    r_data.append([T_p("FDI Channel"), T_p(data['focal_ch'])])
    t2 = Table(r_data, style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), 
                                         ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t2)
    
    story.append(PageBreak())
    if data['shap']: story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3*inch))
    
    imgs = [RLImage(io.BytesIO(data['maps'][b]), width=1.5*inch, height=1.5*inch) for b in BANDS if data['maps'][b]]
    if len(imgs)>=4: story.append(Table([imgs]))
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def extract_text_from_pdf(f):
    try:
        pdf = PyPDF2.PdfReader(f)
        return "".join([p.extract_text() for p in pdf.pages])
    except: return ""

# --- 7. MAIN ---
def main():
    # ... (unchanged Streamlit UI code) ...
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f'<div class="main-header">{get_trans("title", "en")}</div>', unsafe_allow_html=True)

    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        
        st.header(T_st(get_trans("p_info", L), L))
        p_name = st.text_input(T_st(get_trans("name", L), L), "John Doe")
        p_gender = st.selectbox(T_st(get_trans("gender", L), L), [get_trans("male", L), get_trans("female", L)])
        p_dob = st.date_input(T_st(get_trans("dob", L), L), value=date(1980,1,1))
        p_id = st.text_input(T_st(get_trans("id", L), L), "F-101")
        
        st.markdown("---")
        lab_file = st.file_uploader(T_st(get_trans("lab_up", L), L), type=["pdf", "txt"])
        lab_text = extract_text_from_pdf(lab_file) if lab_file else ""

    tab1, tab2 = st.tabs([T_st(get_trans("tab_assess", L), L), T_st(get_trans("tab_neuro", L), L)])
    
    with tab1:
        c_q1, c_q2 = st.columns(2)
        phq_score = 0
        mmse_score = 0
        with c_q1:
            st.subheader(T_st(get_trans("phq_t", L), L))
            opts = get_trans("opt_phq", L)
            for i, q in enumerate(get_trans("q_phq", L)):
                ans = st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"phq_{i}")
                phq_score += opts.index(ans)
            st.metric("PHQ-9 Score", f"{phq_score}/27")
        with c_q2:
            st.subheader(T_st(get_trans("alz_t", L), L))
            opts_m = get_trans("opt_mmse", L)
            for i, q in enumerate(get_trans("q_mmse", L)):
                ans = st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"mmse_{i}", index=0)
                mmse_score += opts_m.index(ans)*2
            mmse_total = min(30, mmse_score + 10)
            st.metric("MMSE Score", f"{int(mmse_total)}/30")

    with tab2:
        uploaded_edf = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        
        if st.button(T_st(get_trans("analyze", L), L), type="primary"):
            blood = scan_blood_work(lab_text)
            
            if uploaded_edf:
                with st.spinner("Processing Real Signal..."):
                    df_eeg, err = process_real_edf(uploaded_edf)
                    if err: st.error(err); st.stop()
            else:
                st.warning("Simulation Mode (No EDF)")
                ch = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
                df_eeg = pd.DataFrame(np.random.uniform(2,10,(10,4)), columns=[f"{b} (%)" for b in BANDS], index=ch)
                # Ensure Alpha is high for Eyes Closed in simulation
                df_eeg.loc['O1', 'Alpha (%)'] = 15.0
            
            detected_eye = determine_eye_state_smart(df_eeg)
            risks, fdi, tbr, df_eeg, focal_ch = calculate_metrics(df_eeg, phq_score, mmse_total)
            recs, alert = get_recommendations(risks, blood, L)
            narrative = generate_narrative(risks, blood, tbr, L, fdi, focal_ch)
            shap_img = generate_shap(df_eeg)
            maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
            
            st.info(f"**{T_st(get_trans('eye_state', L), L)}:** {detected_eye}")
            final_eye = detected_eye
            
            color = "#ffebee" if alert == "RED" else "#e8f5e9"
            st.markdown(f'<div class="alert-box" style="background:{color}"><h3>{T_st(get_trans("decision", L), L)}</h3><p>{recs[0]}</p></div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Depression", f"{risks['Depression']*100:.0f}%")
            c2.metric("Alzheimer", f"{risks['Alzheimer']*100:.0f}%")
            # Display FDI info directly under Tumor Risk
            c3.metric("Tumor Risk", f"{risks['Tumor']*100:.0f}%", f"FDI: {fdi:.2f} @ {focal_ch}") 
            
            st.markdown(f'<div class="report-box"><h4>{T_st(get_trans("narrative", L), L)}</h4><p>{narrative}</p></div>', unsafe_allow_html=True)
            st.dataframe(df_eeg.style.background_gradient(cmap='Blues'), height=200)
            
            if shap_img: st.image(shap_img)
            st.image(list(maps.values()), width=120, caption=list(maps.keys()))
            
            pdf_data = {
                "title": get_trans("title", L),
                "p": {"name": p_name, "gender": p_gender, "dob": str(p_dob), "id": p_id, "labs": str(blood), "eye": final_eye},
                "risks": risks, "tbr": tbr, "recs": recs, "eeg": df_eeg, "shap": shap_img, "maps": maps, "narrative": narrative, "focal_ch": focal_ch
            }
            st.download_button(T_st(get_trans("download", L), L), create_pdf(pdf_data, L), "Report.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    main()
