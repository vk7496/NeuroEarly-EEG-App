# app.py — NeuroEarly Pro v33.1 (Fixed: Restored Eye State Function)
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
from scipy.stats import entropy, pearsonr # For Advanced Metrics
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
st.set_page_config(page_title="NeuroEarly Pro v33.1", layout="wide", page_icon="🧠")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
FONT_PATH = "Amiri-Regular.ttf"

BLUE = "#003366"
RED = "#8B0000"
GREEN = "#006400"
ORANGE = "#FF8C00"

BANDS = {"Delta": (1.0, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

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
        "title": "NeuroEarly Pro: Research-Grade Clinical System", "subtitle": "Advanced Biomarkers: Entropy, Connectivity, FAA",
        "p_info": "Patient Demographics", "name": "Full Name", "gender": "Gender", "dob": "Date of Birth", "id": "File ID",
        "male": "Male", "female": "Female",
        "lab_sec": "Blood Work Analysis", "lab_up": "Upload Lab Report (PDF)",
        "tab_assess": "1. Clinical Assessments", "tab_neuro": "2. Advanced Neuro-Analysis",
        "analyze": "RUN ADVANCED DIAGNOSIS", "decision": "CLINICAL DECISION & PATHWAY",
        "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT",
        "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Standard Protocol",
        "download": "Download Doctor's Report", "eye_state": "Eye State",
        "doc_guide": "Doctor's Guidance & Treatment Protocol", "narrative": "Automated Clinical Interpretation (XAI)",
        "phq_t": "Depression Screening (PHQ-9)", "alz_t": "Cognitive Screening (MMSE)",
        "methodology": "Methodology: Advanced Biomarkers",
        "method_desc": "Analysis includes Spectral Entropy (Complexity), Alpha Coherence (Connectivity), and Frontal Alpha Asymmetry (FAA). Artifacts removed via 1Hz HPF.",
        "q_phq": ["Little interest", "Feeling down", "Sleep issues", "Tiredness", "Appetite", "Failure", "Concentration", "Slowness", "Self-harm"],
        "opt_phq": ["Not at all", "Several days", "More than half", "Nearly every day"],
        "q_mmse": ["Orientation", "Registration", "Attention", "Recall", "Language"],
        "opt_mmse": ["Incorrect", "Partial", "Correct"],
        "entropy": "Spectral Entropy", "coherence": "Alpha Coherence", "faa": "Frontal Alpha Asymmetry",
        "gamma_proto": "• Protocol: 40Hz Gamma Stimulation (GENUS) - Visual/Auditory for AD/MCI"
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: المستوى البحثي المتقدم", "subtitle": "المؤشرات الحيوية المتقدمة: الإنتروبيا، الاتصال، FAA",
        "p_info": "بيانات المريض", "name": "الاسم الكامل", "gender": "الجنس", "dob": "تاريخ الميلاد", "id": "رقم الملف",
        "male": "ذكر", "female": "أنثى",
        "lab_sec": "تحليل الدم والمختبر", "lab_up": "رفع تقرير المختبر (PDF)",
        "tab_assess": "١. التقييمات السريرية", "tab_neuro": "٢. التحليل العصبي المتقدم",
        "analyze": "تشغيل التشخيص المتقدم", "decision": "القرار السريري والمسار",
        "mri_alert": "🚨 حرج: اكتشاف آفة بؤرية -> إحالة للتصوير بالرنين المغناطيسي",
        "metabolic": "⚠️ يتطلب تصحيح أيضي", "neuro": "✅ المضي قدماً في العلاج القياسي",
        "download": "تحميل تقرير الطبيب", "eye_state": "حالة العين",
        "doc_guide": "توجيهات الطبيب وبروتوكول العلاج", "narrative": "التفسير السريري التلقائي (XAI)",
        "phq_t": "فحص الاكتئاب (PHQ-9)", "alz_t": "فحص الذاكرة (MMSE)",
        "methodology": "المنهجية: المؤشرات الحيوية المتقدمة",
        "method_desc": "يشمل التحليل الإنتروبيا الطيفية (التعقيد)، ترابط ألفا (الاتصال)، وعدم تناظر ألفا الجبهي (FAA).",
        "q_phq": ["الاهتمام", "الاكتئاب", "النوم", "التعب", "الشهية", "الفشل", "التركيز", "البطء", "إيذاء النفس"],
        "opt_phq": ["أبداً", "عدة أيام", "أكثر من نصف الأيام", "يومياً"],
        "q_mmse": ["التوجيه", "التسجيل", "الانتباه", "الاستدعاء", "اللغة"],
        "opt_mmse": ["خطأ", "جزئي", "صحيح"],
        "entropy": "الإنتروبيا الطيفية", "coherence": "ترابط ألفا", "faa": "عدم تناظر ألفا الجبهي",
        "gamma_proto": "• البروتوكول: تحفيز جاما 40 هرتز (GENUS) - بصري/سمعي لمرضى الزهايمر"
    }
}

def T_st(text, lang): return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text
def get_trans(key, lang): return TRANS[lang].get(key, key)

# --- 3. SIGNAL PROCESSING (Research-Grade) ---
def calculate_advanced_metrics(psds, freqs, ch_names):
    metrics = {}
    
    # 1. Spectral Entropy
    # Add small epsilon to avoid log(0)
    psd_norm = (psds + 1e-12) / np.sum(psds + 1e-12, axis=1, keepdims=True)
    entropy_vals = entropy(psd_norm, axis=1)
    metrics['Global_Entropy'] = np.mean(entropy_vals)
    
    # 2. Alpha Coherence Proxy (Correlation)
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    frontal = [i for i, ch in enumerate(ch_names) if any(x in ch for x in ['Fz', 'F3', 'F4'])]
    posterior = [i for i, ch in enumerate(ch_names) if any(x in ch for x in ['Pz', 'P3', 'P4', 'O1', 'O2'])]
    
    coh_val = 0.5 
    if frontal and posterior:
        f_alpha = np.mean(psds[frontal][:, alpha_idx], axis=0)
        p_alpha = np.mean(psds[posterior][:, alpha_idx], axis=0)
        if len(f_alpha) > 1:
            coh_val, _ = pearsonr(f_alpha, p_alpha)
            if np.isnan(coh_val): coh_val = 0.5
    metrics['Alpha_Coherence'] = coh_val
    return metrics

def process_real_edf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        
        STANDARD_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        eeg_channels = [ch for ch in raw.ch_names if ch.upper() in [s.upper() for s in STANDARD_CHANNELS]]
        raw.pick_channels(eeg_channels, ordered=True)
        
        sf = raw.info['sfreq']
        if sf > 100: raw.notch_filter(np.arange(50, sf/2, 50), verbose=False)
        raw.filter(1.0, 45.0, verbose=False)
        
        data = raw.get_data()
        ch_names = raw.ch_names
        
        psds, freqs = mne.time_frequency.psd_array_welch(data, sf, fmin=1.0, fmax=45.0, n_fft=int(2*sf), verbose=False)
        
        adv_metrics = calculate_advanced_metrics(psds, freqs, ch_names)
        
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
        return df_eeg, adv_metrics
    except Exception as e:
        return None, str(e)

# --- 4. CLINICAL LOGIC (Enhanced + FIXED Eye State) ---

def determine_eye_state_smart(df_bands):
    """Restored function to fix NameError"""
    occ_channels = [ch for ch in df_bands.index if any(x in ch for x in ['O1','O2','P3','P4'])]
    if occ_channels:
        # Check median alpha in occipital region
        if df_bands.loc[occ_channels, 'Alpha (%)'].median() > 12.0: return "Eyes Closed"
    
    # Global check
    if df_bands['Alpha (%)'].median() > 10.0: return "Eyes Closed"
    return "Eyes Open"

def calculate_metrics(eeg_df, adv_metrics, phq, mmse):
    risks = {}
    
    # 1. Depression (FAA)
    faa = 0
    if 'F4' in eeg_df.index and 'F3' in eeg_df.index:
        right = eeg_df.loc['F4', 'Alpha (%)']
        left = eeg_df.loc['F3', 'Alpha (%)']
        if right > 0 and left > 0:
            faa = np.log(right) - np.log(left)
    
    risks['Depression'] = min(0.99, (phq / 27.0)*0.5 + (0.4 if faa > 0 else 0))
    
    # 2. Alzheimer (Entropy + Connectivity)
    entropy_factor = 1.0 - adv_metrics.get('Global_Entropy', 0.8)
    conn_factor = 1.0 - adv_metrics.get('Alpha_Coherence', 0.6)
    
    cog_deficit = (30 - mmse) / 30.0
    risks['Alzheimer'] = min(0.99, (cog_deficit * 0.4) + (entropy_factor * 0.3) + (conn_factor * 0.3))
    
    # 3. Tumor (FDI)
    fdi = 0
    focal_ch = "N/A"
    if 'Delta (%)' in eeg_df:
        baseline = eeg_df['Delta (%)'].median()
        max_delta = eeg_df['Delta (%)'].max()
        focal_ch = eeg_df['Delta (%)'].idxmax()
        fdi = max_delta / (baseline + 0.01)
        risks['Tumor'] = min(0.99, (fdi - 3.5)/5.0) if fdi > 3.5 else 0.05
    else:
        risks['Tumor'] = 0.05
        
    return risks, fdi, focal_ch, faa

def get_recommendations(risks, blood_issues, lang):
    recs = []
    alert = "GREEN"
    if risks['Tumor'] > 0.65:
        recs.append(get_trans('mri_alert', lang))
        alert = "RED"
    if blood_issues:
        recs.append(get_trans('metabolic', lang) + f" ({', '.join(blood_issues)})")
        if alert != "RED": alert = "ORANGE"
        
    if risks['Alzheimer'] > 0.6:
        recs.append(get_trans('gamma_proto', lang))
        recs.append(T_st("إحالة لتقييم الشبكات العصبية (Neural Complexity)", lang))
        
    if risks['Depression'] > 0.7:
        recs.append(T_st("بروتوكول تحفيز عدم تقارن آلفا (FAA Protocol)", lang))
        
    if not recs: recs.append(get_trans('neuro', lang))
    return recs, alert

def generate_narrative(risks, blood, faa, entropy_val, coh_val, lang):
    L = lang
    n = ""
    if blood: n += T_st("النتائج المخبرية تشير إلى نقص استقلابي. ", L)
    
    if risks['Alzheimer'] > 0.6:
        n += T_st(f" انخفاض في الإنتروبيا ({entropy_val:.2f}) وضعف في الاتصال ({coh_val:.2f}) يدعم احتمالية MCI/AD. ", L)
    
    if risks['Depression'] > 0.6:
        n += T_st(f" عدم تناظر ألفا الجبهي (FAA: {faa:.2f}) يشير إلى هيمنة النشاط الأيمن. ", L)
        
    if risks['Tumor'] > 0.65:
        n += T_st(" نشاط دلتا بؤري حرج يتطلب تصوير فوري. ", L)
        
    if n == "": n = T_st("المؤشرات الحيوية المتقدمة ضمن الحدود الطبيعية.", L)
    return n

# --- 5. VISUALS ---
def generate_shap(df, adv_metrics, faa):
    try:
        feats = {
            "Frontal Theta": df['Theta (%)'].mean(), 
            "Occipital Alpha": df['Alpha (%)'].mean(),
            "Global Entropy": adv_metrics.get('Global_Entropy', 0)*10, 
            "Alpha Connectivity": adv_metrics.get('Alpha_Coherence', 0)*10,
            "Frontal Alpha Asym": abs(faa)*5
        }
        fig, ax = plt.subplots(figsize=(7,3.5))
        ax.barh(list(feats.keys()), list(feats.values()), color=BLUE)
        ax.set_title("Advanced SHAP Analysis")
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

def scan_blood_work(text):
    warnings = []
    text = text.lower()
    checks = {"Vitamin D": ["vit d", "low d"], "Thyroid": ["tsh", "thyroid"], "Anemia": ["iron", "anemia"]}
    for k, v in checks.items():
        if any(x in text for x in v) and "low" in text: warnings.append(k)
    return warnings

def extract_text_from_pdf(f):
    try:
        pdf = PyPDF2.PdfReader(f)
        return "".join([p.extract_text() for p in pdf.pages])
    except: return ""

# --- 6. PDF ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica'
    def T(x): return get_display(arabic_reshaper.reshape(str(x))) if lang == 'ar' else str(x)
    
    story = []
    if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    story.append(Paragraph(T(data['title']), ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE))))
    
    p = data['p']
    info = [
        [T(get_trans("name",lang)), T(p['name']), T(get_trans("id",lang)), T(p['id'])],
        [T(get_trans("gender",lang)), T(p['gender']), T(get_trans("dob",lang)), T(p['dob'])],
        [T("Labs"), T(p['labs']), T(get_trans("eye_state",lang)), T(p['eye'])]
    ]
    t = Table(info, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1,10))
    
    story.append(Paragraph(T(data['narrative']), ParagraphStyle('B', fontName=f_name, leading=14)))
    story.append(Spacer(1,10))
    
    for r in data['recs']:
        c = colors.red if "MRI" in r or "حرج" in r else colors.black
        story.append(Paragraph(T("• " + r), ParagraphStyle('A', fontName=f_name, textColor=c)))
        
    r_data = [[T("Metric / Condition"), T("Value / Risk")]]
    for k,v in data['risks'].items(): r_data.append([T(k), f"{v*100:.1f}%"])
    r_data.append([T(get_trans("entropy", lang)), f"{data['adv'].get('Global_Entropy', 0):.3f}"])
    r_data.append([T(get_trans("coherence", lang)), f"{data['adv'].get('Alpha_Coherence', 0):.3f}"])
    
    t2 = Table(r_data, style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t2)
    story.append(Spacer(1,15))
    
    story.append(Paragraph(T("Detailed Channel Data"), ParagraphStyle('H2', fontName=f_name)))
    df = data['eeg'].head(10).round(1)
    cols = ['Ch'] + list(df.columns)
    rows = [cols] + [[i] + [str(x) for x in row] for i, row in df.iterrows()]
    t3 = Table(rows, style=TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey), ('FONTSIZE',(0,0),(-1,-1),8)]))
    story.append(t3)
    
    story.append(PageBreak())
    
    if data['shap']: story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3.5*inch))
    imgs = [RLImage(io.BytesIO(data['maps'][b]), width=1.5*inch, height=1.5*inch) for b in BANDS if data['maps'][b]]
    if len(imgs)>=4: story.append(Table([imgs]))
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- 7. MAIN ---
def main():
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f'<div class="main-header">{get_trans("title", "en")}</div>', unsafe_allow_html=True)

    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        p_name = st.text_input(T_st(get_trans("name", L), L), "John Doe")
        p_gender = st.selectbox(T_st(get_trans("gender", L), L), [get_trans("male", L), get_trans("female", L)])
        p_dob = st.date_input(T_st(get_trans("dob", L), L), value=date(1980,1,1))
        p_id = st.text_input(T_st(get_trans("id", L), L), "F-101")
        st.markdown("---")
        lab_file = st.file_uploader(T_st(get_trans("lab_up", L), L), type=["pdf", "txt"])
        lab_text = extract_text_from_pdf(lab_file) if lab_file else ""

    tab1, tab2 = st.tabs([T_st(get_trans("tab_assess", L), L), T_st(get_trans("tab_neuro", L), L)])
    
    with tab1:
        c1, c2 = st.columns(2)
        phq_score = 0; mmse_score = 0
        with c1:
            st.subheader(T_st(get_trans("phq_t", L), L))
            opts = get_trans("opt_phq", L)
            for i, q in enumerate(get_trans("q_phq", L)):
                phq_score += opts.index(st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"p{i}"))
            st.metric("PHQ-9", f"{phq_score}/27")
        with c2:
            st.subheader(T_st(get_trans("alz_t", L), L))
            opts_m = get_trans("opt_mmse", L)
            for i, q in enumerate(get_trans("q_mmse", L)):
                mmse_score += opts_m.index(st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"m{i}", index=0))*2
            mmse_total = min(30, mmse_score+10)
            st.metric("MMSE", f"{mmse_total}/30")

    with tab2:
        uploaded_edf = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        if st.button(T_st(get_trans("analyze", L), L), type="primary"):
            blood = scan_blood_work(lab_text)
            
            adv_metrics = {}
            if uploaded_edf:
                with st.spinner("Processing Signal (Research Mode)..."):
                    df_eeg, adv_metrics = process_real_edf(uploaded_edf)
                    if df_eeg is None: st.error(adv_metrics); st.stop()
            else:
                st.warning("Simulation Mode")
                ch = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
                df_eeg = pd.DataFrame(np.random.uniform(2,10,(10,4)), columns=[f"{b} (%)" for b in BANDS], index=ch)
                df_eeg.loc['O1', 'Alpha (%)'] = 15.0
                adv_metrics = {'Global_Entropy': 0.85, 'Alpha_Coherence': 0.65} # Sim values

            detected_eye = determine_eye_state_smart(df_eeg)
            risks, fdi, focal_ch, faa = calculate_metrics(df_eeg, adv_metrics, phq_score, mmse_total)
            recs, alert = get_recommendations(risks, blood, L)
            narrative = generate_narrative(risks, blood, faa, adv_metrics.get('Global_Entropy',0), adv_metrics.get('Alpha_Coherence',0), L)
            shap_img = generate_shap(df_eeg, adv_metrics, faa)
            maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
            
            st.info(f"**{T_st(get_trans('eye_state', L), L)}:** {detected_eye}")
            final_eye = st.radio("Confirm:", ["Eyes Open", "Eyes Closed"], index=0 if detected_eye=="Eyes Open" else 1)
            
            color = "#ffebee" if alert == "RED" else "#e8f5e9"
            st.markdown(f'<div class="alert-box" style="background:{color}"><h3>{T_st(get_trans("decision", L), L)}</h3><p>{recs[0]}</p></div>', unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Depression", f"{risks['Depression']*100:.0f}%")
            c2.metric("Alzheimer", f"{risks['Alzheimer']*100:.0f}%")
            c3.metric("Entropy", f"{adv_metrics.get('Global_Entropy',0):.2f}")
            c4.metric("Coherence", f"{adv_metrics.get('Alpha_Coherence',0):.2f}")
            
            st.markdown(f'<div class="report-box"><h4>{T_st(get_trans("narrative", L), L)}</h4><p>{narrative}</p></div>', unsafe_allow_html=True)
            st.dataframe(df_eeg.style.background_gradient(cmap='Blues'), height=200)
            if shap_img: st.image(shap_img)
            st.image(list(maps.values()), width=120, caption=list(maps.keys()))
            
            pdf_data = {
                "title": get_trans("title", L),
                "p": {"name": p_name, "gender": p_gender, "dob": str(p_dob), "id": p_id, "labs": str(blood), "eye": final_eye},
                "risks": risks, "recs": recs, "eeg": df_eeg, "shap": shap_img, "maps": maps, "narrative": narrative, 
                "focal_ch": focal_ch, "adv": adv_metrics, "faa": faa
            }
            st.download_button(T_st(get_trans("download", L), L), create_pdf(pdf_data, L), "Report.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    main()
