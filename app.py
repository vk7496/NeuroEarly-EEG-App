# app.py — NeuroEarly Pro v30 (Strictly English & MMSE Scoring Refinement)
import os
import io
import tempfile
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import streamlit as st
import PyPDF2
import mne 

# PDF generation (ONLY using standard ReportLab/Helvetica/Arial for English)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v30 (English)", layout="wide", page_icon="🧠")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")

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


# --- 2. LOCALIZATION (Simplified English-Only Strings) ---
# All strings are now hardcoded English
L_STRINGS = {
    "title": "NeuroEarly Pro: Clinical AI Assistant", "subtitle": "Advanced Decision Support System",
    "p_info": "Patient Demographics", "name": "Full Name", "gender": "Gender", "dob": "Date of Birth", "id": "File ID",
    "male": "Male", "female": "Female", "lab_up": "Upload Lab Report (PDF)",
    "tab_assess": "1. Clinical Assessments", "tab_neuro": "2. Neuro-Analysis (EEG)",
    "analyze": "RUN DIAGNOSIS", "decision": "CLINICAL DECISION",
    "mri_alert": "🚨 CRITICAL: FOCAL LESION DETECTED -> REFER FOR MRI/CT (IMMEDIATE ACTION)",
    "metabolic": "⚠️ Metabolic Correction Needed", "neuro": "✅ Proceed with Protocol",
    "download": "Download Doctor's Report", "eye_state": "Eye State (AI Detected)",
    "narrative": "Automated Clinical Narrative",
    "phq_t": "Depression Screening (PHQ-9)", "alz_t": "Cognitive Screening (MMSE)",
    "q_phq": ["Little interest", "Feeling down", "Sleep issues", "Tiredness", "Appetite", "Failure", "Concentration", "Slowness", "Self-harm"],
    "opt_phq": ["Not at all", "Several days", "More than half", "Nearly every day"],
    "q_mmse": ["Orientation", "Registration", "Attention", "Recall", "Language"],
    "opt_mmse": ["Incorrect", "Partial", "Correct"],
    "doc_data_title": "Detailed QEEG Data Table (Relative Power)",
    "doc_recs_title": "Doctor's Guidance and Protocol",
    "delta_band": "Delta Band", "theta_band": "Theta Band", 
    "alpha_band": "Alpha Band", "beta_band": "Beta Band",
    "tumor_risk": "Tumor Risk", "depression": "Depression", "alzheimer": "Alzheimer's/Dementia", "adhd": "ADHD/Attention"
}

# --- 3. SIGNAL PROCESSING (V28 Logic Preserved) ---
def process_real_edf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        
        # --- Channel Whitelisting: ONLY standard 10-20 channels ---
        STANDARD_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 
                             'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2', 'EOG'] 
        
        eeg_channels = [ch for ch in raw.ch_names if ch.upper() in [s.upper() for s in STANDARD_CHANNELS]]
        raw.pick_channels(eeg_channels, ordered=True)
        
        sf = raw.info['sfreq']
        if sf > 100: raw.notch_filter(np.arange(50, sf/2, 50), verbose=False)
        raw.filter(0.5, 45.0, verbose=False)
        
        data = raw.get_data()
        ch_names = raw.ch_names
        
        if not ch_names:
            return None, "Error: No standard EEG channels found after filtering."
            
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
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return None, str(e)

# --- 4. LOGIC & METRICS (V30: Alzheimer Risk Refinement) ---
def determine_eye_state_smart(df_bands):
    if df_bands.empty: return "N/A"
    occ_channels = [ch for ch in df_bands.index if any(x in ch for x in ['O1','O2','P3','P4'])]
    if occ_channels:
        if df_bands.loc[occ_channels, 'Alpha (%)'].median() > 12.0: return "Eyes Closed"
    if df_bands['Alpha (%)'].median() > 10.0: return "Eyes Closed"
    return "Eyes Open"

def calculate_metrics(eeg_df, phq, mmse):
    risks = {}
    tbr = 0
    df_eeg = eeg_df.copy()
    
    if 'Theta (%)' in df_eeg and 'Beta (%)' in df_eeg and not df_eeg.empty:
        tbr = df_eeg['Theta (%)'].median() / (df_eeg['Beta (%)'].median() + 0.01)
        df_eeg['TBR'] = df_eeg['Theta (%)'] / (df_eeg['Beta (%)'] + 0.01)
    
    # Depression Risk: Based on PHQ-9 (Max 27)
    risks['Depression'] = min(0.99, (phq / 27.0) * 0.6 + 0.1)
    
    # Alzheimer's Risk: Based on MMSE (Max 30) - Adjusted for smoother risk increase.
    # Max risk is now 75% for 0/30 MMSE, making the high risk less extreme.
    mmse_normalized_error = (30 - mmse) / 30.0 # 0 for 30/30, 1 for 0/30
    risks['Alzheimer'] = min(0.99, mmse_normalized_error * 0.65 + 0.1) # 10% base, max 75%
    
    fdi = 0
    focal_ch = "N/A"
    
    if 'Delta (%)' in df_eeg and not df_eeg.empty:
        stable_channel_names = ['C3', 'C4', 'P3', 'P4', 'Cz', 'Pz']
        stable_channels = [ch for ch in df_eeg.index if ch in stable_channel_names]
        test_channels = df_eeg.index.tolist()
        
        if stable_channels and test_channels and len(stable_channels) >= 3:
            deltas_test = df_eeg.loc[test_channels, 'Delta (%)']
            deltas_stable = df_eeg.loc[stable_channels, 'Delta (%)']
            max_delta = deltas_test.max()
            median_delta_stable = deltas_stable.median()
            fdi = max_delta / (median_delta_stable + 0.01)
            focal_ch = deltas_test.idxmax()
            
        # Tumor Risk (FDI > 4.0 is suspicious)
        risk_calc = max(0.05, (fdi - 4.0) / 10.0) 
        risks['Tumor'] = min(0.99, risk_calc) if fdi > 4.0 else 0.05
    else:
        risks['Tumor'] = 0.05
    
    # ADHD Risk: Based on TBR
    risks['ADHD'] = min(0.99, (tbr / 3.0)) if tbr > 1.5 else 0.1
    
    if 'Alpha (%)' in df_eeg and not df_eeg.empty:
        mean_alpha = df_eeg['Alpha (%)'].mean()
        std_alpha = df_eeg['Alpha (%)'].std() + 0.01
        df_eeg['Alpha Z'] = (df_eeg['Alpha (%)'] - mean_alpha) / std_alpha
        
    return risks, fdi, tbr, df_eeg, focal_ch

def scan_blood_work(text):
    warnings = []
    text = text.lower()
    checks = {"Vitamin D": ["vit d", "low d"], "Thyroid": ["tsh", "thyroid"], "Anemia": ["iron", "anemia"]}
    for k, v in checks.items():
        if any(x in text for x in v) and "low" in text: warnings.append(k)
    return warnings

def get_recommendations(risks, blood_issues):
    recs = []
    alert = "GREEN"
    
    if risks['Tumor'] > 0.65:
        recs.append(L_STRINGS['mri_alert'])
        alert = "RED"
    
    if blood_issues:
        recs.append(L_STRINGS['metabolic'] + f" ({', '.join(blood_issues)})")
        if alert != "RED": alert = "ORANGE"
        
    if risks['Depression'] > 0.7: recs.append("Psychiatry Referral (Therapy/rTMS Protocol)")
    if risks['ADHD'] > 0.6: recs.append("Neurofeedback (Attention Protocol)")
    if risks['Alzheimer'] > 0.6: recs.append("Neurology Referral (Cognitive Evaluation & Medication)")
    
    if not recs: recs.append(L_STRINGS['neuro'])
    return recs, alert

def generate_narrative(risks, blood, tbr, fdi, focal_ch):
    n = ""
    if blood: n += f"Lab results indicate metabolic deficiencies ({', '.join(blood)}). "
    if risks['Tumor'] > 0.65: n += f" CRITICAL: Focal Delta asymmetry (FDI: {fdi:.2f} at {focal_ch}). Lesion risk must be ruled out by MRI/CT. "
    if risks['ADHD'] > 0.6: n += f" High Theta/Beta Ratio ({tbr:.2f}) suggests an attentional deficit. "
    if risks['Depression'] > 0.7: n += f" High Depression risk ({risks['Depression']*100:.0f}%) observed via PHQ-9. "
    if risks['Alzheimer'] > 0.6: n += f" High Alzheimer/Dementia risk ({risks['Alzheimer']*100:.0f}%) observed via MMSE. "
    if n == "": n = "Neurophysiological profile is within the normal range, and no critical immediate action is required."
    return n

# --- 5. VISUALS (Remains the same) ---
def generate_shap(df):
    try:
        if df.empty: return None
        feats = {
            "Frontal Theta": df['Theta (%)'].mean(), "Occipital Alpha": df['Alpha (%)'].mean(),
            "TBR": df['TBR'].mean() if 'TBR' in df.columns else df['Theta (%)'].mean() / (df['Beta (%)'].mean() + 0.01), 
            "Delta Power": df['Delta (%)'].mean()
        }
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(list(feats.keys()), list(feats.values()), color=BLUE)
        ax.set_title("SHAP Analysis (Feature Importance)")
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except: return None

def generate_topomap(df, band):
    if df.empty or f'{band} (%)' not in df.columns: return None
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

# --- 6. PDF (V30: English-Only Formatting) ---
def create_pdf(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Standard Paragraph styles for English (left-aligned)
    P_NORMAL = styles['Normal']
    P_HEADER = styles['Heading3']
    
    # Helper for table cells (simple string return for English)
    def T_p(text): return str(text) 

    story = []
    
    # Title
    if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    story.append(Paragraph(data['title'], styles['Title'])) 
    story.append(Spacer(1,5))
    
    # Patient Info Table
    p = data['p']
    info = [
        [T_p(L_STRINGS["name"]), T_p(p['name']), T_p(L_STRINGS["id"]), T_p(p['id'])],
        [T_p(L_STRINGS["gender"]), T_p(p['gender']), T_p(L_STRINGS["dob"]), T_p(p['dob'])],
        [T_p("Eye State"), T_p(p['eye']), T_p("Labs"), T_p(p['labs'])]
    ]
    t = Table(info, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(t)
    story.append(Spacer(1,10))
    
    # Narrative
    story.append(Paragraph(data['narrative'], P_NORMAL))
    story.append(Spacer(1,10))
    
    # --- Doctor's Guidance and Protocol ---
    story.append(Paragraph(L_STRINGS["doc_recs_title"], ParagraphStyle('RecTitle', fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor(BLUE))))
    story.append(Spacer(1,5))
    
    for r in data['recs']:
        c = colors.red if "CRITICAL" in r else colors.black
        # Use a list style (•) for protocol/recommendations
        story.append(Paragraph(f"• {r}", 
                               ParagraphStyle(name='Rec', fontName='Helvetica', textColor=c, leading=16, leftIndent=20)))
    story.append(Spacer(1,10))
    
    # Risks Table
    r_data = [[T_p("Condition"), T_p("Risk")]]
    risk_map = {'Depression': L_STRINGS['depression'], 'Alzheimer': L_STRINGS['alzheimer'], 'Tumor': L_STRINGS['tumor_risk'], 'ADHD': L_STRINGS['adhd']}
    for k,v in data['risks'].items(): 
        if k in risk_map: r_data.append([T_p(risk_map[k]), T_p(f"{v*100:.1f}%")])
    r_data.append([T_p("TBR"), T_p(f"{data['tbr']:.2f}")])
    r_data.append([T_p("FDI Channel"), T_p(data['focal_ch'])])
    t2 = Table(r_data, style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(t2)
    story.append(Spacer(1,20))

    # --- Detailed QEEG Data Table ---
    story.append(Paragraph(L_STRINGS["doc_data_title"], ParagraphStyle('DataTitle', fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor(BLUE))))
    story.append(Spacer(1,5))
    
    df_eeg = data['eeg'].copy().round(2)
    if 'TBR' not in df_eeg.columns: df_eeg['TBR'] = 0.0
    if 'Alpha Z' not in df_eeg.columns: df_eeg['Alpha Z'] = 0.0

    cols_to_include = ['Delta (%)', 'Theta (%)', 'Alpha (%)', 'Beta (%)', 'TBR', 'Alpha Z']
    df_pdf = df_eeg[[c for c in cols_to_include if c in df_eeg.columns]]

    headers = ["Ch", "Delta %", "Theta %", "Alpha %", "Beta %", "TBR", "Alpha Z"]
    table_data = [[T_p(h) for h in headers]]
    
    for ch, row in df_pdf.iterrows():
        table_row = [T_p(str(ch))] + [T_p(f"{val:.2f}") for val in row.values]
        table_data.append(table_row)

    t_eeg = Table(table_data)
    t_eeg.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), 
                               ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)])) 
    story.append(t_eeg)
    
    story.append(PageBreak())
    
    # --- Topomap Layout with Labels ---
    
    # Images row
    maps = data['maps']
    imgs = [RLImage(io.BytesIO(maps[b]), width=1.5*inch, height=1.5*inch) for b in BANDS if maps[b]]
    # Labels row 
    labels = [Paragraph(L_STRINGS[f"{b.lower()}_band"], P_NORMAL) for b in BANDS if maps[b]]
    
    if len(imgs) >= 4:
        story.append(Paragraph("Topographic Power Maps", P_HEADER))
        story.append(Spacer(1,5))
        topo_table = Table([imgs, labels], rowHeights=[1.7*inch, 0.3*inch])
        topo_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'),
                                        ('VALIGN', (0,0), (-1,0), 'TOP')]))
        story.append(topo_table)
    
    # Add SHAP chart
    if data['shap']: 
        story.append(Spacer(1, 20))
        story.append(Paragraph("AI Explainability (SHAP Feature Importance)", P_HEADER))
        story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=3*inch))
    
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
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f'<div class="main-header">{L_STRINGS["title"]}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header(L_STRINGS["p_info"])
        p_name = st.text_input(L_STRINGS["name"], "John Doe")
        p_gender = st.selectbox(L_STRINGS["gender"], [L_STRINGS["male"], L_STRINGS["female"]])
        p_dob = st.date_input(L_STRINGS["dob"], value=date(1980,1,1))
        p_id = st.text_input(L_STRINGS["id"], "F-101")
        
        st.markdown("---")
        lab_file = st.file_uploader(L_STRINGS["lab_up"], type=["pdf", "txt"])
        lab_text = extract_text_from_pdf(lab_file) if lab_file else ""

    tab1, tab2 = st.tabs([L_STRINGS["tab_assess"], L_STRINGS["tab_neuro"]])
    
    with tab1:
        c_q1, c_q2 = st.columns(2)
        phq_score = 0
        mmse_score = 0
        
        with c_q1:
            st.subheader(L_STRINGS["phq_t"])
            opts = L_STRINGS["opt_phq"]
            for i, q in enumerate(L_STRINGS["q_phq"]):
                ans = st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"phq_{i}")
                phq_score += opts.index(ans)
            st.metric("PHQ-9 Score", f"{phq_score}/27")
            
        with c_q2:
            st.subheader(L_STRINGS["alz_t"])
            opts_m = L_STRINGS["opt_mmse"]
            
            # MMSE Simplified Scoring (Max 30)
            score_acc = 0
            for i, q in enumerate(L_STRINGS["q_mmse"]):
                ans_index = st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"mmse_{i}", index=2)
                # Correct (index 2) = 6 points, Partial (index 1) = 3 points, Incorrect (index 0) = 0 points
                score_acc += ans_index * 3
            mmse_total = min(30, score_acc)
            st.metric("MMSE Score", f"{int(mmse_total)}/30")

    with tab2:
        uploaded_edf = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        
        if st.button(L_STRINGS["analyze"], type="primary"):
            blood = scan_blood_work(lab_text)
            
            if uploaded_edf:
                with st.spinner("Processing Real Signal..."):
                    df_eeg, err = process_real_edf(uploaded_edf)
                    if err: st.error(err); st.stop()
            else:
                st.warning("Simulation Mode (No EDF uploaded)")
                ch = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
                # Simulation Data 
                data_sim = np.random.uniform(2, 12, (10, 4))
                df_eeg = pd.DataFrame(data_sim, columns=[f"{b} (%)" for b in BANDS], index=ch)
                df_eeg.loc['O1', 'Alpha (%)'] = 15.0 # High Alpha for Eyes Closed simulation
            
            detected_eye = determine_eye_state_smart(df_eeg)
            risks, fdi, tbr, df_eeg, focal_ch = calculate_metrics(df_eeg, phq_score, mmse_total)
            recs, alert = get_recommendations(risks, blood)
            narrative = generate_narrative(risks, blood, tbr, fdi, focal_ch)
            shap_img = generate_shap(df_eeg)
            maps = {b: generate_topomap(df_eeg, b) for b in BANDS}
            
            st.info(f"**{L_STRINGS['eye_state']}:** {detected_eye}")
            final_eye = detected_eye
            
            color = "#ffebee" if alert == "RED" else ("#fffde7" if alert == "ORANGE" else "#e8f5e9")
            st.markdown(f'<div class="alert-box" style="background:{color}"><h3>{L_STRINGS["decision"]}</h3><p>{recs[0]}</p></div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric(L_STRINGS["depression"], f"{risks['Depression']*100:.0f}%")
            c2.metric(L_STRINGS["alzheimer"], f"{risks['Alzheimer']*100:.0f}%")
            c3.metric(L_STRINGS["tumor_risk"], f"{risks['Tumor']*100:.0f}%", f"FDI: {fdi:.2f} @ {focal_ch}") 
            
            st.markdown(f'<div class="report-box"><h4>{L_STRINGS["narrative"]}</h4><p>{narrative}</p></div>', unsafe_allow_html=True)
            st.dataframe(df_eeg.style.background_gradient(cmap='Blues'), height=200)
            
            if shap_img: st.image(shap_img, caption="SHAP Analysis (Feature Importance)")
            st.image(list(maps.values()), width=120, caption=[L_STRINGS[f"{b.lower()}_band"] for b in BANDS])
            
            pdf_data = {
                "title": L_STRINGS["title"],
                "p": {"name": p_name, "gender": p_gender, "dob": str(p_dob), "id": p_id, "labs": str(blood), "eye": final_eye},
                "risks": risks, "tbr": tbr, "recs": recs, "eeg": df_eeg, "shap": shap_img, "maps": maps, "narrative": narrative, "focal_ch": focal_ch
            }
            st.download_button(L_STRINGS["download"], create_pdf(pdf_data), "Report.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    main()
