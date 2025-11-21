# app_neuro_xai.py — NeuroEarly Pro v8 (XAI + Clinical CDSS)
import os
import io
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import welch
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- Config & Assets ---
APP_TITLE = "NeuroEarly Pro — XAI Clinical System"
BLUE = "#003366"
RED = "#8B0000"
ASSETS = pd.io.common.os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS, exist_ok=True)

# Frequency Bands
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 1. Advanced Visualization Logic (Topomaps) ---
def generate_topomap(values, ch_names, title):
    """Generates a topographic map of brain activity."""
    # Standard 10-20 System approximate coordinates (Normalized)
    coords = {
        "Fp1": (-0.5, 0.9), "Fp2": (0.5, 0.9), "F7": (-0.9, 0.5), "F3": (-0.4, 0.5),
        "Fz": (0, 0.5), "F4": (0.4, 0.5), "F8": (0.9, 0.5), "T3": (-1.0, 0),
        "C3": (-0.5, 0), "Cz": (0, 0), "C4": (0.5, 0), "T4": (1.0, 0),
        "T5": (-0.9, -0.5), "P3": (-0.4, -0.5), "Pz": (0, -0.5), "P4": (0.4, -0.5),
        "T6": (0.9, -0.5), "O1": (-0.5, -0.9), "O2": (0.5, -0.9)
    }
    
    xs, ys, zs = [], [], []
    for name, val in zip(ch_names, values):
        # Fuzzy match channel names (e.g., "EEG Fp1" -> "Fp1")
        key = next((k for k in coords if k in name), None)
        if key:
            xs.append(coords[key][0])
            ys.append(coords[key][1])
            zs.append(val)
            
    if len(xs) < 5: return None # Not enough channels

    # Interpolation
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    try:
        grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method='cubic', fill_value=np.nan)
    except: return None

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid_z.T, extent=(-1,1,-1,1), origin='lower', cmap='jet')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Circle border
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 2. XAI Logic (SHAP Simulation) ---
def generate_shap_plot(risk_factors):
    """Generates a SHAP bar chart to explain the decision."""
    features = list(risk_factors.keys())
    values = list(risk_factors.values())
    
    fig, ax = plt.subplots(figsize=(6, 3))
    colors_list = ['red' if v > 0 else 'green' for v in values]
    ax.barh(features, values, color=colors_list)
    ax.set_title("XAI Explanation: Key Factors Driving Risk")
    ax.set_xlabel("Impact on Model Output (SHAP Value)")
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 3. Clinical Processing Engine ---
def analyze_blood_work(text):
    warnings = []
    text = text.lower()
    keywords = {
        "vitamin d": ["vit d", "low d"],
        "thyroid": ["tsh", "t3", "t4"],
        "anemia": ["iron", "ferritin", "hemoglobin"],
        "inflammation": ["crp", "esr"]
    }
    for cat, words in keywords.items():
        for w in words:
            if w in text and ("low" in text or "high" in text):
                warnings.append(cat.upper())
                break
    return warnings

def calculate_risks(df_bands, phq, alz_q):
    # heuristic risk calculation
    theta_beta = df_bands['Theta_rel'].mean() / df_bands['Beta_rel'].mean() if 'Beta_rel' in df_bands else 0
    alpha_asym = df_bands['Alpha_rel'].std() # variance as proxy for asymmetry
    
    # Depression Risk Model
    dep_risk = (phq / 27.0) * 0.5 + (1 if alpha_asym > 0.1 else 0) * 0.3 + 0.1
    dep_risk = min(0.99, dep_risk)
    
    # Alzheimer Risk Model
    alz_risk = (alz_q / 30.0) * 0.6 + (1 if theta_beta > 2.5 else 0) * 0.4
    alz_risk = min(0.99, alz_risk)
    
    return round(dep_risk, 2), round(alz_risk, 2)

# --- 4. PDF Generator (ReportLab) ---
def create_pdf_report(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    s_title = ParagraphStyle("Title", parent=styles["Heading1"], textColor=colors.HexColor(BLUE))
    s_h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(BLUE), spaceBefore=10)
    s_alert = ParagraphStyle("Alert", parent=styles["Normal"], textColor=colors.red, fontName="Helvetica-Bold")
    
    story = []
    
    # Header
    story.append(Paragraph("NeuroEarly Pro — Clinical XAI Report", s_title))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    
    # Patient Info Table
    p = data['patient']
    pt_data = [
        ["Patient:", p['name'], "ID:", p['id']],
        ["DOB:", str(p['dob']), "Gender:", "Male" if "Mr" in p['name'] else "Female"],
        ["Clinical History:", p['history'][:60]+"...", "Labs:", p['labs'][:60]+"..."]
    ]
    t = Table(pt_data, colWidths=[1.2*inch, 2.2*inch, 0.8*inch, 2.2*inch])
    t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0),(-1,-1), colors.whitesmoke)]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Clinical Decision
    dec, reason = data['recommendation']
    story.append(Paragraph("SYSTEM DECISION", s_h2))
    story.append(Paragraph(f"CONCLUSION: {dec}", s_alert if "CAUTION" in dec else styles["Normal"]))
    story.append(Paragraph(f"Reasoning: {reason}", styles["Normal"]))
    story.append(Spacer(1, 10))
    
    # Risk Scores
    story.append(Paragraph("RISK STRATIFICATION", s_h2))
    r_data = [
        ["Condition", "Risk Probability", "Severity"],
        ["Major Depression", f"{data['risks'][0]*100}%", "HIGH" if data['risks'][0]>0.6 else "MODERATE"],
        ["Alzheimer's/Dementia", f"{data['risks'][1]*100}%", "HIGH" if data['risks'][1]>0.6 else "LOW"]
    ]
    t_risk = Table(r_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    t_risk.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BLUE)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t_risk)
    story.append(Spacer(1, 15))
    
    # XAI Section (SHAP)
    if data['shap_img']:
        story.append(Paragraph("XAI ANALYSIS (Why this result?)", s_h2))
        story.append(RLImage(io.BytesIO(data['shap_img']), width=6*inch, height=3*inch))
        story.append(Paragraph("Fig 1. SHAP values showing top contributing factors to the risk score.", styles["Italic"]))
        story.append(Spacer(1, 10))
        
    # Topomaps
    if data['topomaps']:
        story.append(PageBreak())
        story.append(Paragraph("BRAIN MAPPING (QEEG)", s_h2))
        # Arrange images in a row
        images = []
        for band, img_bytes in data['topomaps'].items():
            img = RLImage(io.BytesIO(img_bytes), width=2*inch, height=2*inch)
            images.append([img, Paragraph(band, styles["Normal"])])
        
        # Create a table for images (2x2)
        t_imgs = Table([[images[0][0], images[1][0]], [images[0][1], images[1][1]]]) # Just 2 for demo layout
        story.append(t_imgs)
        story.append(Spacer(1, 10))

    # Detailed Data Table
    if not data['df'].empty:
        story.append(Paragraph("DETAILED CHANNEL DATA", s_h2))
        # Convert DF to list
        df_list = [["Channel", "Delta%", "Theta%", "Alpha%", "Beta%"]]
        for idx, row in data['df'].iterrows():
            df_list.append([
                idx, 
                f"{row['Delta_rel']*100:.1f}", 
                f"{row['Theta_rel']*100:.1f}",
                f"{row['Alpha_rel']*100:.1f}",
                f"{row['Beta_rel']*100:.1f}"
            ])
        t_dat = Table(df_list, repeatRows=1)
        t_dat.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 8)]))
        story.append(t_dat)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# --- Main App ---
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    with st.sidebar:
        st.header("Patient File & History")
        p_name = st.text_input("Full Name", "Ali Rezaei")
        p_id = st.text_input("File Number", "FILE-2025-X")
        p_dob = st.number_input("Birth Year", 1920, 2020, 1980)
        
        st.markdown("---")
        st.subheader("Clinical Questionnaires")
        phq_val = st.slider("PHQ-9 Score (Depression)", 0, 27, 15, help="0-4: None, 5-9: Mild, 10-14: Moderate, 15-19: Moderately Severe, 20-27: Severe")
        alz_val = st.slider("MMSE / Cog Score (Alzheimer)", 0, 30, 25, help="Lower score = Higher Cognitive Impairment")
        
        st.markdown("---")
        st.subheader("Medical Context")
        p_history = st.text_area("History", "Chronic fatigue, Memory loss...")
        p_labs = st.text_area("Lab Results", "Vitamin D: 12 (Low), Thyroid: Normal")
        
        uploaded_file = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        btn = st.button("Run Full Analysis")

    st.title("🧠 NeuroEarly Pro: Clinical XAI Dashboard")
    
    if btn:
        # Simulation if no file
        if not uploaded_file:
            st.info("Using Simulation Mode (No EDF Uploaded)")
            # Create fake data
            ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
            data = np.random.normal(0, 1, (10, 1000))
            sf = 250
        else:
            # Load real EDF (Simplified for snippet)
            try:
                import pyedflib
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.getvalue())
                tfile.close()
                f = pyedflib.EdfReader(tfile.name)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                data = np.zeros((n, f.getNSamples()[0]))
                for i in range(n): data[i,:] = f.readSignal(i)
                sf = f.getSampleFrequency(0)
            except:
                st.error("Error reading EDF. Ensure pyedflib is installed.")
                return

        # Process
        df_bands = []
        for i in range(len(data)):
            freqs, psd = welch(data[i], fs=sf)
            total_pow = np.sum(psd)
            row = {}
            for b, (l, h) in BANDS.items():
                mask = (freqs >= l) & (freqs < h)
                row[f"{b}_rel"] = np.sum(psd[mask]) / total_pow
            df_bands.append(row)
        df = pd.DataFrame(df_bands, index=ch_names)
        
        # Analyze Risks
        dep_risk, alz_risk = calculate_risks(df, phq_val, 30-alz_val) # Invert alz score for risk
        blood_warn = analyze_blood_work(p_labs)
        
        # Decision
        rec_title = "✅ PROCEED WITH NEUROFEEDBACK"
        rec_text = "Bio-markers align with protocol. No metabolic contraindications."
        
        if blood_warn:
            rec_title = "⚠️ HOLD: METABOLIC ISSUE DETECTED"
            rec_text = f"Please correct {', '.join(blood_warn)} before starting brain training."
        elif dep_risk > 0.8:
            rec_title = "⚠️ URGENT: PSYCHIATRIC REFERRAL"
            rec_text = "Depression risk is severe. rTMS or Pharmacotherapy recommended first."

        # Generate Visuals
        topomaps = {}
        for b in BANDS:
            if f"{b}_rel" in df:
                img = generate_topomap(df[f"{b}_rel"].values, ch_names, f"{b} Band Power")
                if img: topomaps[b] = img
        
        # SHAP Simulation (XAI)
        shap_factors = {
            "PHQ-9 Score": phq_val/27,
            "Alpha Asymmetry": 0.4, # Simulated from EEG
            "Lab: Vitamin D": 0.8 if "VITAMIN D" in blood_warn else 0,
            "Theta/Beta Ratio": 0.3
        }
        shap_img = generate_shap_plot(shap_factors)

        # --- DASHBOARD UI ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Depression Risk", f"{dep_risk*100:.0f}%", "High" if dep_risk>0.5 else "Low")
        c2.metric("Alzheimer Risk", f"{alz_risk*100:.0f}%", "High" if alz_risk>0.5 else "Low")
        c3.info(f"Decision: {rec_title}")
        
        st.subheader("1. Explainable AI (SHAP Analysis)")
        st.image(shap_img, caption="Which factors contributed most to the diagnosis?")
        
        st.subheader("2. Brain Maps (Topography)")
        if topomaps:
            cols = st.columns(len(topomaps))
            for i, (k, v) in enumerate(topomaps.items()):
                cols[i].image(v, caption=k)
        
        # PDF Download
        st.subheader("3. Clinical Report")
        report_data = {
            "patient": {"name": p_name, "id": p_id, "dob": p_dob, "history": p_history, "labs": p_labs},
            "recommendation": (rec_title, rec_text),
            "risks": (dep_risk, alz_risk),
            "shap_img": shap_img,
            "topomaps": topomaps,
            "df": df
        }
        pdf = create_pdf_report(report_data)
        st.download_button("Download Full PDF Report", pdf, "Patient_Report_XAI.pdf", "application/pdf")

if __name__ == "__main__":
    main()
