# app_clinical_pro.py — NeuroEarly Pro v7 (CDSS Edition)
import os
import io
import sys
import tempfile
import json
import re
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage

# --- Library Checks ---
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SCIPY = False

try:
    import mne
    HAS_MNE = True
except ImportError: pass

try:
    import pyedflib
    HAS_PYEDF = True
except ImportError: pass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError: pass

try:
    from scipy.signal import welch, coherence
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError: pass

# --- Setup ---
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf" # For Arabic support in PDF
HEALTHY_EDF = ASSETS / "healthy_baseline.edf"

APP_TITLE = "NeuroEarly Pro — Clinical Assistant"
BLUE = "#003366" # Darker medical blue
RED = "#8B0000"
GREEN = "#006400"

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# --- Logic: Blood Work AI Analyzer ---
def analyze_blood_work(text: str) -> List[str]:
    """Simple NLP to detect keywords in blood test notes."""
    warnings = []
    text = text.lower()
    
    # Dictionary of keywords to look for
    checks = {
        "vitamin d": ["vit d", "vitamin d", "low d"],
        "iron/anemia": ["iron", "ferritin", "anemia", "hemoglobin low"],
        "thyroid": ["tsh", "t3", "t4", "thyroid"],
        "b12": ["b12", "cobalamin"],
        "inflammation": ["crp", "esr", "inflammation"]
    }
    
    for category, keywords in checks.items():
        for kw in keywords:
            if kw in text and ("low" in text or "high" in text or "deficien" in text):
                warnings.append(f"Potential {category} issue detected via keyword '{kw}'")
                break # Avoid double counting same category
                
    return warnings

# --- Logic: Clinical Recommendation Engine ---
def generate_recommendation(risk_score: float, phq: int, alz: int, blood_warnings: List[str]) -> Tuple[str, str]:
    """
    Returns: (Decision Title, Detailed Explanation)
    """
    decision = "PENDING"
    details = ""
    
    # Rule 1: Metabolic Priority
    if len(blood_warnings) > 0:
        decision = "⚠️ CAUTION: METABOLIC IMBALANCE"
        details += "Blood work indicates potential deficiencies (e.g., Vitamin D, Iron). "
        details += "Recommendation: Address metabolic/vitamin issues BEFORE starting intensive Neurofeedback/TMS. "
        details += f"Detected issues: {', '.join(blood_warnings)}."
        return decision, details

    # Rule 2: High Risk (QEEG + Symptoms)
    if risk_score > 0.6 or phq > 15 or alz > 10:
        decision = "✅ RECOMMENDATION: START TREATMENT"
        details += " biomarkers indicate significant deviation. "
        if phq > 15: details += "Severe depressive symptoms present. "
        if alz > 10: details += "Cognitive decline markers present. "
        details += "Protocol: Consider Alpha-Theta training or rTMS protocol targeting asymmetry."
        return decision, details

    # Rule 3: Moderate Risk
    if risk_score > 0.3:
        decision = "⚖️ MONITOR / MILD INTERVENTION"
        details += "Moderate deviation. Suggest 10 sessions of SMR training and lifestyle changes. Re-assess in 4 weeks."
        return decision, details

    # Rule 4: Healthy
    decision = "🟢 NO INTERVENTION NEEDED"
    details = "QEEG within normal limits. Symptoms are sub-clinical. Suggest psycho-education and sleep hygiene."
    return decision, details

# --- Helpers ---
def clamp01(x):
    return max(0.0, min(1.0, float(x))) if x else 0.0

def now_str():
    return datetime.now().strftime("%Y-%m-%d")

# --- EEG Processing (Same as before, optimized) ---
@st.cache_data
def read_edf_file(file_path: str):
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw.get_data(), {"sfreq": raw.info["sfreq"], "ch_names": raw.info["ch_names"]}, None
        except Exception as e: return None, None, str(e)
    return None, None, "MNE not installed"

@st.cache_data
def compute_band_powers(data, sf):
    if not HAS_SCIPY: return pd.DataFrame()
    n_ch = data.shape[0]
    rows = []
    for i in range(n_ch):
        f, Pxx = welch(data[i,:], fs=sf, nperseg=min(1024, data.shape[1]))
        mask_tot = (f>=1)&(f<=45)
        total = np.trapz(Pxx[mask_tot], f[mask_tot]) if mask_tot.any() else 0
        row = {}
        for b, (l,h) in BANDS.items():
            mask = (f>=l)&(f<h)
            val = np.trapz(Pxx[mask], f[mask]) if mask.any() else 0
            row[f"{b}_rel"] = val/total if total>0 else 0
        rows.append(row)
    return pd.DataFrame(rows, index=[f"Ch{i+1}" for i in range(n_ch)])

def get_topomap(vals, names, title):
    if not HAS_SCIPY: return None
    # (Simplified Topomap logic from previous code - kept for brevity)
    # Assume standard coordinates for simplicity in this snippet
    return None # Placeholder if scipy/mpl logic is lengthy, but in real code put the function here.

# --- PDF Generation with Table ---
def generate_clinical_pdf(data_dict: dict):
    if not HAS_REPORTLAB: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle("MedTitle", parent=styles["Heading1"], textColor=colors.HexColor(BLUE), fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle("MedH2", parent=styles["Heading2"], textColor=colors.HexColor(BLUE), fontSize=12, spaceBefore=10))
    styles.add(ParagraphStyle("Decision", parent=styles["Heading2"], textColor=colors.red, fontSize=14, borderPadding=5, borderColor=colors.red, borderWidth=1))
    
    story = []
    
    # 1. Header
    story.append(Paragraph(f"NeuroEarly Pro — Clinical Report", styles["MedTitle"]))
    story.append(Paragraph(f"Date: {now_str()}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # 2. Patient Demographics (Table)
    p = data_dict['patient']
    pt_data = [
        ["Patient Name:", p['name'], "DOB / Age:", f"{p['dob']} (approx)"],
        ["ID:", p['id'], "Gender:", "Not Specified"],
        ["History:", p['history'][:50]+"...", "Labs:", p['labs'][:50]+"..."]
    ]
    t_pt = Table(pt_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t_pt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke)]))
    story.append(t_pt)
    story.append(Spacer(1, 15))

    # 3. Clinical Decision (The "Brain" Output)
    dec, det = data_dict['recommendation']
    story.append(Paragraph("CLINICAL DECISION SUPPORT", styles["MedH2"]))
    story.append(Paragraph(f"<b>{dec}</b>", styles["Decision"]))
    story.append(Spacer(1,5))
    story.append(Paragraph(f"<i>Rationale:</i> {det}", styles["Normal"]))
    story.append(Spacer(1, 15))

    # 4. Band Power Table (The Missing Feature)
    if 'df_bands' in data_dict:
        story.append(Paragraph("Quantitative EEG (QEEG) - Relative Power", styles["MedH2"]))
        df = data_dict['df_bands']
        
        # Prepare data for PDF Table
        # Headers
        table_data = [["Channel", "Delta (%)", "Theta (%)", "Alpha (%)", "Beta (%)", "Gamma (%)"]]
        
        # Rows
        for idx, row in df.iterrows():
            r_data = [
                idx,
                f"{row.get('Delta_rel',0)*100:.1f}",
                f"{row.get('Theta_rel',0)*100:.1f}",
                f"{row.get('Alpha_rel',0)*100:.1f}",
                f"{row.get('Beta_rel',0)*100:.1f}",
                f"{row.get('Gamma_rel',0)*100:.1f}"
            ]
            table_data.append(r_data)
            
        # Style the big table
        t_bands = Table(table_data, repeatRows=1)
        t_bands.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BLUE)),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        story.append(t_bands)
    
    story.append(Spacer(1, 15))
    
    # 5. Disclaimer
    story.append(Paragraph("Disclaimer: This report is generated by an AI assistant and must be verified by a licensed physician.", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# --- Main Streamlit UI ---
def main():
    st.set_page_config(page_title="Clinical Assistant", layout="wide")
    
    # Sidebar Inputs (The "Console")
    with st.sidebar:
        st.image(str(LOGO_PATH) if LOGO_PATH.exists() else "https://via.placeholder.com/150", width=100)
        st.title("Patient File")
        
        p_name = st.text_input("Patient Name", "John Doe")
        p_id = st.text_input("File ID", "A-100")
        p_dob = st.number_input("Birth Year", 1940, 2024, 1980)
        
        st.markdown("---")
        st.subheader("Clinical Context")
        p_history = st.text_area("Underlying Conditions", "Diabetes type 2, Hypertension...")
        p_labs = st.text_area("Blood Lab Results (Copy/Paste)", "Vitamin D: 15 ng/mL (Low), Iron: Normal, B12: Borderline")
        
        uploaded_file = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        
        run_analysis = st.button("Run Clinical Analysis", type="primary")

    # Main Screen
    st.title(f"⚕️ NeuroEarly Pro - Clinical Dashboard")
    st.markdown(f"**Attending:** Dr. Supervisor | **Patient:** {p_name} ({2025-p_dob} y/o)")
    
    if run_analysis:
        if not uploaded_file:
            st.warning("⚠️ No EEG file uploaded. Using simulation mode if configured.")
            # In real app, stop here. For demo, we might generate fake data.
            return

        # 1. Save File
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
            tf.write(uploaded_file.getvalue())
            tname = tf.name
        
        # 2. Analyze EEG
        data, meta, err = read_edf_file(tname)
        if err:
            st.error(f"EEG Error: {err}")
            return
            
        df_bands = compute_band_powers(data, meta['sfreq'])
        
        # 3. Analyze Clinical Data (The "AI")
        blood_issues = analyze_blood_work(p_labs)
        
        # 4. Calculate Scores (Simulated for this snippet)
        # In real code, this comes from the questionnaire logic in previous versions
        phq_score = 18 # High depression (example)
        alz_score = 5  # Low cognitive decline
        theta_alpha = df_bands['Theta_rel'].mean() / df_bands['Alpha_rel'].mean()
        risk_ml = 0.75 if theta_alpha > 1.2 else 0.3 # Simple heuristic
        
        # 5. Make Decision
        decision_title, decision_details = generate_recommendation(risk_ml, phq_score, alz_score, blood_issues)
        
        # --- Display Dashboard ---
        
        # Top Row: Decision
        st.info(f"### 🏥 SYSTEM CONCLUSION: {decision_title}")
        st.write(decision_details)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩸 Lab Analysis")
            if blood_issues:
                for i in blood_issues:
                    st.error(f"• {i}")
            else:
                st.success("No obvious metabolic deficiencies detected in text.")
                
            st.subheader("🧠 QEEG Summary")
            st.metric("Theta/Alpha Ratio", f"{theta_alpha:.2f}", delta_color="inverse")
            st.metric("Global Delta Power", f"{df_bands['Delta_rel'].mean()*100:.1f}%")

        with col2:
            st.subheader("📊 Band Power Table")
            # Display the dataframe as a nice interactive table
            st.dataframe(df_bands.style.background_gradient(cmap='Blues', subset=['Delta_rel', 'Theta_rel', 'Alpha_rel']), height=300)

        # --- Generate PDF ---
        report_data = {
            "patient": {"name": p_name, "id": p_id, "dob": p_dob, "history": p_history, "labs": p_labs},
            "recommendation": (decision_title, decision_details),
            "df_bands": df_bands,
            "risk": risk_ml
        }
        
        pdf_bytes = generate_clinical_pdf(report_data)
        if pdf_bytes:
            st.download_button("📥 Download Clinical Report (PDF)", pdf_bytes, "clinical_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
