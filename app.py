#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app.py — NeuroEarly Pro (v5 -> final)
# Full bilingual (English default / Arabic optional RTL via Amiri),
# EDF upload -> band powers -> heatmaps -> SHAP -> PDF report (bilingual)

import os
import io
import sys
import json
import math
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date

from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from PIL import Image

import streamlit as st

# Optional heavy libs
HAS_MNE = False
HAS_PYEDF = False
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_ARABIC = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# Arabic support
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

# Paths & constants
ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"  # put font here
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"  # ensure file exists in repo

# Frequency bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# Utilities
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def maybe_reshape_ar(text: str) -> str:
    if HAS_ARABIC:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    return text

# EDF reading: save BytesIO to temp file and read with mne or pyedflib
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    Reads uploaded EDF (Streamlit UploadedFile).
    Returns (raw, msg) where raw is mne Raw object if possible, else dict with numpy signals.
    """
    if not uploaded:
        return None, "No file uploaded"
    # Write uploaded bytes to temp file on disk — MNE and pyedflib accept filename
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
            tf.write(uploaded.getvalue())
            temp_path = tf.name
        # prefer mne
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
                raw.info['temp'] = True  # safe storage in info if needed (not persistent)
                return raw, None
            except Exception as e:
                # try pyedflib fallback
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(temp_path)
                n = f.signals_in_file
                chan_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                sigs = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                data = {
                    "signals": sigs,
                    "ch_names": chan_names,
                    "sfreq": sfreq,
                    "path": temp_path
                }
                return data, None
            except Exception as e:
                return None, f"pyedflib read failed: {e}"
        return None, "mne/pyedflib not available on server"
    except Exception as e:
        return None, f"Error reading EDF: {e}"

# Compute band powers using Welch
from scipy.signal import welch

def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    """
    Accepts either an mne Raw or a dict produced by pyedflib fallback.
    Returns:
      - df: DataFrame with per-channel absolute and relative powers for each band
      - summary_metrics: dict of global metrics
      - band_arrays: dict band->channel_relative_power_array (for heatmaps)
    """
    if raw_obj is None:
        return None, None, None

    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        data, times = raw_obj.get_data(return_times=True), raw_obj.times
        ch_names = raw_obj.ch_names
        sf = raw_obj.info['sfreq']
    elif isinstance(raw_obj, dict):
        data = raw_obj['signals']
        ch_names = raw_obj['ch_names']
        sf = raw_obj['sfreq']
    else:
        # unknown structure
        return None, None, None

    # ensure data shape channels x samples
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    if data.shape[0] > data.shape[1]:
        # ensure channels on axis 0
        pass

    nchan = data.shape[0]

    band_abs = {b: np.zeros(nchan) for b in bands}
    band_rel = {b: np.zeros(nchan) for b in bands}
    # compute PSD per channel
    for ch in range(nchan):
        sig = data[ch].astype(float)
        if np.all(np.isnan(sig)) or np.all(sig == 0):
            # leave zeros
            continue
        try:
            f, Pxx = welch(sig, fs=sf, nperseg=min(2048, len(sig)))
        except Exception:
            f, Pxx = welch(sig, fs=sf, nperseg=1024)
        total_power = np.trapz(Pxx, f)
        if total_power <= 0:
            total_power = 1e-12
        for bname, (lo, hi) in bands.items():
            idx = np.logical_and(f >= lo, f <= hi)
            val = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
            band_abs[bname][ch] = val
            band_rel[bname][ch] = (val / total_power) if total_power > 0 else 0.0

    # assemble dataframe
    rows = []
    for i, ch in enumerate(ch_names):
        row = {"ch": ch}
        for b in bands:
            row[f"{b}_abs"] = float(band_abs[b][i])
            row[f"{b}_rel"] = float(band_rel[b][i])
        rows.append(row)
    df = pd.DataFrame(rows)

    # summary metrics
    theta_alpha_ratio = (df.loc[:, "Theta_rel"].mean() / df.loc[:, "Alpha_rel"].mean()) if df["Alpha_rel"].mean() > 0 else np.inf
    summary = {
        "theta_alpha_ratio": float(theta_alpha_ratio),
        "theta_mean": float(df["Theta_rel"].mean()),
        "alpha_mean": float(df["Alpha_rel"].mean()),
        "bands_mean": {b: float(df[f"{b}_rel"].mean()) for b in bands}
    }

    return df, summary, band_rel

# Simple heatmap generator: arr is channel-relative-values per channel; we map channels to small grid
def generate_band_heatmap(band_vals: np.ndarray, ch_names: List[str], band_name: str) -> bytes:
    """
    band_vals: array length = n_channels
    produce a small heatmap image and return bytes
    """
    # To produce a heatmap, arrange channels in ~square grid based on count
    n = len(band_vals)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full((rows, cols), np.nan)
    # fill row-major
    for i in range(n):
        r = i // cols
        c = i % cols
        grid[r, c] = band_vals[i]
    # replace NaN with 0 for plotting
    g = np.nan_to_num(grid, nan=0.0)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    im = ax.imshow(g, aspect='auto', cmap='viridis', norm=Normalize(vmin=np.nanmin(g), vmax=np.nanmax(g) if np.nanmax(g)>0 else 1))
    ax.set_title(f"{band_name} relative power")
    ax.axis('off')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# SHAP bar generator (if shap_summary.json present)
def generate_shap_bar(shap_dict: dict, model_key="depression_global") -> Optional[bytes]:
    try:
        features = shap_dict.get(model_key, {})
        if not features:
            return None
        s = pd.Series(features).abs().sort_values(ascending=False)
        top = s.head(10)
        fig, ax = plt.subplots(figsize=(6,2.5))
        top[::-1].plot.barh(ax=ax)
        ax.set_xlabel("abs SHAP value")
        ax.set_title("Top SHAP contributors")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# PDF generator
def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[str]=None,
                        topo_images: Optional[Dict[str, bytes]]=None,
                        conn_image: Optional[bytes]=None,
                        bar_img: Optional[bytes]=None,
                        shap_img: Optional[bytes]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"
        # Add custom styles
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        # Header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1,6))

        # Metrics table
        rows = [["Metric", "Value"]]
        for k,v in summary.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append([f"{k}.{kk}", f"{vv}"])
            else:
                rows.append([k, str(v)])
        t = Table(rows, colWidths=[3.5*inch, 2.5*inch])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t)
        story.append(Spacer(1,8))

        # Bar image (normative)
        if bar_img:
            story.append(Paragraph("Normative Comparison", styles["H2"]))
            story.append(Spacer(1, 0.15*inch))
            story.append(RLImage(io.BytesIO(bar_img), width=5.5*inch, height=3.0*inch))
            story.append(Spacer(1, 0.3*inch))

        # Topomaps / heatmaps
        if topo_images:
            story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
            imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for _,b in topo_images.items()]
            # 2 per row
            rows_imgs = []
            row = []
            for im in imgs:
                row.append(im)
                if len(row)==2:
                    rows_imgs.append(row)
                    row=[]
            if row:
                rows_imgs.append(row)
            for r in rows_imgs:
                story.append(Table([[c for c in r]], colWidths=[2.6*inch]*len(r)))
            story.append(Spacer(1,6))

        # Connectivity image
        if conn_image:
            story.append(Paragraph("Functional Connectivity (Alpha)", styles["H2"]))
            story.append(RLImage(io.BytesIO(conn_image), width=5.5*inch, height=3.0*inch))
            story.append(Spacer(1,6))

        # SHAP
        if shap_img:
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            story.append(RLImage(io.BytesIO(shap_img), width=5.5*inch, height=2.5*inch))
            story.append(Spacer(1,6))

        # Clinical narrative (simple automated text — doctor must review)
        story.append(Paragraph("<b>Clinical narrative & recommendations</b>", styles["H2"]))
        narrative = [
            "This is an automated screening report. Review by specialist recommended.",
            "- Consider clinical correlation and neuroimaging if focal slowing present.",
            "- For elevated theta/alpha ratio consider further cognitive testing and neuropsychological assessment.",
            "- If focal delta index > 2 or asymmetric patterns, consider MRI to rule out focal lesion."
        ]
        for p in narrative:
            story.append(Paragraph(p, styles["Body"]))
        story.append(Spacer(1,12))

        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        story.append(Spacer(1,18))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF generation failed:", e)
        traceback.print_exc()
        return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# top header
st.markdown(f"""
<div style="border-radius:8px;padding:12px;background:linear-gradient(90deg,#0b63d6,#3fb0ff);color:white">
  <div style="font-weight:700;font-size:20px">NeuroEarly Pro — Clinical & Research</div>
  <div style="float:right;font-size:12px">Prepared by Golden Bird LLC</div>
</div>
""", unsafe_allow_html=True)

# Main layout: sidebar left
with st.sidebar:
    lang = st.selectbox("Language / اللغة", ["English", "العربية"])
    # patient info
    st.subheader("Patient info")
    patient_name = st.text_input("Patient Name (optional)")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.subheader("Current medications / Past medical history")
    meds = st.text_area("Current meds (one per line)")
    labs = st.text_area("Relevant labs (B12, TSH, ...)")
    st.markdown("---")
    st.file_uploader("Upload EDF file (.edf)", type=["edf"], key="edf_files", accept_multiple_files=False)
    st.markdown("---")
    st.subheader("Questionnaires")
    st.markdown("PHQ-9 (Depression screening)")
    # PHQ-9 simple: 9 questions, choices 0-3
    phq = {}
    phq_cols = st.columns(3)
    phq_qs = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",  # Q3
        "Feeling tired or having little energy",
        "Poor appetite or overeating",  # Q5
        "Feeling bad about yourself",
        "Trouble concentrating on things",
        "Moving or speaking slowly or being fidgety",  # Q8
        "Thoughts that you would be better off dead"
    ]
    for i, q in enumerate(phq_qs):
        phq[f"q{i+1}"] = st.radio(f"Q{i+1}", [0,1,2,3], index=0, key=f"phq_{i+1}")
    st.markdown("---")
    st.subheader("Alzheimer screening (short)")
    ad_qs = {
        "memory_loss": "Frequently forgetting recent events or conversations?",
        "orientation": "Disoriented to time/place?",
        "executive": "Difficulty planning or solving problems?",
    }
    ad = {}
    for k,v in ad_qs.items():
        ad[k] = st.selectbox(v, ["No", "Sometimes", "Often", "Always"], key=f"ad_{k}")

    st.markdown("---")
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)

# Main area / console
st.markdown("## Console / Visualization")
log_placeholder = st.empty()

uploaded = st.session_state.get("edf_files", None)
process_btn = st.button("Process EDF(s) and Analyze")

# local storage for results
if "results" not in st.session_state:
    st.session_state["results"] = None

if process_btn:
    log_placeholder.info("Saving and reading EDF file... please wait")
    try:
        raw, msg = read_edf_bytes(uploaded)
        if raw is None:
            st.error(f"Error reading EDF: {msg}")
        else:
            st.success("EDF loaded successfully.")
            # compute powers
            df, summary, band_rel = compute_band_powers_from_raw(raw)
            if df is None:
                st.error("Failed to compute band powers.")
            else:
                # store results
                st.session_state["results"] = {
                    "df": df.to_dict(orient="records"),
                    "summary": summary,
                    "bands": {b: band_rel[b].tolist() for b in band_rel}
                }
                log_placeholder.success("Processing complete.")
    except Exception as e:
        st.error(f"Processing error: {e}")
        traceback.print_exc()

# show results if present
res = st.session_state.get("results", None)
if res:
    df = pd.DataFrame(res["df"])
    st.markdown("### QEEG Band summary (relative power)")
    st.dataframe(df, use_container_width=True)
    # show band heatmaps
    st.markdown("### Band heatmaps")
    band_cols = st.columns(2)
    topo_imgs = {}
    for i, b in enumerate(BANDS.keys()):
        arr = np.array(res["bands"][b])
        img_bytes = generate_band_heatmap(arr, df["ch"].tolist(), b)
        topo_imgs[b] = img_bytes
        with band_cols[i%2]:
            st.image(img_bytes, caption=b, use_column_width=True)
    # SHAP
    shap_img = None
    shap_file = ROOT / "shap_summary.json"
    if shap_file.exists():
        try:
            shap_data = json.loads(shap_file.read_text(encoding="utf-8"))
            shap_img = generate_shap_bar(shap_data, model_key="depression_global")
            if shap_img:
                st.subheader("Explainable AI (SHAP)")
                st.image(shap_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Failed to load SHAP: {e}")
    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    # PDF generation button
    try:
        pdf_bytes = generate_pdf_report(res["summary"], lang="ar" if lang=="العربية" else "en",
                                        amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None,
                                        topo_images=topo_imgs, shap_img=shap_img)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF generated.")
        else:
            st.error("PDF generation failed — ensure reportlab is installed and font path valid.")
    except Exception as e:
        st.error(f"PDF generation exception: {e}")

else:
    st.info("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")

# footer
st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro System, 2025")
