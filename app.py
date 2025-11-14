# app.py — NeuroEarly Pro v6.2 (Final Clinical Edition)
# Full bilingual (English default, Arabic optional with Amiri font),
# PHQ-9, AD8, EDF upload, band-power, topomaps (if mne available), SHAP display,
# PDF report generation (reportlab), and robust EDF reading.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

# Optional heavy libs (graceful fallback)
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDFLIB = True
except Exception:
    HAS_PYEDFLIB = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

# Paths & assets
ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"  # user said Amiri in repo root

# Basic constants
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# Utility: timestamp
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# Arabic helper
def maybe_rtl(text: str, lang: str):
    if lang.startswith("ar") and HAS_ARABIC:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text

# Robust EDF reader: accepts UploadedFile (BytesIO) or local path
def read_edf_bytes(uploaded) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[str]], Optional[str]]:
    """
    Returns (data_matrix (n_channels, n_samples), sfreq, ch_names, err_msg)
    """
    if not uploaded:
        return None, None, None, "No file provided"
    # uploaded is Streamlit UploadedFile: has .getvalue() -> bytes
    data_bytes = uploaded.getvalue()
    # Try to use mne first if available (supports BytesIO)
    try:
        if HAS_MNE:
            bio = io.BytesIO(data_bytes)
            # mne can read raw from file-like for EDFs
            raw = mne.io.read_raw_edf(bio, preload=True, verbose=False)
            data, times = raw.get_data(return_times=True)
            sf = raw.info["sfreq"]
            chs = raw.ch_names
            return data, float(sf), chs, None
    except Exception as e:
        # fallback to pyedflib if available
        try:
            if HAS_PYEDFLIB:
                # pyedflib requires a filename — write to temp and read
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                tf.write(data_bytes)
                tf.flush(); tf.close()
                f = pyedflib.EdfReader(str(tf.name))
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                n_samples = f.getNSamples()[0]
                data = np.zeros((n, n_samples))
                for i in range(n):
                    data[i, :] = f.readSignal(i)
                f._close()
                try:
                    os.unlink(tf.name)
                except Exception:
                    pass
                return data, float(sfreq), ch_names, None
        except Exception as e2:
            return None, None, None, f"EDF read error: mne failed ({e}) and pyedflib failed ({e2})"
    return None, None, None, "Could not read EDF (unknown reason)"

# Band power computation (Welch)
from scipy.signal import welch, detrend

def compute_band_powers(data: np.ndarray, sfreq: float, bands: Dict[str,Tuple[float,float]] = BANDS):
    """
    data: n_channels x n_samples
    returns DataFrame with abs and relative powers per channel x band
    """
    if data is None or sfreq is None:
        return None
    n_chan = data.shape[0]
    psd_dict = {}
    # compute welch for each channel
    freqs, _ = welch(data[0], fs=sfreq, nperseg=min(2048, data.shape[1]), detrend='constant')
    pxx = []
    for ch in range(n_chan):
        f, Pxx = welch(data[ch], fs=sfreq, nperseg=min(2048, data.shape[1]), detrend='constant')
        pxx.append(Pxx)
    pxx = np.array(pxx)  # n_chan x len(freqs)
    df_rows = []
    for ch in range(n_chan):
        row = {}
        total_power = np.trapz(pxx[ch], f)
        row["total_power"] = float(total_power)
        for name,(lo,hi) in bands.items():
            idx = np.logical_and(f>=lo, f<=hi)
            band_power = float(np.trapz(pxx[ch, idx], f[idx])) if np.any(idx) else 0.0
            row[f"{name}_abs"] = band_power
            row[f"{name}_rel"] = (band_power/total_power) if total_power>0 else 0.0
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    return df, f  # DataFrame and freqs array

# Topomap generation:
def generate_topomap_images(band_vals: np.ndarray, ch_names: List[str], band_name: str):
    """
    band_vals: length n_ch (relative or abs)
    returns PNG bytes
    If mne present and montage available, use mne.viz.plot_topomap
    Else fallback to bar chart image
    """
    try:
        buf = io.BytesIO()
        n_ch = len(ch_names)
        if HAS_MNE:
            # try to get standard montage positions
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                pos = []
                for ch in ch_names:
                    try:
                        pos.append(montage.get_pos2d(ch))
                    except Exception:
                        pos.append(None)
                # If many positions missing, fallback
                if sum(1 for p in pos if p is not None) < max(4, n_ch//3):
                    raise Exception("Insufficient montage positions")
                # prepare info
                info = mne.create_info(ch_names, sfreq=250.0, ch_types='eeg')
                info.set_montage(montage, match_case=False)
                # use mne topomap
                fig, ax = plt.subplots(figsize=(4,3))
                mne.viz.plot_topomap(band_vals, info, axes=ax, show=False)
                ax.set_title(band_name)
                fig.tight_layout()
                fig.savefig(buf, format='png'); plt.close(fig)
                buf.seek(0)
                return buf.getvalue()
            except Exception:
                pass
        # fallback: bar chart across channels
        fig, ax = plt.subplots(figsize=(6,2.4))
        ax.bar(np.arange(len(ch_names)), band_vals)
        ax.set_xticks(np.arange(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_title(band_name)
        fig.tight_layout()
        fig.savefig(buf, format='png'); plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        return None

# SHAP loader (from shap_summary.json in repo)
def load_shap_summary():
    shap_file = ROOT / "shap_summary.json"
    if shap_file.exists():
        try:
            return json.loads(shap_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# PDF generator (simple bilingual professional)
def generate_pdf_report(summary: dict, lang="en", amiri_path: Optional[str] = None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        # register amiri if exists
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            if amiri_path and Path(amiri_path).exists():
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
        except Exception:
            pass
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=12))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # Header + logo
        if LOGO_PATH.exists():
            try:
                im = RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch)
                story.append(im)
            except Exception:
                pass
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        # patient info
        pinfo = summary.get("patient_info", {})
        pid = pinfo.get("id","-")
        dob = pinfo.get("dob","-")
        created = summary.get("created", now_ts())
        story.append(Paragraph(f"Patient ID: {pid}", styles["Body"]))
        story.append(Paragraph(f"DOB: {dob}", styles["Body"]))
        story.append(Paragraph(f"Report generated: {created}", styles["Body"]))
        story.append(Spacer(1,8))
        # metrics table (small)
        metrics = summary.get("metrics", {})
        rows = [["Metric","Value"]]
        for k,v in metrics.items():
            rows.append([k, str(v)])
        t2 = Table(rows, colWidths=[3.5*inch, 2.5*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t2)
        story.append(Spacer(1,8))
        # Add topomap images if present
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps (band relative power)", styles["H2"]))
            for band, img_b in summary["topo_images"].items():
                try:
                    img = RLImage(io.BytesIO(img_b), width=3.6*inch, height=2.2*inch)
                    story.append(img)
                except Exception:
                    pass
        story.append(Spacer(1,6))
        # SHAP section
        if summary.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            try:
                img = RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=3.0*inch)
                story.append(img)
            except Exception:
                pass
        story.append(Spacer(1,6))
        # Narrative & recommendations
        story.append(Paragraph("Clinical narrative & recommendations", styles["H2"]))
        narrative = summary.get("narrative","Automated screening report. Interpret clinically.")
        story.append(Paragraph(narrative, styles["Body"]))
        if summary.get("recommendations"):
            for r in summary["recommendations"]:
                story.append(Paragraph(f"- {r}", styles["Body"]))
        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.exception(e)
        return None

# PHQ-9 and AD8 questionnaires
PHQ9 = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking so slowly that other people could have noticed",
    "Thoughts that you would be better off dead"
]

AD8 = [
    "Problems with judgment (e.g., bad decision-making)",
    "Less interest in hobbies/activities",
    "Repetition of questions/stories/comments",
    "Trouble learning to use a tool/ appliance",
    "Forgetting the correct month or year",
    "Difficulty handling complicated financial affairs",
    "Trouble remembering appointments",
    "Daily problems with thinking and memory"
]

# App UI
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")
# header
HEADER = """
<div style="background:linear-gradient(90deg,#0b63d6,#2fb2ff);padding:12px;border-radius:8px;color:white;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-weight:700;font-size:20px">NeuroEarly Pro — Clinical & Research</div>
    <div style="font-size:12px">Prepared by Golden Bird LLC</div>
  </div>
</div>
"""
st.markdown(HEADER, unsafe_allow_html=True)

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English","Arabic"])
    use_ar = lang.startswith("Arabic")
    patient_name = st.text_input("Patient Name (not printed)")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female","Other"])
    meds = st.text_area("Current meds (one per line) / الأدوية الحالية")
    labs = st.text_area("Relevant labs (one per line) / آزمایشات")
    st.markdown("---")
    st.header("Upload")
    uploaded = st.file_uploader("Upload EDF file (.edf)", type=["edf","EDF"], accept_multiple_files=False)
    if uploaded:
        st.write("Uploaded:", uploaded.name)
    st.markdown("---")
    st.header("Questionnaires")
    st.write("PHQ-9 (Depression)")
    phq_ans = []
    for i,q in enumerate(PHQ9):
        label = maybe_rtl(q, "ar") if use_ar else q
        ans = st.radio(f"Q{i+1}", options=[0,1,2,3], index=0, key=f"phq_{i}")
        phq_ans.append(ans)
    st.write("AD8 (Cognitive)")
    ad8_ans = []
    for i,q in enumerate(AD8):
        label = maybe_rtl(q, "ar") if use_ar else q
        ans = st.radio(f"AD{i+1}", options=[0,1], index=0, key=f"ad8_{i}")
        ad8_ans.append(ans)
    st.button("Process EDF(s) and Analyze", key="process_btn")

# Main area (console + visual)
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Console / Visualization")
    log = st.empty()
    status = st.empty()
with col2:
    st.subheader("Upload & Quick stats")
    quick = st.empty()

# Only process when button pressed
if st.session_state.get("process_btn"):
    log.info("Reading EDF file... please wait")
    data, sfreq, ch_names, err = read_edf_bytes(uploaded) if uploaded else (None,None,None,"No file")
    if err:
        st.error(maybe_rtl(f"Error reading EDF: {err}", "ar" if use_ar else "en"))
    else:
        st.success("EDF loaded successfully." if not use_ar else maybe_rtl("فایل با موفقیت خوانده شد.", "ar"))
        quick.write(f"Sampling rate (Hz): {sfreq}")
        quick.write(f"Channels: {len(ch_names)}")
        # compute band powers
        df_bands, freqs = compute_band_powers(data, sfreq)
        if df_bands is None:
            st.error("Band power computation failed.")
        else:
            # show DataFrame
            st.subheader("QEEG Band summary (relative power)")
            display_df = df_bands[[c for c in df_bands.columns if c.endswith("_rel") or c.endswith("_abs")]]
            # add channel names
            display_df.insert(0, "ch", ch_names)
            st.dataframe(display_df.style.format(precision=4), height=360)
            # compute global theta/alpha ratio
            try:
                theta_mean = display_df["Theta_rel"].mean()
                alpha_mean = display_df["Alpha_rel"].mean()
                theta_alpha_ratio = (theta_mean/alpha_mean) if alpha_mean>0 else 0.0
            except Exception:
                theta_alpha_ratio = 0.0
            st.metric("Theta/Alpha Ratio", f"{theta_alpha_ratio:.3f}")
            # topomaps: for each band create image
            topo_images = {}
            for band in BANDS.keys():
                vals = display_df[f"{band}_rel"].values
                img_b = generate_topomap_images(vals, ch_names, band)
                if img_b:
                    topo_images[band] = img_b
                    st.image(img_b, caption=f"{band} (relative)", use_column_width=False)
                else:
                    st.info(f"Topomap for {band} not available.")
            # SHAP: show if shap_summary.json exists
            shap_summary = load_shap_summary()
            shap_img_b = None
            if shap_summary:
                # choose model key heuristically
                model_key = "depression_global"
                if theta_alpha_ratio > 1.3:
                    model_key = "alzheimers_global"
                features = shap_summary.get(model_key, {})
                if features:
                    s = pd.Series(features).abs().sort_values(ascending=False)
                    # bar chart
                    fig = plt.figure(figsize=(6,3))
                    ax = fig.add_subplot(111)
                    s.head(10).plot.bar(ax=ax)
                    ax.set_title("Top contributors (SHAP)")
                    fig.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    shap_img_b = buf.getvalue()
                    st.image(shap_img_b, caption="SHAP top features", use_column_width=True)
            # Prepare summary for PDF
            summary = {
                "patient_info": {"id": patient_id, "dob": dob.isoformat()},
                "metrics": {
                    "theta_alpha_ratio": round(theta_alpha_ratio,3),
                    "theta_mean": round(float(display_df["Theta_rel"].mean()),4),
                    "alpha_mean": round(float(display_df["Alpha_rel"].mean()),4),
                },
                "topo_images": topo_images,
                "shap_img": shap_img_b,
                "created": now_ts(),
                "narrative": "Automated screening. Review by specialist recommended.",
                "recommendations": [
                    "Consider clinical correlation and neuroimaging if focal slowing present.",
                    "For elevated theta/alpha ratio consider further cognitive testing."
                ],
            }
            # PDF generation
            pdf_bytes = generate_pdf_report(summary, lang="ar" if use_ar else "en", amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
            if pdf_bytes:
                st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                st.success("PDF generated.")
            else:
                st.error("PDF generation failed — reportlab may not be installed on the server.")
    # done processing

# Footer notes
st.markdown("---")
st.markdown("**Notes:** Default language is English; Arabic uses Amiri font if present. For best connectivity and microstate results install `mne` and `scikit-learn` on the server.")
