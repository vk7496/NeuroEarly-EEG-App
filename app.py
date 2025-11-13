# app.py — NeuroEarly Pro (v6.2 Blue Clinical Edition)
# Full bilingual (English default, Arabic option), EDF reader (safe), band powers, topomaps, SHAP, PDF reporter.

import os
import io
import sys
import json
import math
import time
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

import streamlit as st

# Optional heavy libs. App works in degraded mode when missing.
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_SCIPY = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_RESHAPER = False
HAS_BIDI = False
HAS_SKLEARN = False

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
    from scipy.signal import welch
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
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
    HAS_ARABIC_RESHAPER = True
except Exception:
    HAS_ARABIC_RESHAPER = False

try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except Exception:
    HAS_BIDI = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# --- Configuration / assets ---
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"      # put your logo here
AMIRI_PATH = BASE_DIR / "Amiri-Regular.ttf"        # Amiri font path if present
SHAP_JSON = BASE_DIR / "shap_summary.json"         # optional shap export

# App look & feel
MAIN_BLUE = "#3A7BD5"
LIGHT_BG = "#f7fbff"

# EEG bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Helper: now timestamp
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# ---------- Utilities for Arabic text if needed ----------
def maybe_rtl(text: str) -> str:
    """Return reshaped bidi Arabic text if libs present, else original."""
    if not text:
        return text
    if HAS_ARABIC_RESHAPER and HAS_BIDI:
        try:
            reshaped = arabic_reshaper.reshape(text)
            bidi = get_display(reshaped)
            return bidi
        except Exception:
            return text
    return text

# ---------- EDF reading (safe) ----------
def read_edf_bytes(uploaded_file) -> Tuple[Optional[Any], Optional[str], Optional[Dict[str,Any]]]:
    """
    Read uploaded EDF-like file (BytesIO) and return:
      - raw: mne Raw object if available or simple dict data if not
      - msg: error message or None
      - meta: dict with sfreq, ch_names, data_shape etc
    Implementation: write bytes to temp file and pass filename to readers.
    """
    if uploaded_file is None:
        return None, "No file provided", None
    try:
        # Save to named temp file (avoid BytesIO in mne/pyedflib)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Try mne first if available
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                sfreq = raw.info.get('sfreq', None)
                ch_names = raw.ch_names
                data = raw.get_data()
                meta = {"sfreq": sfreq, "ch_names": ch_names, "shape": data.shape}
                return raw, None, meta
            except Exception as e_mne:
                # fallback to pyedflib
                pass

        if HAS_PYEDFLIB:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreqs = [f.getSampleFrequency(i) for i in range(n)]
                # pick first sample freq if consistent
                sfreq = sfreqs[0] if sfreqs else None
                # read signals as numpy (may be large)
                signals = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                meta = {"sfreq": sfreq, "ch_names": ch_names, "shape": signals.shape}
                return signals, None, meta
            except Exception as e_py:
                return None, f"pyedflib read failed: {e_py}", None

        # If no heavy readers available: return error
        return None, "No EDF reader (mne or pyedflib) available on server.", None

    except Exception as e:
        return None, f"Error reading EDF: {e}", None
    finally:
        # don't remove the temp file immediately (readers may still have references)
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                # safe remove
                os.remove(tmp_path)
        except Exception:
            pass

# ---------- Band power computation ----------
def compute_band_powers_from_raw(raw_like, sfreq=None, ch_names=None, nperseg=2048) -> pd.DataFrame:
    """
    raw_like: either MNE Raw object or 2D numpy (n_channels, n_samples).
    Returns: DataFrame with each channel row and absolute & relative band power columns.
    """
    # Prepare data
    if HAS_MNE and isinstance(raw_like, mne.io.BaseRaw):
        data = raw_like.get_data()  # shape (n_ch, n_samples)
        sfreq = raw_like.info['sfreq']
        ch_names = raw_like.ch_names
    elif isinstance(raw_like, np.ndarray):
        data = raw_like
    else:
        raise ValueError("Unsupported raw_like type for band power computation.")

    n_ch = data.shape[0]
    rows = []
    for i in range(n_ch):
        sig = data[i, :]
        # Use Welch
        if HAS_SCIPY:
            freqs, psd = welch(sig, fs=sfreq, nperseg=min(nperseg, len(sig)))
        else:
            # fallback: simple FFT-based estimation
            n = len(sig)
            freqs = np.fft.rfftfreq(n, 1.0/sfreq)
            psd = np.abs(np.fft.rfft(sig))**2 / n
        band_abs = {}
        total_power = np.trapz(psd, freqs)
        for bname, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            val = np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
            band_abs[bname+"_abs"] = float(val)
            band_abs[bname+"_rel"] = float(val / total_power) if total_power>0 else 0.0
        row = {"ch": ch_names[i] if ch_names else f"ch_{i}"}
        row.update(band_abs)
        rows.append(row)
    df = pd.DataFrame(rows)
    # order columns sensibly
    cols = ["ch"] + [f"{b}_abs" for b in BANDS.keys()] + [f"{b}_rel" for b in BANDS.keys()]
    df = df[cols]
    return df

# ---------- Simple topomap scatter (no head interpolation) ----------
# We'll use approximate 10-20 2D positions for common channels. If not in file, scatter by index.
STD_1020_POS = {
    # subset approximate x,y coords in [-1,1]
    "Fp1": (-0.2, 1.0), "Fp2": (0.2, 1.0),
    "F7": (-1.0, 0.4), "F3": (-0.4, 0.5), "Fz": (0.0, 0.6), "F4": (0.4, 0.5), "F8": (1.0, 0.4),
    "T3": (-1.0, 0.0), "C3": (-0.4, 0.0), "Cz": (0.0, 0.0), "C4": (0.4, 0.0), "T4": (1.0, 0.0),
    "T5": (-1.0, -0.4), "P3": (-0.4, -0.5), "Pz": (0.0, -0.5), "P4": (0.4, -0.5), "T6": (1.0, -0.4),
    "O1": (-0.2, -1.0), "O2": (0.2, -1.0)
}

def draw_topomap_scatter(values: Dict[str,float], title="", figsize=(4,3)):
    """
    values: dict channel->value
    returns PNG bytes
    """
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys, vals, labels = [], [], [], []
    for i,(ch,v) in enumerate(values.items()):
        pos = STD_1020_POS.get(ch)
        if pos is None:
            # place evenly on semicircle if unknown
            theta = 2*math.pi*(i/len(values))
            pos = (math.cos(theta)*0.7, math.sin(theta)*0.7)
        xs.append(pos[0]); ys.append(pos[1]); vals.append(v); labels.append(ch)
    sc = ax.scatter(xs, ys, c=vals, s=200, cmap="Blues", edgecolor="k")
    for xi, yi, lab in zip(xs, ys, labels):
        ax.text(xi, yi, lab, fontsize=8, ha="center", va="center")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("equal")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------- PDF generator ----------
def generate_pdf_report(summary: Dict[str,Any], lang="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
    """
    Create bilingual clinical PDF containing metrics, images, and narrative.
    Returns bytes or None on failure.
    """
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"
        # add custom styles
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(MAIN_BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(MAIN_BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # Header
        if LOGO_PATH.exists():
            try:
                story.append(RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch))
            except Exception:
                pass
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        # Patient info
        pinfo = summary.get("patient_info", {})
        pid = pinfo.get("id","")
        dob = pinfo.get("dob","")
        created = summary.get("created", now_ts())
        story.append(Paragraph(f"<b>Patient ID:</b> {pid}", styles["Body"]))
        story.append(Paragraph(f"<b>Date of birth:</b> {dob}", styles["Body"]))
        story.append(Paragraph(f"<b>Report generated:</b> {created}", styles["Body"]))
        story.append(Spacer(1,12))
        # Insert bar image if present
        if summary.get("bar_img"):
            try:
                story.append(Paragraph("QEEG Summary (Theta/Alpha etc.)", styles["H2"]))
                story.append(RLImage(io.BytesIO(summary["bar_img"]), width=5.5*inch, height=3*inch))
                story.append(Spacer(1,8))
            except Exception:
                pass
        # Topomap images
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["H2"]))
            imgs = []
            for band, bbytes in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(bbytes), width=2.6*inch, height=1.6*inch))
                except Exception:
                    pass
            if imgs:
                # put 2 per row
                rows = []
                row = []
                for i,im in enumerate(imgs):
                    row.append(im)
                    if len(row)==2:
                        rows.append(row); row=[]
                if row: rows.append(row)
                t = Table(rows, colWidths=[3*inch, 3*inch])
                t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
                story.append(t)
                story.append(Spacer(1,8))
        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("Top contributors (XAI)", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.0*inch))
                story.append(Spacer(1,8))
            except Exception:
                pass
        # Narrative & recommendations
        story.append(Paragraph("<b>Clinical narrative & recommendations</b>", styles["H2"]))
        for line in summary.get("recommendations", []):
            story.append(Paragraph(line, styles["Body"]))
        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        return None

# ---------- Simple XAI SHAP plot (bar) ----------
def shap_bar_from_json(shap_file: Path, model_key="depression_global"):
    try:
        if not shap_file.exists():
            return None
        data = json.loads(shap_file.read_text(encoding="utf-8"))
        features = data.get(model_key, {})
        if not features:
            return None
        s = pd.Series(features).abs().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,2.2))
        s.head(10).plot.bar(ax=ax)
        ax.set_title("Top contributors (SHAP)")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("SHAP plot error:", e)
        return None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# Top banner
st.markdown(f"""
<div style="background:linear-gradient(90deg,{MAIN_BLUE}, #6fb1ff); padding:12px; border-radius:8px; color:white; display:flex; align-items:center; justify-content:space-between">
  <div style="font-weight:700; font-size:18px;">🧠 NeuroEarly Pro v6.2 — Clinical & Research</div>
  <div style="font-size:12px; opacity:0.95;">Prepared by Golden Bird LLC</div>
</div>
""", unsafe_allow_html=True)

# Layout: left sidebar (patient & upload), right main
with st.sidebar:
    st.header(maybe_rtl("Settings / Patient" if st.session_state.get("lang","en")=="en" else "إعدادات / المريض"))
    lang = st.selectbox("Language / اللغة", options=["English","Arabic"], index=0)
    st.session_state["lang"] = "ar" if lang=="Arabic" else "en"
    st.write("---")
    pid = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female"])
    st.write("---")
    st.subheader("Blood Tests (summary)")
    labs = st.text_area("Enter labs (one per line) e.g. B12: 250 pg/mL")
    st.write("---")
    st.subheader("Medications")
    meds = st.text_area("Current meds (one per line)")
    st.write("---")
    st.subheader("Upload")
    uploaded = st.file_uploader("Upload EDF file (.edf)", type=["edf"], accept_multiple_files=False)
    st.write("")
    process_btn = st.button("Process EDF(s) and Analyze")

# Console area and main area
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Console")
    logbox = st.empty()
with col2:
    st.subheader("Upload & Quick stats")
    status_box = st.empty()
    results_area = st.container()

# When user presses process:
if process_btn:
    logbox.info("Saving and reading EDF file... please wait")
    if uploaded is None:
        logbox.error("No EDF uploaded.")
    else:
        raw, msg, meta = read_edf_bytes(uploaded)
        if raw is None:
            logbox.error(f"EDF load error: {msg}")
        else:
            # success
            status_box.success(f"EDF loaded successfully. Shape: {meta.get('shape')} Sfreq: {meta.get('sfreq')}")
            # compute band powers
            try:
                if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
                    ch_names = raw.ch_names
                    sfreq = raw.info['sfreq']
                    band_df = compute_band_powers_from_raw(raw, sfreq=sfreq, ch_names=ch_names)
                elif isinstance(raw, np.ndarray):
                    # assume raw is signals (n_ch, n_samples) and meta contains ch_names & sfreq
                    ch_names = meta.get("ch_names", [f"ch{i}" for i in range(raw.shape[0])])
                    sfreq = meta.get("sfreq", 250.0)
                    band_df = compute_band_powers_from_raw(raw, sfreq=sfreq, ch_names=ch_names)
                else:
                    raise ValueError("Unsupported raw type after loading.")
                # show table
                with results_area:
                    st.markdown("### QEEG Band summary (relative power)")
                    st.dataframe(band_df.style.format({col: "{:.4f}" for col in band_df.columns if col.endswith("_rel")}), height=360)
                # Generate bar (normative) example: Theta/Alpha ratio summary
                # compute theta/alpha ratio global
                theta_rel = band_df["Theta_rel"].mean()
                alpha_rel = band_df["Alpha_rel"].mean()
                theta_alpha_ratio = (theta_rel/alpha_rel) if alpha_rel>0 else 0.0
                st.session_state["theta_alpha_ratio"] = theta_alpha_ratio
                # bar chart data
                bar_series = pd.Series({
                    "Theta_rel_mean": float(theta_rel),
                    "Alpha_rel_mean": float(alpha_rel),
                    "Theta/Alpha": float(theta_alpha_ratio)
                })
                fig1, ax1 = plt.subplots(figsize=(6,2))
                bar_series.plot.bar(ax=ax1)
                ax1.set_title("Theta/Alpha vs Norm")
                buf_bar = io.BytesIO(); fig1.tight_layout(); fig1.savefig(buf_bar, format="png"); plt.close(fig1); buf_bar.seek(0)
                st.image(buf_bar.getvalue(), use_column_width=True)
                # per-band topomap scatter images
                topo_imgs = {}
                for bname in BANDS.keys():
                    # create dict channel->relative value for band
                    values = {row["ch"]: row[f"{bname}_rel"] for _,row in band_df.iterrows()}
                    topo_png = draw_topomap_scatter(values, title=bname)
                    topo_imgs[bname] = topo_png
                    st.image(topo_png, caption=f"{bname} topomap (relative)", width=240)
                # SHAP visual
                shap_png = shap_bar_from_json(SHAP_JSON) if SHAP_JSON.exists() else None
                if shap_png:
                    st.image(shap_png, caption="XAI: Top contributors (SHAP)")
                else:
                    st.info("XAI unavailable — upload shap_summary.json to enable SHAP.")
                # Build summary object for PDF
                summary = {
                    "patient_info": {"id": pid, "dob": str(dob)},
                    "metrics": {"theta_alpha_ratio": theta_alpha_ratio},
                    "topo_images": topo_imgs,
                    "bar_img": buf_bar.getvalue(),
                    "shap_img": shap_png,
                    "recommendations": [
                        "This is an automated screening report. Clinical correlation recommended.",
                        "Consider MRI if focal delta index >2 or focal abnormalities persist.",
                        "Follow-up in 3-6 months for moderate risk cases."
                    ],
                    "created": now_ts()
                }
                pdf_bytes = generate_pdf_report(summary, lang=st.session_state["lang"], amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
                if pdf_bytes:
                    st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                    st.success("PDF generated.")
                else:
                    st.error("PDF generation failed — ensure reportlab is installed on server.")
                # CSV of metrics
                try:
                    df_export = band_df.copy()
                    csv = df_export.to_csv(index=False).encode("utf-8")
                    st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
                except Exception:
                    pass

            except Exception as e:
                st.exception(e)

# If no processed results yet, show instructions
else:
    st.info("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")

# ---------- Questionnaires (below main) ----------
st.markdown("---")
st.header("Questionnaires")

# PHQ-9 (Depression)
st.subheader("PHQ-9 (Depression screening)")
cols = st.columns(3)
phq = {}
phq_defaults = [0]*9
for i in range(9):
    with cols[i%3]:
        phq[f"Q{i+1}"] = st.radio(f"Q{i+1}", options=[0,1,2,3], index=0, key=f"phq_{i+1}")

phq_score = sum(phq.values())
st.write(f"PHQ-9 score: **{phq_score}**")
st.write("Interpretation: 0-4 none/minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20+ severe")

# Alzheimer screening short (example)
st.subheader("Alzheimer screening (brief)")
alz_q = {}
# Example important items with custom options (user asked attention to Q3/5/8)
alz_q["memory_change"] = st.selectbox("Memory problems noted by family?", ["No","Yes","Unclear"], key="alz_1")
alz_q["orientation"] = st.selectbox("Orientation problems?", ["No","Yes"], key="alz_2")
# item 3 special
alz_q["word_find"] = st.radio("Difficulty finding words (Q3)?", ["None","Occasional","Frequent"], index=0, key="alz_3")
alz_q["daily_tasks"] = st.selectbox("Difficulty with daily tasks?", ["No","Some difficulty","Needs help"], key="alz_4")
# item 5 special
alz_q["lost_place"] = st.radio("Gets lost in familiar places (Q5)?", ["No","Sometimes","Often"], index=0, key="alz_5")
alz_q["mood_change"] = st.selectbox("Significant mood/behavior change?", ["No","Yes"], key="alz_6")
# item 8 special
alz_q["disorientation_time"] = st.radio("Disorientation to time (Q8)?", ["No","Sometimes","Often"], index=0, key="alz_8")

st.write("Alzheimer screening (summary): ", alz_q)

# Footer notes
st.markdown("---")
st.markdown("Prepared by Golden Bird LLC. For clinical use only; not a diagnostic device. Always correlate with clinical exam and other investigations.")
