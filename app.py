# app.py — NeuroEarly Pro v7 (Clinical)
# Features:
# - Bilingual (English default / Arabic optional RTL with Amiri)
# - Sidebar left: language, patient info, meds, labs, EDF upload
# - Main: console/log, topomaps for Delta/Theta/Alpha/Beta/Gamma, band table
# - Right/below: questionnaires (PHQ-9, AD8), scoring & flags
# - Robust EDF reading (mne preferred, pyedflib fallback; handles BytesIO)
# - SHAP visualization if shap_summary.json present
# - PDF report (ReportLab) with Amiri font for Arabic if available
# - Graceful degradation when optional libs missing

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

import streamlit as st
from PIL import Image as PILImage

# Optional heavy libs - try import, set flags
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False
HAS_SCIPY = False

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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
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
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    import scipy
    from scipy.signal import welch
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ----------------- Config / Assets -----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"  # place font here if you want Arabic PDF
LOGO_PATH_PNG = ASSETS / "goldenbird_logo.png"
LOGO_PATH_SVG = ROOT / "assetsGoldenBird_logo.svg.svg"  # fallback if present
SHAP_JSON = ROOT / "shap_summary.json"

# Default healthy baseline EDF (use your uploaded baseline)
HEALTHY_EDF = Path("/mnt/data/test_edf.edf")  # <-- existing file you uploaded

APP_TITLE = "NeuroEarly Pro — Clinical & Research"
BLUE = "#2D9CDB"
LIGHT_BG = "#eef7ff"
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# ----------------- Helpers -----------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def read_edf_bytes(uploaded) -> Tuple[Optional[object], Optional[str]]:
    """
    Read uploaded EDF object robustly.
    Returns tuple (raw, msg) where raw is mne Raw or dict {'signals', 'ch_names', 'sfreq'} for fallback.
    """
    if not uploaded:
        return None, "No file"
    # uploaded may be a streamlit UploadedFile object
    data = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()
    # write to a temporary file and point mne/pyedflib at it (avoids BytesIO incompatibility)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        # try mne
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                os.unlink(tmp_path)
                return raw, None
            except Exception as e:
                # fallback to pyedflib
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                signals = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                os.unlink(tmp_path)
                return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
            except Exception as e:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return None, f"pyedflib read error: {e}"
        else:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return None, "No backend available (install mne or pyedflib)"
    except Exception as e:
        return None, f"Temporary file write/read failed: {e}"

def ensure_1d(arr):
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1)
    return a

def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    """
    Accepts mne Raw or fallback dict {'signals', 'ch_names', 'sfreq'}.
    Returns dictionary: {ch_name: {band_abs, band_rel ...}}, and overall averages.
    """
    # prepare signals: shape (n_ch, n_samples)
    if raw_obj is None:
        return None
    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        raw = raw_obj.copy().pick_types(eeg=True, meg=False)
        sf = raw.info['sfreq']
        data, _ = raw.get_data(return_times=True), raw.ch_names  # data shape (n_ch, n_samples)
        ch_names = raw.ch_names
    else:
        # fallback dict
        dd = raw_obj
        data = np.asarray(dd["signals"])
        ch_names = dd["ch_names"]
        sf = dd["sfreq"]
        # ensure shape: (n_ch, n_samples)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    n_ch = data.shape[0]
    # compute PSD using Welch if scipy available, else simple FFT window
    band_summary = {}
    total_power_per_ch = np.zeros(n_ch)
    freqs = None
    psd = None
    if HAS_SCIPY:
        # use welch
        nperseg = int(min(4 * sf, data.shape[1]))
        psd_list = []
        freqs = None
        for ch in range(n_ch):
            f, Pxx = welch(data[ch, :], fs=sf, nperseg=nperseg)
            psd_list.append(Pxx)
            freqs = f
        psd = np.vstack(psd_list)  # shape (n_ch, len(freqs))
    else:
        # simple FFT (less stable) - compute power spectral density
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0/sf)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft_vals

    # total power per channel
    total_power_per_ch = psd.sum(axis=1) + 1e-12
    for i, ch in enumerate(ch_names):
        band_values_abs = {}
        band_values_rel = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            band_power = psd[i, mask].sum() if mask.any() else 0.0
            band_values_abs[bname] = float(band_power)
            band_values_rel[bname] = float(band_power / total_power_per_ch[i])
        band_summary[ch] = {f"{b}_abs": band_values_abs[b] for b in band_values_abs}
        # add rel
        for b in band_values_rel:
            band_summary[ch][f"{b}_rel"] = band_values_rel[b]
    # Compute global metrics (averages)
    global_metrics = {}
    # Theta/Alpha ratio as mean across channels of (theta_rel / alpha_rel) safe
    theta_alpha_list = []
    for i, ch in enumerate(ch_names):
        a_rel = band_summary[ch].get("Alpha_rel", 0.0)
        t_rel = band_summary[ch].get("Theta_rel", 0.0)
        if a_rel > 0:
            theta_alpha_list.append(t_rel / a_rel)
    global_metrics["theta_alpha_ratio"] = float(np.mean(theta_alpha_list)) if theta_alpha_list else 0.0
    # mean connectivity placeholder (needs mne/connectivity)
    global_metrics["mean_connectivity_alpha"] = 0.0
    # FDI example (focal delta index) rough: max delta_rel across channels * 100
    delta_rels = [band_summary[ch].get("Delta_rel", 0.0) for ch in ch_names]
    global_metrics["FDI"] = float(max(delta_rels) * 100.0)
    return {"bands": band_summary, "metrics": global_metrics, "ch_names": ch_names, "sfreq": sf, "psd": psd, "freqs": freqs}

def topomap_png_from_vals(vals: np.ndarray, band_name:str="Band"):
    """
    Create a simple heat-grid representation (fallback topomap) from 1D channel values.
    Returns PNG bytes or None.
    """
    try:
        arr = np.asarray(vals).astype(float)
        if arr.ndim != 1:
            arr = arr.ravel()
        n = len(arr)
        # create nearest square grid (fill missing with nan)
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper', vmin=np.nanmin(grid), vmax=np.nanmax(grid))
        ax.set_title(f"{band_name}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

def make_band_table(band_summary:Dict[str,Any], selected_channels:Optional[List[str]]=None):
    """
    Convert band_summary to a pandas DataFrame for display.
    """
    rows = []
    chs = list(band_summary.keys())
    for ch in chs:
        row = {"ch": ch}
        for b in BANDS:
            row[f"{b}_abs"] = band_summary[ch].get(f"{b}_abs", 0.0)
            row[f"{b}_rel"] = band_summary[ch].get(f"{b}_rel", 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index("ch")
    return df

# ----------------- PDF report generator (simplified but complete) -----------------
def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[Path]=None) -> Optional[bytes]:
    """
    Create a bilingual PDF report (English default, Arabic if lang == 'ar') with embedded images.
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
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri font register failed:", e)
        # custom styles
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1, 6))
        # patient info
        pi = summary.get("patient_info", {})
        info_table = [
            ["Patient ID:", pi.get("id","-"), "DOB:", pi.get("dob","-")],
            ["Report created:", summary.get("created", now_ts()), "Exam:", pi.get("exam","EEG")]
        ]
        t = Table(info_table, colWidths=[70,150,70,120])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Spacer(1,12))
        # metrics table (simple)
        metrics = summary.get("metrics", {})
        mrows = [["Metric","Value"]]
        for k,v in metrics.items():
            mrows.append([k, f"{v}"])
        mt = Table(mrows, colWidths=[200,200])
        mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(Paragraph("<b>Key QEEG metrics</b>", styles["H2"]))
        story.append(mt)
        story.append(Spacer(1,12))
        # attach images if present
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topomaps</b>", styles["H2"]))
            for name, img_bytes in summary["topo_images"].items():
                try:
                    img = RLImage(io.BytesIO(img_bytes), width=200, height=140)
                    story.append(img)
                    story.append(Spacer(1,6))
                except Exception:
                    pass
        # SHAP summary if present
        if summary.get("shap_img"):
            story.append(Spacer(1,6))
            story.append(Paragraph("<b>Explainability (SHAP)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=360, height=120))
            except Exception:
                pass
        # recommendations
        story.append(Spacer(1,10))
        story.append(Paragraph("<b>Structured clinical recommendations</b>", styles["H2"]))
        recs = summary.get("recommendations", [
            "This is an automated screening report. Clinical correlation is required.",
            "Consider MRI if focal delta index > 2 or extreme focal abnormalities.",
            "Follow-up in 3-6 months for moderate risk cases."
        ])
        for r in recs:
            story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,6))
        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        return None

# ----------------- Questionnaire utilities -----------------
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself"
]

AD8_QUESTIONS = [
    "Problems with judgment (e.g., problems making decisions, bad financial decisions)",
    "Less interest in hobbies/activities",
    "Repeats same things over and over",
    "Trouble learning how to use gadgets",
    "Forgets correct month or year",
    "Trouble handling complicated financial affairs",
    "Trouble remembering appointments",
    "Daily problems with thinking and memory"
]

def score_phq9(answers:List[int]):
    total = sum(answers)
    # thresholds (example)
    if total >= 20:
        level = "Severe"
    elif total >= 15:
        level = "Moderately Severe"
    elif total >= 10:
        level = "Moderate"
    elif total >= 5:
        level = "Mild"
    else:
        level = "Minimal"
    return total, level

def score_ad8(answers:List[int]):
    # AD8 binary scoring (0/1) — typical cutoff >=2 indicates cognitive impairment
    s = sum(answers)
    risk = "High" if s >= 2 else "Low"
    return s, risk

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# header
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},{'#7DD3FC'});padding:12px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:space-between">
  <div style="font-weight:700;font-size:22px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
    <div style="margin-right:12px;color:white;font-size:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:36px"/>' if LOGO_PATH_PNG.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# layout: left sidebar, main, right column
left_col, main_col, right_col = st.columns([1, 2.2, 1])

with left_col:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English", "Arabic"])
    patient_name = st.text_input("Patient Name (optional)")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown", "Male", "Female", "Other"])
    meds = st.text_area("Current meds (one per line)")
    labs = st.text_area("Relevant labs (B12, TSH, ...)")
    st.markdown("---")
    st.subheader("Upload")
    uploaded_files = st.file_uploader("Upload EDF file (.edf)", type=["edf"], accept_multiple_files=False)
    st.write("")
    process_btn = st.button("Process EDF(s) and Analyze")

with main_col:
    st.subheader("Console / Visualization")
    console_box = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_ans = []
    for i,q in enumerate(PHQ9_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1,2,3], index=0, key=f"phq_{i}")
        phq_ans.append(v)
    st.markdown("---")
    st.markdown("**AD8 (Cognitive)**")
    ad8_ans = []
    for i,q in enumerate(AD8_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1], index=0, key=f"ad8_{i}")
        ad8_ans.append(v)
    st.markdown("---")
    gen_pdf = st.button("Generate PDF report (for clinician)")

# Variables to hold processing results
processing_result = None
summary = {}

# If Process button clicked: read EDF and compute
if process_btn:
    console_box.info("📁 Saving and reading EDF file... please wait")
    progress_bar.progress(0.05)
    raw, msg = read_edf_bytes(uploaded_files) if uploaded_files else (None, "No file")
    if raw is None:
        status_placeholder.error(f"Error reading EDF: {msg}")
    else:
        progress_bar.progress(0.2)
        status_placeholder.success(f"✅ EDF loaded successfully.")
        # compute band powers
        try:
            console_box.info("🔬 Computing band powers and metrics...")
            res = compute_band_powers_from_raw(raw, bands=BANDS)
            progress_bar.progress(0.5)
            # produce topomap images for each band using channel means
            topo_imgs = {}
            if res and res.get("bands"):
                # compute per-band channel values: use relative power per channel
                chs = res["ch_names"]
                for b in BANDS:
                    vals = []
                    for ch in chs:
                        chinfo = res["bands"].get(ch, {})
                        vals.append(chinfo.get(f"{b}_rel", 0.0))
                    img_bytes = topomap_png_from_vals(vals, band_name=b)
                    if img_bytes:
                        topo_imgs[b] = img_bytes
            summary = {
                "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
                "bands": res.get("bands") if res else {},
                "metrics": res.get("metrics") if res else {},
                "topo_images": topo_imgs,
                "created": now_ts()
            }
            progress_bar.progress(0.8)
            console_box.success("✅ Processing complete.")
            status_placeholder.info("View results below. You can generate PDF or review SHAP if available.")
            processing_result = res
            progress_bar.progress(1.0)
        except Exception as e:
            console_box.error(f"Processing exception: {e}")
            traceback.print_exc()
            status_placeholder.error("Processing failed. See console for details.")

# Display results if available
if processing_result:
    st.markdown("---")
    st.subheader("Interactive Dashboard (Results)")
    df = make_band_table(processing_result["bands"])
    st.dataframe(df, height=320)
    # show topomaps
    st.markdown("### Topomaps")
    cols = st.columns(2)
    idx = 0
    for bname, img_bytes in summary.get("topo_images", {}).items():
        with cols[idx % 2]:
            st.image(img_bytes, caption=bname, use_column_width=True)
        idx += 1
    # display metrics
    st.markdown("### Key metrics")
    m = summary.get("metrics", {})
    st.write(m)

# Generate PDF if requested
if gen_pdf:
    if not summary:
        st.error("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")
    else:
        st.info("Generating PDF...")
        pdf_bytes = generate_pdf_report(summary, lang=("ar" if lang == "Arabic" else "en"), amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF generated.")
        else:
            st.error("PDF generation failed — ensure reportlab and fonts are available on the server.")

# SHAP visualization if file present
if SHAP_JSON.exists() and HAS_SHAP:
    try:
        with open(SHAP_JSON, "r") as f:
            shap_summary = json.load(f)
        # minimal bar chart of feature importances
        feats = shap_summary.get("features", [])
        vals = shap_summary.get("values", [])
        if feats and vals:
            st.subheader("SHAP feature importances")
            fig, ax = plt.subplots(figsize=(6,2))
            ax.barh(feats, vals)
            ax.set_xlabel("mean(|shap|)")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP display failed: {e}")
else:
    if not SHAP_JSON.exists():
        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
    elif not HAS_SHAP:
        st.info("shap library not installed — SHAP visualizations unavailable.")

# footer / notes
st.markdown("---")
st.markdown("**Notes:** Default language is English; Arabic is available for text sections and the PDF uses Amiri font if present.")
st.markdown("For best connectivity & microstate results install `mne` and `scipy` on the server.")
st.markdown("Place pre-trained models or shap summary in `models/` and `shap_summary.json` to enable model scoring and XAI.")

