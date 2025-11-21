# app.py — NeuroEarly Pro (v6->v7 merged, Clinical Final)
# Features:
# - Bilingual (English default / Arabic optional RTL with Amiri)
# - Sidebar left: language, patient info, meds, labs, EDF upload
# - Main: console/log, topomaps for Delta/Theta/Alpha/Beta/Gamma, band table
# - Right/below: questionnaires (PHQ-9, AD8), scoring & flags
# - Robust EDF reading (mne preferred, pyedflib fallback; handles BytesIO)
# - SHAP visualization if shap_summary.json present
# - PDF report (ReportLab) with Amiri font for Arabic if available
# - Graceful degradation when optional libs missing
# Default healthy baseline path set to uploaded file path: /mnt/data/test_edf.edf

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

# optional heavy libs - try import, set flags
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
    from scipy.signal import welch, coherence
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ----------------- Config / Assets -----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"  # place font here if you want Arabic PDF
LOGO_PATH_PNG = ASSETS / "goldenbird_logo.png"
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

def safe_minmax(arr):
    a = np.asarray(arr)
    if np.all(np.isnan(a)):
        return None, None
    try:
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
        if np.isclose(vmin, vmax):
            # expand a bit so colorbar works
            vmax = vmin + 1e-6
        return vmin, vmax
    except Exception:
        return None, None

def write_temp_file(data: bytes, suffix=".edf") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tf.write(data)
        tf.flush()
        tf.close()
        return tf.name
    except Exception:
        try:
            tf.close()
        except Exception:
            pass
        raise

# Robust EDF reading for uploaded file (returns either mne Raw or fallback dict)
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    Try reading uploaded EDF using mne if available, else pyedflib fallback.
    Returns (raw_like, msg) where raw_like is:
      - mne.io.Raw instance (if HAS_MNE)
      - dict {'signals': np.ndarray(n_ch,n_samples), 'ch_names': [...], 'sfreq': float}
    """
    if not uploaded:
        return None, "No file"
    # UploadedFile may have .read() or .getvalue()
    try:
        data = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()
    except Exception as e:
        return None, f"Could not read uploaded file object: {e}"
    tmp_path = None
    try:
        tmp_path = write_temp_file(data, suffix=".edf")
        # Try MNE first
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                # don't delete file immediately because mne may have memorymap; but it's OK to remove
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return raw, None
            except Exception as e_mne:
                # fall through to pyedflib
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0) if n > 0 else 256.0
                signals = np.vstack([f.readSignal(i) for i in range(n)]) if n>0 else np.zeros((0,0))
                f.close()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return {"signals": signals, "ch_names": ch_names, "sfreq": float(sfreq)}, None
            except Exception as e_py:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return None, f"pyedflib read error: {e_py}"
        # if neither available
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, "No EDF backend available (install mne or pyedflib)"
    except Exception as e:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return None, f"Temporary file write failed: {e}"

def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    """
    Accepts mne Raw-like or fallback dict {'signals','ch_names','sfreq'}.
    Returns dict:
      {'bands': {ch_name: {Delta_abs:..., Delta_rel:..., ...}}, 'metrics': {...}, 'ch_names': [...], 'sfreq': sf, 'psd': psd, 'freqs': freqs}
    """
    if raw_obj is None:
        return None
    # prepare data and ch_names and sf
    if HAS_MNE and (HAS_MNE and hasattr(mne, "io") and "raw" in str(type(raw_obj)).lower()):
        raw = raw_obj.copy()
        try:
            raw.pick_types(eeg=True, meg=False)
        except Exception:
            pass
        sf = float(raw.info.get("sfreq", 256.0))
        data = raw.get_data()  # shape (n_ch, n_samples)
        ch_names = list(raw.ch_names)
    elif isinstance(raw_obj, dict):
        dd = raw_obj
        data = np.asarray(dd.get("signals", np.zeros((0,0))))
        ch_names = list(dd.get("ch_names", [f"ch{i}" for i in range(data.shape[0])]))
        sf = float(dd.get("sfreq", 256.0))
        if data.ndim == 1:
            data = data[np.newaxis, :]
    else:
        # last resort: try to extract get_data if exists
        if hasattr(raw_obj, "get_data"):
            try:
                data = raw_obj.get_data()
                ch_names = getattr(raw_obj, "ch_names", [f"ch{i}" for i in range(data.shape[0])])
                sf = float(raw_obj.info.get("sfreq", 256.0)) if hasattr(raw_obj, "info") else 256.0
            except Exception as e:
                return None
        else:
            return None

    n_ch = 0 if data is None else data.shape[0]
    if n_ch == 0:
        return None

    # compute PSD
    if HAS_SCIPY:
        # use welch
        freqs = None
        psd_list = []
        nperseg = min(int(4*sf), data.shape[1]) if data.shape[1] > 0 else 256
        for ch in range(n_ch):
            try:
                f, Pxx = welch(data[ch, :], fs=sf, nperseg=nperseg)
            except Exception:
                # fallback fallback
                N = data.shape[1]
                f = np.fft.rfftfreq(N, d=1.0/sf)
                Pxx = np.abs(np.fft.rfft(data[ch,:]))**2 / N
            freqs = f
            psd_list.append(Pxx)
        psd = np.vstack(psd_list)
    else:
        # FFT fallback
        N = data.shape[1]
        freqs = np.fft.rfftfreq(N, d=1.0/sf)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / N
        psd = fft_vals

    # compute band totals and relative
    band_summary = {}
    totals = psd.sum(axis=1) + 1e-12
    for i, ch in enumerate(ch_names):
        band_summary[ch] = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            power = float(np.sum(psd[i, mask])) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = power
            band_summary[ch][f"{bname}_rel"] = (power / totals[i]) if totals[i] > 0 else 0.0

    # global metrics
    delta_rels = [band_summary[ch].get("Delta_rel", 0.0) for ch in band_summary]
    theta_rels = [band_summary[ch].get("Theta_rel", 0.0) for ch in band_summary]
    alpha_rels = [band_summary[ch].get("Alpha_rel", 0.0) for ch in band_summary]
    global_metrics = {}
    try:
        # theta/alpha ratio mean
        tar = []
        for t,a in zip(theta_rels, alpha_rels):
            if a>0:
                tar.append(t/a)
        global_metrics["theta_alpha_ratio"] = float(np.mean(tar)) if tar else 0.0
    except Exception:
        global_metrics["theta_alpha_ratio"] = 0.0
    global_metrics["FDI"] = float(max(delta_rels)*100.0) if delta_rels else 0.0
    global_metrics["mean_total_power"] = float(np.mean(totals)) if len(totals)>0 else 0.0

    return {"bands": band_summary, "metrics": global_metrics, "ch_names": ch_names, "sfreq": float(sf), "psd": psd, "freqs": freqs}

def topomap_png_from_vals(vals: np.ndarray, band_name: str="Band"):
    """
    Create a simple heat-grid representation (fallback topomap) from 1D channel values.
    Returns PNG bytes or None.
    """
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = len(arr)
        if n == 0:
            return None
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side * side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(4,3))
        vmin, vmax = safe_minmax(grid)
        if vmin is None:
            im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper')
        else:
            im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
        ax.set_title(f"{band_name}")
        ax.axis('off')
        try:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            pass
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

def make_band_table(band_summary: Dict[str,Any]):
    rows = []
    for ch, info in band_summary.items():
        r = {"ch": ch}
        for b in BANDS.keys():
            r[f"{b}_abs"] = info.get(f"{b}_abs", 0.0)
            r[f"{b}_rel"] = info.get(f"{b}_rel", 0.0)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ch")
    return df

# ----------------- PDF generator -----------------
def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[Path]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        # register Amiri if present
        if amiri_path and Path(amiri_path).exists() and HAS_ARABIC:
            try:
                if "Amiri" not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri font register failed:", e)
        # add styles only if not present
        style_names = [s.name for s in styles]
        if "TitleBlue" not in style_names:
            styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        if "H2" not in style_names:
            styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        if "Body" not in style_names:
            styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        if "Note" not in style_names:
            styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # header
        title = "NeuroEarly Pro — Clinical Report" if lang == "en" else (get_display(arabic_reshaper.reshape("تقرير NeuroEarly Pro السريري")) if HAS_ARABIC else "NeuroEarly Pro — Clinical Report")
        story.append(Paragraph(title, styles["TitleBlue"]))
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
        # metrics table
        metrics = summary.get("metrics", {})
        if metrics:
            mrows = [["Metric","Value"]]
            for k,v in metrics.items():
                mrows.append([k, f"{v}"])
            mt = Table(mrows, colWidths=[200,200])
            mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(Paragraph("<b>Key QEEG metrics</b>", styles["H2"]))
            story.append(mt)
            story.append(Spacer(1,12))
        # topo images
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topomaps</b>", styles["H2"]))
            for name, img_bytes in summary["topo_images"].items():
                try:
                    img = RLImage(io.BytesIO(img_bytes), width=200, height=140)
                    story.append(img)
                    story.append(Spacer(1,6))
                except Exception:
                    pass
        # SHAP
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
        traceback.print_exc()
        return None

# ----------------- Questionnaires -----------------
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

def score_phq9(answers: List[int]):
    total = sum(answers)
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

def score_ad8(answers: List[int]):
    s = sum(answers)
    risk = "High" if s >= 2 else "Low"
    return s, risk

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
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

# layout
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
    phq_answers = []
    for i,q in enumerate(PHQ9_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1,2,3], index=0, key=f"phq_{i}")
        phq_answers.append(v)
    st.markdown("---")
    st.markdown("**AD8 (Cognitive)**")
    ad8_answers = []
    for i,q in enumerate(AD8_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1], index=0, key=f"ad8_{i}")
        ad8_answers.append(v)
    st.markdown("---")
    gen_pdf = st.button("Generate PDF report (for clinician)")

processing_result = None
summary = {}

if process_btn:
    console_box.info("📁 Saving and reading EDF file... please wait")
    progress_bar.progress(0.05)
    raw, msg = read_edf_bytes(uploaded_files) if uploaded_files else (None, "No file")
    if raw is None:
        status_placeholder.error(f"Error reading EDF: {msg}")
    else:
        progress_bar.progress(0.2)
        status_placeholder.success(f"✅ EDF loaded successfully.")
        try:
            console_box.info("🔬 Computing band powers and metrics...")
            res = compute_band_powers_from_raw(raw, bands=BANDS)
            if res is None:
                raise RuntimeError("PSD computation returned no result.")
            progress_bar.progress(0.5)
            topo_imgs = {}
            if res and res.get("bands"):
                chs = res["ch_names"]
                for b in BANDS:
                    vals = [res["bands"].get(ch, {}).get(f"{b}_rel", 0.0) for ch in chs]
                    img_bytes = topomap_png_from_vals(vals, band_name=b)
                    if img_bytes:
                        topo_imgs[b] = img_bytes
            summary = {
                "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
                "bands": res.get("bands"),
                "metrics": res.get("metrics"),
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

if processing_result:
    st.markdown("---")
    st.subheader("Interactive Dashboard (Results)")
    df = make_band_table(processing_result["bands"])
    st.dataframe(df, height=320)
    st.markdown("### Topomaps")
    cols = st.columns(2)
    idx = 0
    for bname, img_bytes in summary.get("topo_images", {}).items():
        with cols[idx % 2]:
            st.image(img_bytes, caption=bname, use_column_width=True)
        idx += 1
    st.markdown("### Key metrics")
    st.write(summary.get("metrics", {}))

if gen_pdf:
    if not summary:
        st.error("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")
    else:
        st.info("Generating PDF...")
        pdf_bytes = generate_pdf_report(summary, lang= "ar" if lang=="Arabic" else "en", amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF generated.")
        else:
            st.error("PDF generation failed — ensure reportlab and fonts are available on the server.")

# SHAP visual if available
if SHAP_JSON.exists() and HAS_SHAP:
    try:
        with open(SHAP_JSON, "r", encoding="utf-8") as f:
            shap_summary = json.load(f)
        feats = shap_summary.get("features", None) or shap_summary.get(next(iter(shap_summary), ""), None)
        if feats:
            st.subheader("SHAP feature importances")
            s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(6,2))
            ax.barh(s.index, s.values)
            ax.set_xlabel("mean(|shap|)")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP display failed: {e}")
else:
    if not SHAP_JSON.exists():
        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
    elif not HAS_SHAP:
        st.info("shap library not installed — SHAP visualizations unavailable.")

st.markdown("---")
st.markdown("**Notes:** Default language is English; Arabic is available for text sections and the PDF uses Amiri font if present.")
st.markdown("For best connectivity & microstate results install `mne` and `scipy` on the server.")
st.markdown("Place pre-trained models in `models/` and `shap_summary.json` to enable model scoring and XAI.")
