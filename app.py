# app.py — NeuroEarly Pro (Streamlit Cloud ready, MNE-enabled)
# v1.0 — Clinical edition (Bilingual, PDF, SHAP, Connectivity, Topomaps)

import os
import io
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

# Optional heavy libs
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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
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
    from scipy.signal import welch, butter, sosfiltfilt, iirnotch, coherence
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- Config / Assets ----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"
# default baseline (use this path only for local testing - on Cloud put baseline in repo)
DEFAULT_BASELINE = Path("/mnt/data/test_edf.edf")  # <-- your uploaded test file path (keeps compatibility)

APP_TITLE = "NeuroEarly Pro — Clinical"
BLUE = "#0b63d6"
LIGHT_BG = "#eef7ff"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# ---------------- Helpers ----------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_mkdir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def write_temp_bytes(data: bytes, suffix=".edf") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(data)
        tf.flush()
        return tf.name

# Robust EDF reader (mne preferred, pyedflib fallback)
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    uploaded: streamlit UploadedFile or path string
    returns: raw (mne Raw) or fallback dict {'signals','ch_names','sfreq'}
    """
    if uploaded is None:
        return None, "No file provided"
    # If it's str/Path pointing to a file
    if isinstance(uploaded, (str, Path)):
        p = str(uploaded)
        try:
            if HAS_MNE:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                return raw, None
            elif HAS_PYEDF:
                f = pyedflib.EdfReader(p)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                signals = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
            else:
                return None, "No EDF backend available (install mne or pyedflib)"
        except Exception as e:
            return None, f"Read error: {e}"
    # Else assume it's UploadedFile object
    try:
        raw_bytes = uploaded.read()
    except Exception as e:
        try:
            raw_bytes = uploaded.getvalue()
        except Exception as ee:
            return None, f"Uploaded file access error: {ee}"
    # write to temp file to avoid BytesIO issues with mne
    tmp_path = None
    try:
        tmp_path = write_temp_bytes(raw_bytes, suffix=".edf")
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                try: os.unlink(tmp_path)
                except Exception: pass
                return raw, None
            except Exception:
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                signals = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                try: os.unlink(tmp_path)
                except Exception: pass
                return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
            except Exception as e:
                try: os.unlink(tmp_path)
                except Exception: pass
                return None, f"pyedflib read error: {e}"
        try: os.unlink(tmp_path)
        except Exception: pass
        return None, "No EDF backend installed"
    except Exception as e:
        return None, f"Temp write error: {e}"

# Basic notch + bandpass denoising (on raw numpy array)
def denoise_signal_array(arr: np.ndarray, sf: float):
    """
    arr: n_channels x n_samples
    returns denoised arr (same shape)
    """
    if not HAS_SCIPY:
        return arr
    try:
        # notch 50/60 Hz
        filtered = arr.copy()
        for line_freq in (50.0, 60.0):
            try:
                q = 30.0
                b,a = iirnotch(line_freq, q, sf)
                filtered = sosfiltfilt((b,a), filtered, axis=1)
            except Exception:
                pass
        # bandpass 0.5-45 Hz
        try:
            low = 0.5; high = 45.0
            sos_b, sos_a = butter(4, [low/(sf/2.0), high/(sf/2.0)], btype='band', output='sos')
            filtered = sosfiltfilt(sos_b, filtered, axis=1)
        except Exception:
            pass
        return filtered
    except Exception:
        return arr

# compute band powers using Welch (scipy) or fallback FFT
def compute_band_powers_from_rawobj(raw_obj, bands=BANDS) -> Optional[Dict[str,Any]]:
    """
    Accepts mne Raw or fallback dict {'signals','ch_names','sfreq'}.
    Returns dict with band powers by channel and metrics.
    """
    try:
        if raw_obj is None:
            return None
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            raw = raw_obj.copy().pick_types(eeg=True, meg=False, stim=False)
            sf = float(raw.info['sfreq'])
            data = raw.get_data()  # shape (n_ch, n_samples)
            ch_names = raw.ch_names
        else:
            # fallback dict
            dd = raw_obj
            data = np.asarray(dd["signals"])
            ch_names = dd["ch_names"]
            sf = float(dd["sfreq"])
            if data.ndim == 1:
                data = data[np.newaxis, :]
        # denoise
        data = denoise_signal_array(data, sf)
        n_ch = data.shape[0]
        # PSD
        if HAS_SCIPY:
            from scipy.signal import welch
            freqs = None
            psd_list = []
            for ch in range(n_ch):
                f, Pxx = welch(data[ch,:], fs=sf, nperseg=min(2048, data.shape[1]))
                psd_list.append(Pxx)
                freqs = f
            psd = np.vstack(psd_list)
        else:
            n = data.shape[1]
            freqs = np.fft.rfftfreq(n, d=1.0/sf)
            fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
            psd = fft_vals
        # compute band values
        band_summary = {}
        for i,ch in enumerate(ch_names):
            band_summary[ch] = {}
            total = float(np.trapz(psd[i,:], freqs)) if freqs is not None else float(psd[i,:].sum())
            for bname,(fmin,fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs < fmax)
                val = float(np.trapz(psd[i,mask], freqs[mask])) if mask.any() else 0.0
                band_summary[ch][f"{bname}_abs"] = val
                band_summary[ch][f"{bname}_rel"] = (val / total) if total > 0 else 0.0
        # global metrics
        global_metrics = {}
        # theta/alpha ratio: mean across channels (theta_rel / alpha_rel)
        tas = []
        for ch in band_summary:
            a_rel = band_summary[ch].get("Alpha_rel", 0.0)
            t_rel = band_summary[ch].get("Theta_rel", 0.0)
            if a_rel > 0:
                tas.append(t_rel / a_rel)
        global_metrics["theta_alpha_ratio"] = float(np.mean(tas)) if tas else 0.0
        # FDI: focal delta index (max delta_rel / mean delta_rel)
        delta_rels = [band_summary[ch].get("Delta_rel", 0.0) for ch in band_summary]
        mean_delta = float(np.mean(delta_rels)) if delta_rels else 0.0
        max_delta = float(np.max(delta_rels)) if delta_rels else 0.0
        global_metrics["FDI"] = (max_delta / (mean_delta + 1e-12)) if mean_delta > 0 else 0.0
        global_metrics["mean_connectivity_alpha"] = 0.0  # filled later if connectivity computed
        return {"bands": band_summary, "metrics": global_metrics, "ch_names": ch_names, "sfreq": sf, "psd": psd, "freqs": freqs, "raw_data": data}
    except Exception as e:
        st.error(f"compute_band_powers error: {e}")
        traceback.print_exc()
        return None

# fallback grid topomap generator (returns PNG bytes)
def topomap_png_from_vals(vals: np.ndarray, band_name:str="Band"):
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = len(arr)
        side = int(np.ceil(np.sqrt(n)))
        grid = np.full(side*side, np.nan)
        grid[:n] = arr
        grid = grid.reshape(side, side)
        fig,ax = plt.subplots(figsize=(3.2,2.4))
        vmin = np.nanmin(grid) if np.isfinite(np.nanmin(grid)) else 0.0
        vmax = np.nanmax(grid) if np.isfinite(np.nanmax(grid)) else vmin+1.0
        im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
        ax.set_title(band_name, fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# connectivity compute (mne or coherence or corr)
def compute_connectivity_matrix(raw_obj, sf, ch_names, band=(8.0,13.0)):
    try:
        lo, hi = band
        # try mne connectivity if available
        if HAS_MNE:
            try:
                from mne.connectivity import spectral_connectivity
                # if raw_obj is mne Raw:
                if isinstance(raw_obj, mne.io.BaseRaw):
                    data, times = raw_obj.get_data(return_times=True), None
                else:
                    data = raw_obj["raw_data"] if isinstance(raw_obj, dict) and "raw_data" in raw_obj else None
                if data is None:
                    return None, None
                # spectral_connectivity expects epochs or array-like: (n_signals, n_times)
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    [data], method='coh', mode='fourier', sfreq=sf,
                    fmin=lo, fmax=hi, faverage=True, verbose=False)
                # con shape (n_signals, n_signals)
                conn_mat = np.squeeze(con)
                # create image
                fig,ax = plt.subplots(figsize=(4,3))
                im = ax.imshow(conn_mat, cmap='viridis', vmin=0, vmax=1)
                ax.set_title("Connectivity (alpha)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                return conn_mat, buf.getvalue()
            except Exception:
                pass
        # fallback: coherence using scipy
        if HAS_SCIPY:
            try:
                n_ch = len(ch_names)
                conn = np.zeros((n_ch, n_ch))
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        try:
                            f, Cxy = coherence(raw_obj["raw_data"][i], raw_obj["raw_data"][j], fs=sf, nperseg=min(2048, raw_obj["raw_data"].shape[1]))
                            mask = (f >= lo) & (f <= hi)
                            mean_coh = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                        except Exception:
                            mean_coh = 0.0
                        conn[i,j] = mean_coh; conn[j,i] = mean_coh
                fig,ax = plt.subplots(figsize=(4,3))
                im = ax.imshow(conn, cmap='viridis', vmin=0, vmax=1)
                ax.set_title("Connectivity (alpha)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                return conn, buf.getvalue()
            except Exception:
                return None, None
        # fallback corrcoeff
        try:
            x = raw_obj["raw_data"]
            x = (x - x.mean(axis=1,keepdims=True)) / (x.std(axis=1,keepdims=True)+1e-12)
            conn = np.corrcoef(x)
            fig,ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(conn, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title("Connectivity (corr)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return conn, buf.getvalue()
        except Exception:
            return None, None
    except Exception:
        return None, None

# SHAP render from shap_summary.json
def render_shap_from_json(shap_path: Path, model_key_hint="depression_global"):
    if not shap_path.exists() or not HAS_SHAP:
        return None
    try:
        with open(shap_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        key = model_key_hint if model_key_hint in sj else next(iter(sj.keys()))
        feats = sj.get(key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig,ax = plt.subplots(figsize=(6,3))
        s.plot.barh(ax=ax)
        ax.set_xlabel("mean(|shap value|)")
        ax.invert_yaxis()
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# PDF report generator (ReportLab), bilingual support
def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[Path]=None) -> Optional[bytes]:
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
            except Exception:
                base_font = "Helvetica"
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1,6))
        # patient info table
        pi = summary.get("patient_info", {})
        tdata = [["Patient ID", pi.get("id","-"), "DOB", pi.get("dob","-")],
                 ["Report date", summary.get("created", now_ts()), "Exam", pi.get("exam","EEG")]]
        t = Table(tdata, colWidths=[80,140,80,140])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t); story.append(Spacer(1,8))
        # Metrics
        metrics = summary.get("metrics", {})
        if metrics:
            mrows = [["Metric","Value"]]
            for k,v in metrics.items():
                mrows.append([k, f"{float(v):.4f}" if isinstance(v,(int,float)) else str(v)])
            mt = Table(mrows, colWidths=[200,200])
            mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(Paragraph("<b>Key QEEG metrics</b>", styles["H2"])); story.append(mt); story.append(Spacer(1,8))
        # Topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
            imgs = []
            for band, img in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(img), width=200, height=120))
                except Exception:
                    pass
            # two per row
            row=[]
            for i,im in enumerate(imgs):
                row.append(im)
                if (i%2)==1:
                    story.append(Table([row], colWidths=[220,220])); row=[]
            if row:
                story.append(Table([row], colWidths=[220,220]))
            story.append(Spacer(1,8))
        # Connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("<b>Functional Connectivity</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=420, height=240))
            except Exception:
                pass
            story.append(Spacer(1,8))
        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("<b>Explainable AI (SHAP)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=420, height=200))
            except Exception:
                pass
            story.append(Spacer(1,8))
        # Recommendations
        story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
        for r in summary.get("recommendations", ["Automated screening — clinical correlation required."]):
            story.append(Paragraph(r, styles["Body"])); story.append(Spacer(1,4))
        story.append(Spacer(1,8))
        story.append(Paragraph("Prepared by Golden Bird LLC", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        traceback.print_exc()
        return None

# ---------------- UI / App Start ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# header bar
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},#7DD3FC); padding:10px; border-radius:6px; color:white; display:flex; align-items:center; justify-content:space-between;">
  <div style="font-size:20px; font-weight:700;">🧠 {APP_TITLE}</div>
  <div style="display:flex; align-items:center;">
    <div style="margin-right:12px;">Prepared by Golden Bird LLC</div>
    {"<img src='assets/goldenbird_logo.png' style='height:36px'/>" if LOGO_PATH.exists() else ""}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# layout
left_col, main_col, right_col = st.columns([1.0, 2.2, 1.0])

with left_col:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English", "Arabic"], key="lang_select")
    lang_code = "ar" if lang.startswith("A") else "en"
    patient_id = st.text_input("Patient ID", key="pid")
    dob = st.date_input("DOB", value=date(1980,1,1), max_value=date(2025,12,31), key="dob")
    sex = st.selectbox("Sex", ["Unknown","Male","Female","Other"], key="sex")
    meds = st.text_area("Current meds (one per line)", key="meds", height=120)
    labs = st.text_area("Relevant labs (B12, TSH, ...) (you may paste lab text)", key="labs", height=120)
    st.markdown("---")
    st.subheader("Upload EDF")
    uploaded = st.file_uploader("Upload EDF (.edf)", type=["edf"], accept_multiple_files=False, key="edf_up")
    st.markdown("")
    process_btn = st.button("Process EDF & Generate Results", key="process_btn")

with main_col:
    st.subheader("Console / Visualization")
    console = st.empty()
    progress = st.progress(0.0)
    results_placeholder = st.empty()

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_res = []
    for i,question in enumerate([
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking so slowly or being fidgety/restless",
        "Thoughts that you would be better off dead or of hurting yourself"
    ]):
        v = st.selectbox(f"PHQ Q{i+1}", ["0 - Not at all","1 - Several days","2 - More than half the days","3 - Nearly every day"], index=0, key=f"phq_{i}")
        phq_res.append(int(v.split(" - ")[0]))
    st.markdown("---")
    st.markdown("**AD8 (Cognitive: short)**")
    ad8_res = []
    for i,q in enumerate([
        "Problems with judgment (making decisions)",
        "Less interest in hobbies/activities",
        "Repeats things over and over",
        "Trouble learning to use gadgets",
        "Forgets correct month or year",
        "Trouble handling financial affairs",
        "Trouble remembering appointments",
        "Daily problems with thinking and memory"
    ]):
        v = st.selectbox(f"AD8 Q{i+1}", ["0 - No", "1 - Yes"], index=0, key=f"ad8_{i}")
        ad8_res.append(int(v.split(" - ")[0]))
    st.markdown("---")
    gen_pdf_btn = st.button("Generate PDF (current results)", key="gen_pdf")

# session state holders
if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = {}

# Process button logic
if process_btn:
    console.info("Reading EDF and preparing data..."); progress.progress(0.05)
    raw_obj, msg = read_edf_bytes(uploaded) if uploaded else (None, "No file uploaded")
    if raw_obj is None:
        # if no upload, try default baseline if exists (local testing only)
        if DEFAULT_BASELINE.exists():
            raw_obj, msg = read_edf_bytes(str(DEFAULT_BASELINE))
        if raw_obj is None:
            console.error(f"EDF read failed: {msg}"); st.error(f"EDF read failed: {msg}"); progress.progress(0.0)
    if raw_obj is not None:
        progress.progress(0.2)
        console.info("Computing band powers and metrics..."); progress.progress(0.35)
        res = compute_band_powers_from_rawobj(raw_obj, bands=BANDS)
        if res is None:
            st.error("Band computation failed"); progress.progress(0.0)
        else:
            # compute connectivity
            try:
                conn_mat, conn_img = compute_connectivity_matrix(res, res["sfreq"], res["ch_names"], band=BANDS.get("Alpha",(8.0,13.0)))
                if conn_mat is not None:
                    res["metrics"]["mean_connectivity_alpha"] = float(np.nanmean(conn_mat))
                    res["connectivity_image"] = conn_img
                else:
                    res["connectivity_image"] = None
            except Exception:
                res["connectivity_image"] = None
            # create topomaps
            topo_imgs = {}
            for b in BANDS.keys():
                vals = []
                for ch in res["ch_names"]:
                    vals.append(res["bands"].get(ch, {}).get(f"{b}_rel", 0.0))
                img = topomap_png_from_vals(np.array(vals), band_name=b)
                if img:
                    topo_imgs[b] = img
            res["topo_images"] = topo_imgs
            # compute FDI
            delta_vals = np.array([res["bands"][ch].get("Delta_rel",0.0) for ch in res["ch_names"]])
            mean_delta = float(np.nanmean(delta_vals)) if delta_vals.size else 0.0
            max_delta = float(np.nanmax(delta_vals)) if delta_vals.size else 0.0
            res["metrics"]["FDI"] = (max_delta / (mean_delta+1e-12)) if mean_delta>0 else 0.0
            # SHAP
            shap_img = None
            if SHAP_JSON.exists() and HAS_SHAP:
                shap_img = render_shap_from_json(SHAP_JSON, model_key_hint="depression_global")
            res["shap_img"] = shap_img
            # clinical scores
            phq_score = sum(phq_res)
            ad_score = sum(ad8_res)
            res["clinical"] = {"phq_score": phq_score, "ad_score": ad_score}
            # heuristics for final risk
            ta_val = res["metrics"].get("theta_alpha_ratio", 0.0)
            ta_norm = min(1.0, ta_val / 2.0)
            phq_norm = min(1.0, phq_score / 27.0)
            ad_norm = min(1.0, ad_score / 24.0)
            final_depression = 0.6*phq_norm + 0.4*ta_norm
            final_alzheimer = 0.6*ad_norm + 0.4*ta_norm
            final_tumor = min(1.0, max(0.0, (res["metrics"].get("FDI",0.0) - 2.0)/5.0))
            res["final_risks"] = {"Depression": float(final_depression), "Alzheimer": float(final_alzheimer), "Tumor": float(final_tumor)}
            # normative comparison (if baseline available)
            if DEFAULT_BASELINE.exists():
                try:
                    base_raw, _ = read_edf_bytes(str(DEFAULT_BASELINE))
                    base_res = compute_band_powers_from_rawobj(base_raw, bands=BANDS) if base_raw is not None else None
                    if base_res is not None:
                        # create normative bar (theta/alpha patient vs baseline)
                        p_ta = ta_val
                        b_ta = base_res["metrics"].get("theta_alpha_ratio", 0.0)
                        fig,ax = plt.subplots(figsize=(5.5,2.2))
                        ax.bar([0,1],[p_ta,b_ta], color=[BLUE,"#2ca02c"])
                        ax.set_xticks([0,1]); ax.set_xticklabels(["Patient","Baseline"])
                        ax.set_title("Theta/Alpha: patient vs baseline")
                        buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                        res["normative_bar"] = buf.getvalue()
                except Exception:
                    res["normative_bar"] = None
            # recommendations based on simple thresholds
            recs = []
            if final_alzheimer > 0.25:
                recs.append("Consider referral for cognitive evaluation and AD8 / neuropsych testing.")
            if final_depression > 0.3:
                recs.append("Consider psychiatric evaluation and PHQ-9 follow-up; monitor suicidality.")
            if final_tumor > 0.3:
                recs.append("FDI elevated — consider MRI to rule out focal lesion.")
            if not recs:
                recs.append("No immediate red flags; continue routine follow-up.")
            res["recommendations"] = recs
            # save to session
            st.session_state["last_summary"] = {
                "patient_info": {"id": patient_id, "dob": str(dob), "name": "", "exam":"EEG"},
                "bands": res["bands"],
                "metrics": res["metrics"],
                "topo_images": res.get("topo_images", {}),
                "connectivity_image": res.get("connectivity_image"),
                "normative_bar": res.get("normative_bar"),
                "shap_img": res.get("shap_img"),
                "clinical": res.get("clinical"),
                "final_risks": res.get("final_risks"),
                "recommendations": res.get("recommendations"),
                "created": now_ts()
            }
            progress.progress(1.0)
            console.success("Processing complete.")
            # display results
            with results_placeholder:
                st.markdown("## Results")
                col1, col2 = st.columns([2,1])
                with col1:
                    # show band table
                    rows = []
                    for ch in res["ch_names"]:
                        d = res["bands"].get(ch, {})
                        rows.append({
                            "channel": ch,
                            **{f"{b}_rel": d.get(f"{b}_rel",0.0) for b in BANDS.keys()}
                        })
                    df_show = pd.DataFrame(rows).set_index("channel")
                    st.dataframe(df_show.style.format("{:.4f}"), height=300)
                    st.markdown("### Topomaps")
                    tcols = st.columns(2)
                    idx=0
                    for b,img in res["topo_images"].items():
                        try:
                            tcols[idx%2].image(img, caption=b, use_column_width=True)
                        except Exception:
                            pass
                        idx+=1
                    if res.get("normative_bar"):
                        st.markdown("### Normative comparison")
                        st.image(res["normative_bar"], width=520)
                with col2:
                    st.markdown("### Key metrics")
                    st.write(res["metrics"])
                    st.markdown("### Final Risks")
                    for k,v in res["final_risks"].items():
                        st.metric(k, f"{v*100:.1f}%")
                    if res.get("connectivity_image"):
                        st.markdown("### Connectivity (alpha)")
                        st.image(res["connectivity_image"], width=300)
                    if res.get("shap_img"):
                        st.markdown("### SHAP (XAI)")
                        st.image(res["shap_img"], width=300)
                    st.markdown("### Recommendations")
                    for r in res["recommendations"]:
                        st.write("- " + r)
    # end process

# Generate PDF button
if gen_pdf_btn:
    if not st.session_state.get("last_summary"):
        st.error("No processed results to generate PDF. Run analysis first.")
    else:
        s = st.session_state["last_summary"]
        pdf_bytes = generate_pdf_report(s, lang=("ar" if lang_code=="ar" else "en"), amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download PDF", pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation failed — ensure reportlab & fonts are installed in requirements.txt and present in repo.")

# SHAP prompt if missing
if not SHAP_JSON.exists():
    st.info("No shap_summary.json found in repo root — place shap_summary.json to enable SHAP XAI visuals.")
elif not HAS_SHAP:
    st.info("shap library not installed — SHAP visuals unavailable. Add 'shap' to requirements.txt and redeploy.")

st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro")
