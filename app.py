# app.py -- NeuroEarly Pro (part 1/3)
# Minimal required files in repo: Amiri-Regular.ttf, assets/goldenbird_logo.png, requirements.txt with mne, pyedflib, shap, reportlab, matplotlib, etc.
# Default language: English (can select Arabic)
# Author: generated for user (Golden Bird LLC)

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
matplotlib.use("Agg")  # ensure non-GUI backend on servers
import matplotlib.pyplot as plt

import streamlit as st

# Optional heavy libs
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_SHAP = False
HAS_REPORTLAB = False
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
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ---------------------------
# Constants & paths
# ---------------------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"  # adjust if in subfolder
LOGO_PATH = ASSETS / "goldenbird_logo.png"  # ensure exists
SHAP_JSON = ROOT / "shap_summary.json"  # optional shap results file
MODEL_PATH = ROOT / "models"  # placeholder directory for any pre-trained models

# Visual constants
LIGHT_BG = "#eaf6ff"
PRIMARY_BLUE = "#0b63d6"  # use for header accents
PDF_BLUE = "#0b63d6"

# Frequency bands (standard)
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Ensure session state keys exist
if "results" not in st.session_state:
    st.session_state["results"] = []  # list of dicts per-file
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# ---------------------------
# Helper functions: formatting, timestamp
# ---------------------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def load_shap_json(path: Path):
    if path.exists() and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return None
    return None

# Safe numeric formatting
def fmt(x, prec=4):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

# ---------------------------
# Basic EEG helpers (lightweight)
# ---------------------------
def read_edf_bytes(uploaded) -> Tuple[Optional[mne.io.Raw], Optional[str]]:
    """
    Try reading uploaded EDF using mne if available, else return None.
    Returns (raw, msg)
    """
    if not uploaded:
        return None, "No file"
    buf = io.BytesIO(uploaded.getvalue())
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(buf, preload=True, verbose=False)
            return raw, None
        else:
            return None, "mne not available"
    except Exception as e:
        return None, str(e)

def compute_band_powers(raw, bands=BANDS):
    """
    Compute relative band powers per channel using Welch PSD (via numpy/mne)
    Returns df with columns like 'Delta_rel', 'Theta_rel', ...
    """
    if raw is None:
        return pd.DataFrame()
    try:
        sf = int(raw.info["sfreq"])
        picks = mne.pick_types(raw.info, eeg=True, meg=False, include=[])
        data = raw.get_data(picks=picks)  # shape (n_ch, n_times)
        ch_names = [raw.ch_names[p] for p in picks]
        # use mne.time_frequency.psd_welch if available
        psds, freqs = mne.time_frequency.psd_welch(raw, picks=picks, fmin=1.0, fmax=45.0, verbose=False)
        # psds shape (n_channels, n_freqs)
        df = []
        for ch_idx, ch in enumerate(ch_names):
            pxx = psds[ch_idx]
            total = np.trapz(pxx, freqs) if freqs.size > 0 else 0.0
            row = {"channel": ch}
            for bname, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                p_band = np.trapz(pxx[mask], freqs[mask]) if mask.sum() > 0 else 0.0
                rel = (p_band / total) if total > 0 else 0.0
                row[f"{bname}_abs"] = float(p_band)
                row[f"{bname}_rel"] = float(rel)
            df.append(row)
        df = pd.DataFrame(df)
        return df
    except Exception as e:
        print("compute_band_powers error:", e)
        return pd.DataFrame()

def compute_theta_alpha_ratio(dfbands):
    """Compute global theta/alpha ratio from dfbands (per-channel relative powers)"""
    try:
        if dfbands.empty:
            return None
        # Mean across channels
        theta = dfbands["Theta_rel"].mean()
        alpha = dfbands["Alpha_rel"].mean()
        return float(theta / (alpha + 1e-12))
    except Exception:
        return None

# ---------------------------
# Visualization helpers (matplotlib) -> return PNG bytes
# ---------------------------
def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def bar_comparison_chart(theta_alpha, alpha_asym):
    """Generate bar chart comparing Theta/Alpha and Alpha asymmetry vs normative ranges."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    metrics = ["Theta/Alpha", "Alpha Asym (F3-F4)"]
    values = [theta_alpha if theta_alpha is not None else 0.0, alpha_asym if alpha_asym is not None else 0.0]
    # Normative ranges (example)
    norm_low = [0.3, -0.02]   # lower bound for 'healthy'
    norm_high = [1.1, 0.02]   # upper bound for 'healthy'
    bars = ax.bar(metrics, values, color=PRIMARY_BLUE)
    # show healthy box as rectangle around each bar
    for i in range(len(metrics)):
        ax.add_patch(plt.Rectangle((i-0.2, norm_low[i]), 0.4, norm_high[i]-norm_low[i],
                                   facecolor="#ffffff", edgecolor="gray", alpha=0.3, linewidth=1))
        # danger zone top
        ax.add_patch(plt.Rectangle((i-0.2, norm_high[i]), 0.4, max(0.02, values[i]-norm_high[i]),
                                   facecolor="#ffcccc", edgecolor=None, alpha=0.4))
    ax.set_ylabel("Value")
    ax.set_title("Normative comparison")
    ax.axhline(0, color="k", linewidth=0.4)
    return fig_to_png_bytes(fig)

def topomap_image_for_band(vals, ch_names, band_name="Alpha", sfreq=256):
    """
    Generate a simple topomap per band using matplotlib. If MNE available provide interpolation.
    vals: array of channel values (len == n_channels)
    ch_names: list of channel names
    """
    try:
        fig = plt.figure(figsize=(4.0, 3.0))
        ax = fig.add_subplot(111)
        # Fallback: simple bar of channels if no montage available
        x = np.arange(len(ch_names))
        ax.bar(x, vals, color=PRIMARY_BLUE)
        ax.set_xticks(x); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_title(f"{band_name} topography (approx.)")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("topomap error:", e)
        return None

# ---------------------------
# UI: Sidebar (patient, settings, uploads)
# ---------------------------

st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide", initial_sidebar_state="expanded")

# Header (full-width)
st.markdown(
    f"""
    <div style="background: linear-gradient(90deg, {PRIMARY_BLUE}, #2fa1ff); padding:12px; border-radius:8px; color:white;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-size:20px;font-weight:700;">🧠 NeuroEarly Pro — Clinical & Research</div>
            <div style="font-size:12px;">Prepared by Golden Bird LLC&nbsp;&nbsp;
            {'<img src="'+str(LOGO_PATH.as_posix())+'" style="height:36px; vertical-align:middle;" />' if LOGO_PATH.exists() else ''}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Patient / Settings")
    lang = st.selectbox("Language / اللغة", ["English", "Arabic"], index=0 if st.session_state["lang"] == "en" else 1)
    st.session_state["lang"] = "en" if lang == "English" else "ar"
    st.text_input("Patient ID", key="patient_id", placeholder="ID (for report, will not show name)")
    dob = st.date_input("DOB", value=date(1980,1,1), key="dob")
    sex = st.selectbox("Sex / الجنس", ["Unknown", "Male", "Female", "Other"], index=0)
    st.markdown("---")
    st.subheader("Blood Tests (summary)")
    st.text_area("Enter labs (one per line) e.g. B12: 250 pg/mL  TSH: 2.1 µIU/mL", key="labs", height=120)
    st.subheader("Current medications / Past medical history")
    st.text_area("List current meds (one per line)", key="meds", height=100)
    st.markdown("---")
    st.subheader("Upload EDF files")
    uploaded = st.file_uploader("Drag & drop EDF files here (.edf)", type=["edf"], accept_multiple_files=True)
    if uploaded:
        st.session_state["uploaded_files"] = uploaded
    st.markdown("")
    if st.button("Process EDF(s)"):
        # Process button triggers processing loop below (in main area)
        st.session_state["process_now"] = True

# ---------------------------
# Main column: console + questionnaires
# ---------------------------
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("1) Questionnaires")
    st.markdown("PHQ-9 (Depression screening). Options 0-3.")
    # PHQ-9 items with modified wording for items 3/5/8 per user request
    phq_texts = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Sleep changes: (choose) Insomnia / Hypersomnia",
        "Feeling tired or having little energy",
        "Changes in appetite: Increased or decreased",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating",
        "Moving or speaking slowly OR being fidgety/restless",
        "Thoughts that you would be better off dead"
    ]
    phq_vals = []
    cols = st.columns(3)
    for i, t in enumerate(phq_texts):
        with cols[i % 3]:
            val = st.radio(f"Q{i+1}: {t}", [0,1,2,3], index=0, key=f"phq_{i+1}")
            phq_vals.append(val)
    st.markdown("PHQ-9 total: compute below in processing results.")

with col2:
    st.subheader("2) AD8 (Cognitive screening)")
    # AD8 items (A1..A8) binary 0/1
    ad8_texts = [
        "Problems with judgment (e.g. bad decision making)",
        "Less interest in hobbies/activities",
        "Repeats questions, stories, or statements",
        "Trouble learning to use tools/ appliances",
        "Forgets correct month or year",
        "Difficulty handling complicated financial affairs",
        "Difficulty remembering appointments",
        "Daily problems with thinking and memory"
    ]
    ad8_vals = []
    for i, t in enumerate(ad8_texts):
        val = st.radio(f"A{i+1}: {t}", [0,1], index=0, key=f"ad8_{i+1}")
        ad8_vals.append(val)

st.markdown("---")
st.write("Console / Visualization")
st.info("No processed results yet. Upload EDF and press 'Process EDF(s)' in the left panel.", icon="ℹ️")

# ---------------------------
# Processing: triggered when user requests or when files uploaded
# ---------------------------
def process_uploaded_files():
    st.session_state["results"] = []
    uploads = st.session_state.get("uploaded_files") or []
    if not uploads:
        st.warning("Please upload EDF file(s) first.")
        return

    progress = st.progress(0)
    total = len(uploads)
    for i, up in enumerate(uploads):
        try:
            raw, msg = read_edf_bytes(up)
            if raw is None:
                st.error(f"File {up.name}: cannot read EDF. {msg}")
                continue
            # compute band powers
            dfbands = compute_band_powers(raw)
            theta_alpha = compute_theta_alpha_ratio(dfbands)
            # quick alpha asymmetry example using channels F3,F4 if present
            alpha_asym = None
            try:
                if not dfbands.empty and "Alpha_rel" in dfbands.columns:
                    # attempt F3, F4
                    a_f3 = dfbands.loc[dfbands["channel"]=="F3", "Alpha_rel"].mean() if "F3" in dfbands["channel"].values else np.nan
                    a_f4 = dfbands.loc[dfbands["channel"]=="F4", "Alpha_rel"].mean() if "F4" in dfbands["channel"].values else np.nan
                    if not np.isnan(a_f3) and not np.isnan(a_f4):
                        alpha_asym = float(a_f3 - a_f4)
            except Exception:
                alpha_asym = None

            # generate simple visuals
            ch_names = dfbands["channel"].tolist() if not dfbands.empty else []
            # bar chart comparing Theta/Alpha vs Norm
            bar_img = bar_comparison_chart(theta_alpha or 0.0, alpha_asym or 0.0)
            # topomap images per band (approx)
            topo_imgs = {}
            for band in ["Delta","Theta","Alpha","Beta"]:
                if not dfbands.empty and f"{band}_rel" in dfbands.columns:
                    vals = dfbands[f"{band}_rel"].values
                    topo_imgs[band] = topomap_image_for_band(vals, ch_names, band_name=band)

            # connectivity: basic coherence (placeholder)
            conn_mat = None
            conn_narr = ""
            try:
                if HAS_MNE:
                    # compute connectivity in alpha band (simplified)
                    # using mne.connectivity.spectral_connectivity requires complex settings; here we add placeholder
                    conn_mat = np.zeros((len(ch_names), len(ch_names))) if ch_names else None
                    conn_narr = "Mean connectivity (alpha) computed (placeholder)"
                else:
                    conn_mat = None
                    conn_narr = "Connectivity not available (mne missing)"
            except Exception as e:
                conn_mat = None
                conn_narr = f"Connectivity error: {e}"

            # save aggregated features and narrative
            agg = {
                "filename": up.name,
                "theta_alpha_ratio": theta_alpha,
                "alpha_asym_F3_F4": alpha_asym,
                "dfbands": dfbands.to_dict(orient="list") if not dfbands.empty else {},
                "topo_images": topo_imgs,
                "connectivity_narrative": conn_narr,
                "connectivity_matrix": conn_mat,
                "bar_img": bar_img
            }

            st.session_state["results"].append(agg)
            progress.progress(int(((i+1)/total)*100))
        except Exception as e:
            st.error(f"Error processing {up.name}: {e}")
            print(traceback.format_exc())
    st.success("Processing complete.")

# trigger processing if requested
if st.session_state.get("process_now", False):
    process_uploaded_files()
    st.session_state["process_now"] = False

# If there are results, show summary (quick)
if st.session_state["results"]:
    st.success(f"Processed {len(st.session_state['results'])} file(s).")
    # show first file summaries
    res0 = st.session_state["results"][0]
    st.markdown("### Quick metrics (first file)")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Theta/Alpha (mean)", fmt(res0.get("theta_alpha_ratio", 0.0)))
    with cols[1]:
        st.metric("Alpha Asym (F3-F4)", fmt(res0.get("alpha_asym_F3_F4", 0.0)))
    with cols[2]:
        st.write(res0.get("connectivity_narrative", ""))

    if res0.get("bar_img"):
        st.image(res0["bar_img"], use_column_width="always")
    # show topomaps
    tcols = st.columns(len(res0.get("topo_images", {})) or 1)
    for j, (band, img) in enumerate(res0.get("topo_images", {}).items()):
        if img:
            with tcols[j % len(tcols)]:
                st.image(img, caption=f"{band} topography", width=240)

# Save UI state (download CSV)
if st.session_state["results"]:
    try:
        df_export = pd.DataFrame([{
            "filename": r["filename"],
            "theta_alpha_ratio": r.get("theta_alpha_ratio"),
            "alpha_asym_F3_F4": r.get("alpha_asym_F3_F4")
        } for r in st.session_state["results"]])
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

# End of part 1 (UI + lightweight processing)
# Next: part 2 will implement:
#  - full XAI (SHAP) loading and visualization
#  - detailed connectivity (coherence/wPLI) if mne is available
#  - microstate / focal delta detection & tumor index calculation
#  - ML model scoring (depression + alzheimers) using pre-trained models (placeholders)
#  - PDF generator using reportlab with Amiri font and bilingual text
# ---------------------------
# Part 2/3: Advanced EEG analysis: Connectivity, Microstates, FDI (tumor), ML scoring, SHAP visuals
# ---------------------------

# Optional imports for advanced processing
try:
    from scipy.signal import coherence
except Exception:
    coherence = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

# Helper: ensure result dict fields exist and are consistent
def ensure_result_fields(res: dict):
    res.setdefault("agg", {})
    res.setdefault("topo_images", {})
    res.setdefault("bar_img", None)
    res.setdefault("conn_img", None)
    res.setdefault("conn_narr", None)
    res.setdefault("focal", {})
    res.setdefault("shap_img", None)
    res.setdefault("shap_table", {})
    res.setdefault("ml_scores", {})  # e.g., {"depression":0.12,"alzheimers":0.18}
    return res

# ---------------------------
# Connectivity (PLI / wPLI / Coherence fallback)
# ---------------------------
def compute_functional_connectivity(data: np.ndarray, sf: float, ch_names: List[str], band: Tuple[float,float]=(8.0,13.0)):
    """
    Attempt PLI/wPLI via mne.connectivity.spectral_connectivity (preferred).
    Fallback: compute magnitude-squared coherence per pair (scipy.signal.coherence).
    Returns: conn_mat (nchan x nchan float), narration (str), conn_img_bytes (png), mean_conn (float)
    """
    nchan = data.shape[0] if data is not None else 0
    if nchan == 0:
        return None, "(no data)", None, 0.0
    conn_mat = np.zeros((nchan, nchan))
    narration = ""
    mean_conn = 0.0

    # Try MNE PLI/wPLI
    if HAS_MNE:
        try:
            # create RawArray
            info = mne.create_info(ch_names, sf, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            try:
                from mne.connectivity import spectral_connectivity
                # compute wPLI (more robust) then try pli if not available
                for method in ("wpli", "pli"):
                    try:
                        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                            [raw], method=method, mode='multitaper',
                            sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False
                        )
                        con = np.squeeze(con)
                        if con.ndim == 2 and con.shape[0] == nchan:
                            conn_mat = con
                            narration = f"{method.upper()} {band[0]}-{band[1]} Hz (MNE)"
                            break
                    except Exception as e:
                        # try next method
                        continue
            except Exception as e:
                narration = "(mne.connectivity spectral_connectivity failed)"
        except Exception as e:
            narration = "(mne rawarray/conversion failed)"

    # Fallback: coherence
    if conn_mat.sum() == 0:
        if coherence is None:
            narration = narration or "(connectivity unavailable: no mne and no scipy.coherence)"
        else:
            try:
                # compute pairwise coherence averaged in band
                for i in range(nchan):
                    for j in range(i, nchan):
                        try:
                            f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                            mask = (f >= band[0]) & (f <= band[1])
                            val = float(np.nanmean(Cxy[mask])) if mask.sum() else 0.0
                        except Exception:
                            val = 0.0
                        conn_mat[i, j] = val
                        conn_mat[j, i] = val
                narration = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
            except Exception as e:
                narration = "(coherence computation failed)"

    try:
        mean_conn = float(np.nanmean(conn_mat)) if conn_mat.size else 0.0
    except Exception:
        mean_conn = 0.0

    # Create image visual of connectivity matrix
    conn_img = None
    try:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        im = ax.imshow(conn_mat, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title("Functional Connectivity")
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, fontsize=6, rotation=90)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        conn_img = fig_to_png_bytes(fig)
    except Exception:
        conn_img = None

    return conn_mat, narration, conn_img, mean_conn

# ---------------------------
# Microstate analysis (simple) - kmeans on GFP peaks (approx.)
# ---------------------------
def simple_microstate_analysis(raw, n_states=4, sfreq=256):
    """
    Very simplified microstate segmentation:
    - compute EEG global field power (GFP), pick peaks, take topographies at peaks
    - run KMeans on those topographies to get cluster centers (microstate maps)
    - compute percent coverage and mean duration (approx)
    Returns dict with 'maps' (list of arrays), 'coverage' (per-state), 'n_states'
    """
    out = {"maps": [], "coverage": {}, "n_states": n_states}
    try:
        if raw is None or not HAS_MNE:
            return out
        data = raw.get_data()  # shape (nchan, ntime)
        sf = int(raw.info.get("sfreq", sfreq))
        # compute GFP (std across channels) and peaks
        gfp = np.std(data, axis=0)
        # find peaks: simple thresholding of gfp
        thr = np.percentile(gfp, 75)
        peaks_idx = np.where(gfp >= thr)[0]
        if peaks_idx.size < n_states:
            # not enough peaks; fallback: sample equidistant maps
            sample_idx = np.linspace(0, data.shape[1]-1, min(200, data.shape[1])).astype(int)
            maps = data[:, sample_idx].T  # (n_samples, nchan)
        else:
            maps = data[:, peaks_idx].T
        # optionally reduce dimensionality
        if SKL_AVAILABLE:
            pca = PCA(n_components=min(30, maps.shape[1]))
            maps_red = pca.fit_transform(maps)
            kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
            labels = kmeans.fit_predict(maps_red)
            centers_red = kmeans.cluster_centers_
            # project centers back approximately
            centers = pca.inverse_transform(centers_red)
            # store normalized maps
            for c in centers:
                norm = c / (np.linalg.norm(c) + 1e-12)
                out["maps"].append(norm)
            # coverage
            cov = {}
            for s in range(n_states):
                cov[s] = float((labels == s).sum()) / (labels.size if labels.size else 1)
            out["coverage"] = cov
        else:
            # fallback: no kmeans – compute simple mean map
            mean_map = np.mean(maps, axis=0)
            out["maps"] = [mean_map / (np.linalg.norm(mean_map)+1e-12)]
            out["coverage"] = {0:1.0}
    except Exception as e:
        print("microstate err:", e)
    return out

# ---------------------------
# Focal Delta Index (FDI) improved
# ---------------------------
def compute_focal_delta_index(dfbands: pd.DataFrame, ch_names: List[str]=None) -> Dict[str, Any]:
    """
    Compute:
    - per-channel FDI = delta_abs / global_mean_delta
    - max FDI and index
    - pairwise asymmetry ratios for key pairs (T7/T8 etc.)
    - return narrative and alerts list
    """
    out = {"fdi": {}, "alerts": [], "max_idx": None, "max_val": None, "asymmetry": {}}
    try:
        if dfbands is None or dfbands.empty:
            return out
        delta_abs_col = "Delta_abs" if "Delta_abs" in dfbands.columns else "Delta_abs"
        if delta_abs_col not in dfbands.columns:
            # if not found, try lowercase
            for c in dfbands.columns:
                if c.lower().startswith("delta") and "abs" in c.lower():
                    delta_abs_col = c; break
        delta = np.array(dfbands[delta_abs_col].values, dtype=float)
        gm = float(np.nanmean(delta)) if delta.size else 1e-9
        for i, v in enumerate(delta):
            fdi = float(v / (gm if gm>0 else 1e-9))
            out["fdi"][i] = fdi
            if fdi > 2.0:
                ch = ch_names[i] if ch_names and i < len(ch_names) else f"Ch{i}"
                out["alerts"].append({"type":"FDI","channel":ch,"value":float(fdi)})
        # pairs for asymmetry
        pairs = [("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        name_map = {}
        if ch_names:
            for i,n in enumerate(ch_names):
                name_map[n.upper()] = i
        for L,R in pairs:
            if L in name_map and R in name_map:
                li = name_map[L]; ri = name_map[R]
                dl = float(delta[li]) if li < len(delta) else 0.0
                dr = float(delta[ri]) if ri < len(delta) else 0.0
                ratio = float(dr / (dl + 1e-9)) if dl>0 else float("inf") if dr>0 else 1.0
                out["asymmetry"][f"{L}/{R}"] = ratio
                if (isinstance(ratio,float) and (ratio>3.0 or ratio<0.33)) or (ratio==float("inf")):
                    out["alerts"].append({"type":"asymmetry","pair":f"{L}/{R}","ratio":ratio})
        max_idx = int(np.argmax(list(out["fdi"].values()))) if out["fdi"] else None
        max_val = out["fdi"].get(max_idx,None) if max_idx is not None else None
        out["max_idx"] = max_idx; out["max_val"] = max_val
    except Exception as e:
        print("compute_fdi error:", e)
    return out

# ---------------------------
# ML scoring: load models if present, else heuristic
# ---------------------------
def load_ml_model(kind: str):
    """
    Attempt to load a pretrained model from MODEL_PATH (pickle). Names: depression.pkl, alzheimers.pkl
    Returns model object or None.
    """
    try:
        import joblib
    except Exception:
        joblib = None
    fname = MODEL_PATH / f"{kind}.pkl"
    if joblib and fname.exists():
        try:
            model = joblib.load(str(fname))
            return model
        except Exception as e:
            print(f"load model {kind} failed: {e}")
            return None
    return None

# Simple scoring function using aggregate features; replace by real model if available
def score_ml_models(agg: Dict[str,float], phq_total: int, ad8_total: int) -> Dict[str,float]:
    """
    Return scores in [0,1] for depression and alzheimers.
    If model files present, call them. Else, use heuristic.
    """
    scores = {"depression": None, "alzheimers": None}
    # Try loading
    dep_model = load_ml_model("depression")
    alz_model = load_ml_model("alzheimers")
    feat_vector = [
        agg.get("theta_alpha_ratio",0.0),
        agg.get("theta_beta_ratio",0.0),
        agg.get("alpha_rel_mean",0.0),
        agg.get("gamma_rel_mean",0.0) if "gamma_rel_mean" in agg else 0.0,
        phq_total/27.0 if phq_total else 0.0,
        ad8_total/8.0 if ad8_total else 0.0
    ]
    X = np.array(feat_vector).reshape(1,-1)
    try:
        if dep_model is not None:
            p = dep_model.predict_proba(X)[:,1].item()
            scores["depression"] = float(p)
        else:
            # heuristic: combine phq and some EEG
            ta = agg.get("theta_alpha_ratio",0.0)
            phq_norm = (phq_total/27.0) if phq_total else 0.0
            scores["depression"] = float(min(1.0, 0.5*phq_norm + 0.3*(ta/1.6)))
    except Exception as e:
        scores["depression"] = 0.0
    try:
        if alz_model is not None:
            p = alz_model.predict_proba(X)[:,1].item()
            scores["alzheimers"] = float(p)
        else:
            # heuristic: theta/alpha and connectivity
            ta = agg.get("theta_alpha_ratio",0.0)
            conn = agg.get("mean_connectivity", 0.0)
            conn_norm = 1.0 - conn if conn is not None else 1.0
            scores["alzheimers"] = float(min(1.0, 0.6*(ta/1.6) + 0.3*conn_norm + 0.1*(ad8_total/8.0 if ad8_total else 0.0)))
    except Exception as e:
        scores["alzheimers"] = 0.0
    return scores

# ---------------------------
# SHAP visualization helpers
# ---------------------------
def generate_shap_bar_from_summary(shap_summary: dict, model_key: str="alzheimers_global", top_n=10) -> Optional[bytes]:
    """
    shap_summary is expected to be a dict of dicts: {model_key: {feature:importance,...}, ...}
    Return png bytes of horizontal bar of abs importance.
    """
    if not shap_summary:
        return None
    try:
        feats = shap_summary.get(model_key, {})
        if not feats:
            # try to pick any key
            anyk = next(iter(shap_summary.keys()))
            feats = shap_summary.get(anyk, {})
        s = pd.Series(feats).abs().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, 2.0))
        s.sort_values().plot.barh(ax=ax, color=PRIMARY_BLUE)
        ax.set_xlabel("SHAP (abs)")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("shap bar err:", e)
        return None

# ---------------------------
# Integrate advanced analyses into processing pipeline
# Replace earlier placeholder connectivity/microstate in process_uploaded_files()
# ---------------------------
def enrich_results_with_advanced_analyses():
    """
    For all items in st.session_state['results'], compute:
    - agg features (theta_alpha, etc.)
    - focal delta & tumor alerts
    - connectivity mat + image + mean
    - microstate analysis (if mne)
    - ml scoring
    - shap visuals if shap_summary.json found
    Save results back into st.session_state['results'][i]
    """
    shap_summary = load_shap_json(SHAP_JSON) if SHAP_JSON.exists() else None
    results = st.session_state.get("results", [])
    if not results:
        return
    for idx, res in enumerate(results):
        try:
            res = ensure_result_fields(res)
            # If dfbands stored as dict of lists, convert to DataFrame
            dfbands = None
            if isinstance(res.get("dfbands", None), dict):
                try:
                    dfbands = pd.DataFrame(res["dfbands"])
                except Exception:
                    dfbands = pd.DataFrame()
            elif isinstance(res.get("dfbands", None), pd.DataFrame):
                dfbands = res["dfbands"]
            else:
                dfbands = pd.DataFrame(res.get("dfbands", {}))
            ch_names = list(dfbands["channel"].values) if not dfbands.empty and "channel" in dfbands.columns else (res.get("ch_names") or res.get("ch_names", []))

            # aggregate features
            agg = {}
            # mean relative per band
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                c = f"{band}_rel"
                agg[f"{band.lower()}_rel_mean"] = float(dfbands[c].mean()) if (not dfbands.empty and c in dfbands.columns) else 0.0
            agg["theta_alpha_ratio"] = float((agg.get("theta_rel_mean",0.0) / (agg.get("alpha_rel_mean",1e-12))) if agg.get("alpha_rel_mean",0.0)>0 else 0.0)
            agg["theta_beta_ratio"] = float((agg.get("theta_rel_mean",0.0) / (agg.get("beta_rel_mean",1e-12))) if agg.get("beta_rel_mean",0.0)>0 else 0.0)
            agg["alpha_asym_f3_f4"] = 0.0
            if ch_names:
                try:
                    names = [n.upper() for n in ch_names]
                    if "F3" in names and "F4" in names:
                        i3 = names.index("F3"); i4 = names.index("F4")
                        a3 = float(dfbands.iloc[i3].get("Alpha_rel", dfbands.iloc[i3].get("alpha_rel",0.0)))
                        a4 = float(dfbands.iloc[i4].get("Alpha_rel", dfbands.iloc[i4].get("alpha_rel",0.0)))
                        agg["alpha_asym_f3_f4"] = float(a3 - a4)
                except Exception:
                    agg["alpha_asym_f3_f4"] = 0.0

            # compute FDI (tumor)
            focal = compute_focal_delta_index(dfbands, ch_names=ch_names)
            res["focal"] = focal

            # connectivity (use raw if available in res)
            raw_obj = res.get("raw_obj", None)  # if earlier we stored mne Raw
            # if not, try to reconstruct data array from dfbands? we need raw to compute connectivity; fallback: None
            conn_mat, conn_narr, conn_img, mean_conn = None, "(not computed)", None, 0.0
            if res.get("connectivity_matrix") is not None and isinstance(res["connectivity_matrix"], np.ndarray):
                # use existing
                conn_mat = res["connectivity_matrix"]
                conn_narr = res.get("connectivity_narr", "(existing)")
                mean_conn = float(np.nanmean(conn_mat)) if conn_mat.size else 0.0
                # create image if missing
                if res.get("conn_img") is None:
                    try:
                        fig, ax = plt.subplots(figsize=(4.2,3.2))
                        im = ax.imshow(conn_mat, cmap="viridis", interpolation="nearest", aspect="auto")
                        ax.set_title("Connectivity")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        res["conn_img"] = fig_to_png_bytes(fig)
                    except Exception:
                        res["conn_img"] = None
            else:
                # If raw object stored, compute connectivity
                if res.get("raw_obj") is not None and HAS_MNE:
                    try:
                        raw = res["raw_obj"]
                        data = raw.get_data(picks=mne.pick_types(raw.info, eeg=True))
                        conn_mat, conn_narr, conn_img, mean_conn = compute_functional_connectivity(data, float(raw.info.get("sfreq",256.0)), ch_names=list(raw.ch_names), band=(8.0,13.0))
                        res["conn_img"] = conn_img
                    except Exception as e:
                        # fallback: if dfbands contains only relative powers cannot compute full connectivity
                        conn_mat, conn_narr, conn_img, mean_conn = (None, "(connectivity not computed)", None, 0.0)
                else:
                    # no raw available; can't compute connectivity reliably
                    conn_mat, conn_narr, conn_img, mean_conn = (None, "(no raw data for connectivity)", None, 0.0)
            agg["mean_connectivity"] = mean_conn
            res["conn_narr"] = conn_narr

            # microstate analysis
            micro = None
            if res.get("raw_obj") is not None and HAS_MNE:
                try:
                    micro = simple_microstate_analysis(res["raw_obj"], n_states=4, sfreq=int(res["raw_obj"].info.get("sfreq",256)))
                except Exception:
                    micro = None
            res["microstate"] = micro

            # ML scoring
            # Retrieve questionnaire totals from UI keys
            phq_total = 0
            for i in range(1,10):
                k = f"phq_{i}"
                try:
                    phq_total += int(st.session_state.get(k,0))
                except Exception:
                    pass
            ad8_total = 0
            for i in range(1,9):
                k = f"ad8_{i}"
                try:
                    ad8_total += int(st.session_state.get(k,0))
                except Exception:
                    pass
            scores = score_ml_models(agg, phq_total, ad8_total)
            res["ml_scores"] = scores

            # SHAP visuals: if shap_summary.json present, generate shap_img
            if shap_summary:
                try:
                    # choose key heuristically
                    model_key = "depression_global" if agg.get("theta_alpha_ratio",0.0) <= 1.3 else "alzheimers_global"
                    shap_img = generate_shap_bar_from_summary(shap_summary, model_key=model_key, top_n=10)
                    res["shap_img"] = shap_img
                    res["shap_table"] = shap_summary.get(model_key, {})
                except Exception:
                    res["shap_img"] = None
                    res["shap_table"] = {}
            else:
                res["shap_img"] = None
                res["shap_table"] = {}

            # normative bar (Theta/Alpha + Alpha asym)
            try:
                bar_img = bar_comparison_chart(agg.get("theta_alpha_ratio",0.0), agg.get("alpha_asym_f3_f4",0.0))
                res["bar_img"] = bar_img
            except Exception:
                res["bar_img"] = res.get("bar_img", None)

            # store aggregated features
            res["agg"] = agg

            # Save back to session
            st.session_state["results"][idx] = res
        except Exception as e:
            print("enrich_results error:", e)
            st.session_state["results"][idx]["error"] = str(e)

# If processing step completed earlier, run enrichment
if st.session_state.get("results"):
    enrich_results_with_advanced_analyses()

# After enrichment, show updated visual cues if any
if st.session_state.get("results"):
    r0 = st.session_state["results"][0]
    st.markdown("## Advanced analysis results (first file)")
    st.write("ML scores:", r0.get("ml_scores", {}))
    st.write("Focal alerts:", r0.get("focal", {}).get("alerts", []))
    if r0.get("conn_img"):
        st.image(r0["conn_img"], caption="Connectivity", width=640)
    if r0.get("shap_img"):
        st.image(r0["shap_img"], caption="SHAP contributors", width=640)
    if r0.get("bar_img"):
        st.image(r0["bar_img"], caption="Normative bar", width=640)

# End of Part 2/3
# ---------------------------
# Part 3/3: PDF generator (robust bilingual), final UI v2 polish & export
# Place this AFTER Part 2 in app.py
# ---------------------------

# Imports used here (some may be already imported above — safe to re-import)
import base64
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4 as RL_A4
from reportlab.lib.units import inch as RL_INCH
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

# --- Helpers used by PDF and UI ---
def safe_bytes_img(img_bytes):
    """Return io.BytesIO wrapped object if bytes present, else None"""
    if not img_bytes:
        return None
    try:
        return io.BytesIO(img_bytes)
    except Exception:
        return None

def attach_logo_rl(logo_path: Path, width=72, height=72):
    try:
        if logo_path.exists():
            pil = PILImage.open(str(logo_path))
            buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
            return RLImage(buf, width=width, height=height)
    except Exception:
        return None
    return None

# Robust PDF generator (final)
def generate_pdf_report_final(result: dict,
                              patient_info: dict,
                              phq_total: int,
                              ad8_total: int,
                              lang: str = "en",
                              amiri_path: Optional[Path] = None,
                              logo_path: Optional[Path] = None) -> bytes:
    """
    Build a professional bilingual PDF for a single result entry.
    result: one element from st.session_state['results'] after enrichment
    patient_info: dict with id, dob, sex, meds, labs
    phq_total, ad8_total: questionnaire totals
    lang: 'en' or 'ar' (controls main narrative language; both languages appear)
    returns: pdf bytes
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is required for PDF export.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=RL_A4,
                            topMargin=28, bottomMargin=28, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    # register Amiri if available
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception:
            base_font = "Helvetica"

    styles.add(ParagraphStyle(name="Title", fontName=base_font, fontSize=16, textColor=rl_colors.HexColor(PDF_BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name="Header", fontName=base_font, fontSize=11, textColor=rl_colors.HexColor(PDF_BLUE), spaceAfter=4))
    styles.add(ParagraphStyle(name="Normal", fontName=base_font, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="Small", fontName=base_font, fontSize=9, leading=11, textColor=rl_colors.grey))

    story = []

    # Header row: title + logo
    title = Paragraph("<b>NeuroEarly Pro — Clinical QEEG Report</b>", styles["Title"])
    logo_rl = attach_logo_rl(Path(logo_path) if logo_path else LOGO_PATH)
    try:
        if logo_rl:
            header = Table([[title, logo_rl]], colWidths=[4.8*RL_INCH, 1.4*RL_INCH])
            header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(header)
        else:
            story.append(title)
    except Exception:
        story.append(title)
    story.append(Spacer(1, 8))

    # Executive summary top
    ml_score = result.get("ml_scores", {}).get("alzheimers") or result.get("ml_scores", {}).get("depression") or result.get("ml_risk") or 0.0
    ml_display = f"{ml_score*100:.1f}%" if isinstance(ml_score, (int,float)) else str(ml_score)
    exec_text = f"<b>Final ML Risk Score:</b> {ml_display} &nbsp;&nbsp; <b>PHQ-9:</b> {phq_total} &nbsp;&nbsp; <b>AD8:</b> {ad8_total}"
    story.append(Paragraph(exec_text, styles["Normal"]))
    story.append(Spacer(1, 8))

    # Patient info table (ID, DOB, Sex, Meds summary, Labs summary)
    pid = patient_info.get("id", "—")
    dob = patient_info.get("dob", "—")
    sex = patient_info.get("sex", "—")
    meds = patient_info.get("meds", "").strip().splitlines()
    labs = patient_info.get("labs", "").strip().splitlines()
    meds_s = ", ".join(meds[:6]) + ("..." if len(meds) > 6 else "")
    labs_s = ", ".join(labs[:6]) + ("..." if len(labs) > 6 else "")
    ptab = [["Field", "Value"], ["Patient ID", pid], ["DOB", dob], ["Sex", sex], ["Medications", meds_s or "—"], ["Blood tests", labs_s or "—"]]
    t = Table(ptab, colWidths=[1.5*RL_INCH, 4.5*RL_INCH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#eaf6ff")), ("GRID", (0,0), (-1,-1), 0.25, rl_colors.lightgrey)]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["Header"]))
    agg = result.get("agg", {})
    metrics_rows = [["Metric", "Value", "Note"]]
    metrics_template = [
        ("theta_alpha_ratio", "Theta/Alpha Ratio", "Slowing indicator"),
        ("theta_beta_ratio", "Theta/Beta Ratio", "Stress/inattention"),
        ("alpha_asym_f3_f4", "Alpha Asymmetry (F3-F4)", "Left-right asymmetry"),
        ("gamma_rel_mean", "Gamma Relative Mean", "Cognition-related"),
        ("mean_connectivity", "Mean Connectivity (alpha)", "Functional coherence")
    ]
    for key, label, note in metrics_template:
        val = agg.get(key, result.get(key, "N/A"))
        try:
            display_val = f"{float(val):.4f}"
        except Exception:
            display_val = str(val)
        metrics_rows.append([label, display_val, note])
    mt = Table(metrics_rows, colWidths=[2.8*RL_INCH, 1.2*RL_INCH, 2.0*RL_INCH])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor("#eef7ff")), ("GRID",(0,0),(-1,-1),0.25,rl_colors.grey)]))
    story.append(mt)
    story.append(Spacer(1, 8))

    # Normative comparison (bar)
    if result.get("bar_img"):
        try:
            story.append(Paragraph("<b>Normative Comparison</b>", styles["Header"]))
            bi = safe_bytes_img(result["bar_img"])
            if bi:
                story.append(RLImage(bi, width=5.6*RL_INCH, height=1.6*RL_INCH))
                story.append(Spacer(1, 6))
        except Exception:
            pass

    # Topography maps (2 per row)
    topo = result.get("topo_images", {}) or {}
    if topo:
        story.append(Paragraph("<b>Topography Maps</b>", styles["Header"]))
        imgs = []
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            b = topo.get(band)
            if b:
                imgs.append(safe_bytes_img(b))
        if imgs:
            rows = []
            row = []
            for im in imgs:
                if im:
                    row.append(RLImage(im, width=2.6*RL_INCH, height=1.6*RL_INCH))
                else:
                    row.append("")
                if len(row) == 2:
                    rows.append(row); row = []
            if row: rows.append(row)
            for r in rows:
                tbl = Table([r], colWidths=[3*RL_INCH, 3*RL_INCH])
                tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
                story.append(tbl)
            story.append(Spacer(1,6))

    # Connectivity
    if result.get("conn_img"):
        story.append(Paragraph("<b>Functional Connectivity (Alpha)</b>", styles["Header"]))
        ci = safe_bytes_img(result["conn_img"])
        if ci:
            story.append(RLImage(ci, width=5.6*RL_INCH, height=2.4*RL_INCH))
            story.append(Spacer(1,6))

    # SHAP
    if result.get("shap_img"):
        story.append(Paragraph("<b>Explainable AI — SHAP top contributors</b>", styles["Header"]))
        si = safe_bytes_img(result["shap_img"])
        if si:
            story.append(RLImage(si, width=5.6*RL_INCH, height=1.8*RL_INCH))
            story.append(Spacer(1,6))
    elif result.get("shap_table"):
        story.append(Paragraph("<b>Explainable AI — Top contributors (table)</b>", styles["Header"]))
        stbl = [["Feature","Importance"]]
        for k,v in list(result.get("shap_table", {}).items())[:10]:
            try:
                stbl.append([k, f"{float(v):.4f}"])
            except Exception:
                stbl.append([k, str(v)])
        t3 = Table(stbl, colWidths=[3.6*RL_INCH, 2.0*RL_INCH])
        t3.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,rl_colors.grey), ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#eef7ff"))]))
        story.append(t3)
        story.append(Spacer(1,6))

    # Tumor / Focal delta narrative
    if result.get("focal"):
        story.append(Paragraph("<b>Focal Delta / Tumor indicators</b>", styles["Header"]))
        narrative = f"Max FDI: {result['focal'].get('max_val')} at idx {result['focal'].get('max_idx')}"
        story.append(Paragraph(narrative, styles["Normal"]))
        if result["focal"].get("alerts"):
            for a in result["focal"]["alerts"]:
                story.append(Paragraph(f"- {a}", styles["Normal"]))
        story.append(Spacer(1,6))

    # Microstate summary (if present)
    if result.get("microstate"):
        ms = result["microstate"]
        story.append(Paragraph("<b>Microstate summary</b>", styles["Header"]))
        story.append(Paragraph(f"Number of states: {ms.get('n_states', '—')}. Coverage: {ms.get('coverage', {})}", styles["Normal"]))
        story.append(Spacer(1,6))

    # Recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["Header"]))
    recs = result.get("recommendations") or [
        "Correlate QEEG findings with PHQ-9 and AD8 scores.",
        "Check B12, TSH and metabolic panel to exclude reversible causes.",
        "If ML Risk Score > 25% and Theta/Alpha > 1.4 => consider MRI / FDG-PET referral.",
        "Follow-up in 3-6 months for moderate risk cases."
    ]
    for r in recs:
        story.append(Paragraph(f"- {r}", styles["Normal"]))
    story.append(Spacer(1,8))

    # Footer
    story.append(Paragraph("Prepared and designed by Golden Bird LLC — Oman | 2025", styles["Small"]))
    story.append(Spacer(1,6))

    # Build PDF
    try:
        doc.build(story)
    except Exception as e:
        print("PDF build exception:", e)
    buffer.seek(0)
    data = buffer.getvalue()
    buffer.close()
    return data

# ---------------------------
# Final UI polish (right-side area): tabs and PDF generation
# ---------------------------
# update the main UI area to use tabs and nicer cards
st.markdown("## Interactive Dashboard (Results)")

if st.session_state.get("results"):
    r0 = st.session_state["results"][0]
    # construct patient_info dict
    patient_info = {
        "id": st.session_state.get("patient_id", ""),
        "dob": str(st.session_state.get("dob", "")),
        "sex": st.session_state.get("sex", ""),
        "meds": st.session_state.get("meds", ""),
        "labs": st.session_state.get("labs", "")
    }
    # compute questionnaire totals
    phq_total = sum(int(st.session_state.get(f"phq_{i}", 0)) for i in range(1,10))
    ad8_total = sum(int(st.session_state.get(f"ad8_{i}", 0)) for i in range(1,9))

    # Tabs
    tabs = st.tabs(["Overview", "Connectivity", "XAI (SHAP)", "Microstates", "Export"])
    with tabs[0]:
        st.subheader("Overview")
        st.metric("Final ML Risk Score", f"{(r0.get('ml_scores',{}).get('alzheimers') or r0.get('ml_scores',{}).get('depression') or 0.0)*100:.1f}%")
        # show small QEEG table
        qdf = pd.DataFrame([{
            "Theta/Alpha": r0.get("agg",{}).get("theta_alpha_ratio"),
            "Theta/Beta": r0.get("agg",{}).get("theta_beta_ratio"),
            "Alpha Asym F3-F4": r0.get("agg",{}).get("alpha_asym_f3_f4"),
            "Mean Connectivity": r0.get("agg",{}).get("mean_connectivity")
        }])
        st.table(qdf.T.rename(columns={0:"Value"}))

        colA, colB = st.columns([1,1])
        with colA:
            if r0.get("bar_img"):
                st.image(r0["bar_img"], caption="Normative comparison", width=520)
        with colB:
            if r0.get("topo_images"):
                # show alpha topomap larger
                alpha_img = r0["topo_images"].get("Alpha")
                if alpha_img:
                    st.image(alpha_img, caption="Alpha topography", width=320)

    with tabs[1]:
        st.subheader("Connectivity")
        st.write(r0.get("conn_narr", "Connectivity narrative not available."))
        if r0.get("conn_img"):
            st.image(r0["conn_img"], caption="Connectivity matrix", width=680)
        else:
            st.info("Connectivity image not ready.")

    with tabs[2]:
        st.subheader("Explainable AI (SHAP)")
        if r0.get("shap_img"):
            st.image(r0["shap_img"], caption="SHAP top contributors", width=680)
        elif r0.get("shap_table"):
            st.table(pd.DataFrame(list(r0.get("shap_table", {}).items()), columns=["Feature","Importance"]).set_index("Feature"))
        else:
            st.info("SHAP not available. Upload shap_summary.json or compute SHAP locally.")

    with tabs[3]:
        st.subheader("Microstates")
        if r0.get("microstate"):
            st.write(r0["microstate"])
        else:
            st.info("Microstate analysis not available.")

    with tabs[4]:
        st.subheader("Export / PDF")
        pdf_lang = st.selectbox("Select PDF language", options=["English","Arabic"], index=0)
        lang_code = "en" if pdf_lang=="English" else "ar"
        amiri = AMIRI_PATH if AMIRI_PATH.exists() else None
        logo = LOGO_PATH if LOGO_PATH.exists() else None
        if st.button("Generate & Download Clinical PDF"):
            try:
                pdf_bytes = generate_pdf_report_final(r0, patient_info, phq_total, ad8_total, lang=lang_code, amiri_path=amiri, logo_path=logo)
                if pdf_bytes:
                    st.success("PDF generated successfully.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation produced no data.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.text(safe_trace(e))
else:
    st.info("No processed results yet. Upload and Process EDF(s).")

# Small help & notes
st.markdown("---")
st.markdown(
    """
    **Notes:**  
    - Default language is English; Arabic is available for text sections and the PDF uses Amiri font if present.  
    - For best connectivity & microstate results install `mne` and `scikit-learn`.  
    - Place pre-trained models in `models/depression.pkl` and `models/alzheimers.pkl` to enable model scoring.
    """)
