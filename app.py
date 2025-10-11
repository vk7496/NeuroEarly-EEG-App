# app.py — NeuroEarly Pro (final, part 1/3)
"""
NeuroEarly Pro — Clinical Edition (final)
Features included in this file (split in 3 parts):
- Robust EDF reading (mne / pyedflib fallback)
- Preprocessing (notch + bandpass)
- PSD computation with safe guards against empty arrays / NaNs
- Aggregation of band metrics (Theta/Alpha, asymmetry etc.)
- Topography image generator (matplotlib + griddata fallback)
- Connectivity computation (mne_connectivity if installed else scipy.coherence fallback)
- Utilities for PDF formatting & Arabic shaping (later parts)
"""
import os
import io
import json
import tempfile
import traceback
import datetime
from datetime import date
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="NeuroEarly Pro – AI EEG Assistant", layout="wide")

# ---------------- Optional heavy libs (detect at runtime) ----------------
HAS_MNE = False
HAS_MNE_CONN = False
HAS_PYEDF = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_ARABIC_TOOLS = False
HAS_SHAP = False
HAS_JOBLIB = False

# try imports with safe fallbacks
try:
    import mne
    HAS_MNE = True
    # try mne connectivity (optional)
    try:
        import mne_connectivity  # noqa: F401
        HAS_MNE_CONN = True
    except Exception:
        HAS_MNE_CONN = False
except Exception:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm, patches
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        HAS_ARABIC_TOOLS = True
    except Exception:
        HAS_ARABIC_TOOLS = False
except Exception:
    HAS_REPORTLAB = False
    HAS_ARABIC_TOOLS = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

# SciPy utilities (required)
from scipy.signal import welch, butter, filtfilt, iirnotch, coherence
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None

# ---------------- Project files & assets ----------------
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"
TOPO_PLACEHOLDER = ASSETS_DIR / "topo_placeholder.svg"
CONN_PLACEHOLDER = ASSETS_DIR / "conn_placeholder.svg"
SHAP_JSON = Path("shap_summary.json")
MODEL_DEP = Path("model_depression.pkl")
MODEL_AD = Path("model_alzheimer.pkl")
AMIRI_TTF = Path("Amiri-Regular.ttf")

# Frequency bands (add Gamma)
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}
DEFAULT_SF = 256.0

# Normative cut ranges (example - tune with normative DB later)
NORM_RANGES = {
    "theta_alpha_ratio": {"healthy_low": 0.0, "healthy_high": 1.1, "at_risk_low": 1.1, "at_risk_high": 1.4},
    "alpha_asym_F3_F4": {"healthy_low": -0.05, "healthy_high": 0.05, "at_risk_low": -0.2, "at_risk_high": -0.05}
}

# ---------------- Utility functions ----------------
def now_ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e: Exception) -> None:
    tb = traceback.format_exc()
    st.error("Internal error — see logs")
    st.code(tb)
    print(tb)

def reshape_ar(text: str) -> str:
    """Return Arabic-shaped text if tools available, else raw."""
    if not text:
        return ""
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def format_for_pdf_value(v: Any) -> str:
    """Format numeric/dict/list values safely for PDF printing."""
    try:
        if v is None:
            return "N/A"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            # clip NaN/inf
            if np.isnan(v) or np.isinf(v):
                return "N/A"
            return f"{v:.4f}"
        if isinstance(v, dict):
            parts = []
            for k, val in v.items():
                parts.append(f"{k}={format_for_pdf_value(val)}")
            return "; ".join(parts)
        if isinstance(v, (list, tuple)):
            return ", ".join([format_for_pdf_value(x) for x in v])
        return str(v)
    except Exception:
        return str(v)

# ---------------- EDF IO ----------------
def save_tmp_upload(uploaded_file: "streamlit.uploaded_file_manager.UploadedFile") -> str:
    """Save uploaded file to a temporary path and return filename."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path: str) -> Dict[str, Any]:
    """
    Read EDF using mne (preferred) or pyedflib fallback.
    Returns dict with: backend, raw (if mne), data (np.ndarray channels x samples),
    ch_names, sfreq.
    """
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data()
            chs = raw.ch_names
            sf = raw.info.get("sfreq", None)
            return {"backend": "mne", "raw": raw, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            # fallback to pyedflib if available
            print("mne read EDF failed, trying pyedflib:", e)
    if HAS_PYEDF:
        try:
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            chs = f.getSignalLabels()
            try:
                sf = f.getSampleFrequency(0)
            except Exception:
                sf = None
            sigs = []
            for i in range(n):
                try:
                    s = f.readSignal(i).astype(np.float64)
                    sigs.append(s)
                except Exception:
                    sigs.append(np.zeros(1))
            f._close()
            data = np.vstack(sigs)
            return {"backend": "pyedflib", "raw": None, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            raise IOError(f"pyedflib failed to read EDF: {e}")
    raise ImportError("No EDF backend available. Install 'mne' or 'pyedflib'.")

# ---------------- Filtering ----------------
def notch_filter(sig: np.ndarray, sf: Optional[float], freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Apply notch filter (50Hz default). If sf invalid -> return input."""
    if sf is None or sf <= 0:
        return sig
    try:
        b, a = iirnotch(freq, Q, sf)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig: np.ndarray, sf: Optional[float], low: float = 0.5, high: float = 45.0, order: int = 4) -> np.ndarray:
    """Apply bandpass; safe for short signals."""
    if sf is None or sf <= 0:
        return sig
    try:
        ny = 0.5 * sf
        low_n = max(low / ny, 1e-6)
        high_n = min(high / ny, 0.999)
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(raw_data: np.ndarray, sf: Optional[float], do_notch: bool = True) -> np.ndarray:
    """Apply notch + bandpass channel-wise. Returns cleaned array same shape."""
    cleaned = np.zeros_like(raw_data, dtype=np.float64)
    for i in range(raw_data.shape[0]):
        s = raw_data[i].astype(np.float64)
        if do_notch:
            s = notch_filter(s, sf)
        s = bandpass_filter(s, sf)
        cleaned[i] = s
    return cleaned

# ---------------- PSD (safe) ----------------
def compute_psd_bands(data: np.ndarray, sf: Optional[float], nperseg: int = 1024) -> pd.DataFrame:
    """
    Compute PSD band absolute and relative power per channel.
    Returns DataFrame with rows per channel and columns like 'Alpha_abs', 'Alpha_rel', etc.
    This function is robust to empty signals.
    """
    rows = []
    nch = int(data.shape[0]) if data is not None and len(data.shape) >= 1 else 0
    for ch in range(nch):
        sig = data[ch]
        if sig is None or len(sig) < 4:
            # too short -> zeros
            row = {"channel_idx": ch}
            for band in BANDS.keys():
                row[f"{band}_abs"] = 0.0
                row[f"{band}_rel"] = 0.0
            rows.append(row)
            continue
        try:
            freqs, pxx = welch(sig, fs=sf if sf is not None else DEFAULT_SF, nperseg=min(nperseg, max(256, len(sig))))
            # ensure arrays
            if freqs is None or pxx is None or len(freqs) == 0 or len(pxx) == 0:
                total = 0.0
            else:
                # replace NaNs/Infs
                pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)
                total = float(np.trapz(pxx, freqs)) if np.isfinite(np.sum(pxx)) and freqs.size > 0 else 0.0
        except Exception as e:
            print("PSD welch error:", e)
            freqs = np.array([])
            pxx = np.array([])
            total = 0.0
        row = {"channel_idx": ch}
        for band, (lo, hi) in BANDS.items():
            try:
                if freqs.size == 0:
                    abs_p = 0.0
                else:
                    mask = (freqs >= lo) & (freqs <= hi)
                    if mask.sum() == 0:
                        abs_p = 0.0
                    else:
                        seg_freqs = freqs[mask]
                        seg_pxx = pxx[mask]
                        seg_pxx = np.nan_to_num(seg_pxx, nan=0.0, posinf=0.0, neginf=0.0)
                        abs_p = float(np.trapz(seg_pxx, seg_freqs)) if seg_freqs.size > 0 else 0.0
                rel = float(abs_p / total) if total > 0 else 0.0
            except Exception as e:
                print("Band power compute error:", e)
                abs_p = 0.0
                rel = 0.0
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        rows.append(row)
    if len(rows) == 0:
        # return empty DF with expected columns
        cols = ["channel_idx"] + [f"{b}_abs" for b in BANDS.keys()] + [f"{b}_rel" for b in BANDS.keys()]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)

# ---------------- Aggregation ----------------
def aggregate_bands(df_bands: pd.DataFrame, ch_names: Optional[list] = None) -> Dict[str, float]:
    """
    Aggregate PSD per band across channels and compute ratios & asymmetry.
    Returns dict with normalized/mean features.
    """
    out: Dict[str, float] = {}
    if df_bands is None or df_bands.empty:
        # zero defaults
        for band in BANDS.keys():
            out[f"{band.lower()}_abs_mean"] = 0.0
            out[f"{band.lower()}_rel_mean"] = 0.0
        out["theta_alpha_ratio"] = 0.0
        out["theta_beta_ratio"] = 0.0
        out["beta_alpha_ratio"] = 0.0
        out["alpha_asym_F3_F4"] = 0.0
        return out
    # safe mean
    for band in BANDS.keys():
        try:
            out[f"{band.lower()}_abs_mean"] = float(np.nanmean(df_bands[f"{band}_abs"].values))
        except Exception:
            out[f"{band.lower()}_abs_mean"] = 0.0
        try:
            out[f"{band.lower()}_rel_mean"] = float(np.nanmean(df_bands[f"{band}_rel"].values))
        except Exception:
            out[f"{band.lower()}_rel_mean"] = 0.0
    # ratios with safe division
    alpha_rel = out.get("alpha_rel_mean", 1e-9)
    beta_rel = out.get("beta_rel_mean", 1e-9)
    theta_rel = out.get("theta_rel_mean", 0.0)
    out["theta_alpha_ratio"] = float(theta_rel / alpha_rel) if alpha_rel > 0 else 0.0
    out["theta_beta_ratio"] = float(theta_rel / beta_rel) if beta_rel > 0 else 0.0
    out["beta_alpha_ratio"] = float(beta_rel / alpha_rel) if alpha_rel > 0 else 0.0
    # alpha asymmetry F3-F4 (best-effort using channel names)
    out["alpha_asym_F3_F4"] = 0.0
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def find_index(token):
                for i, nm in enumerate(names):
                    if token in nm:
                        return i
                return None
            i3 = find_index("F3")
            i4 = find_index("F4")
            if i3 is not None and i4 is not None:
                # find corresponding rel alpha in df_bands by channel_idx
                a3 = df_bands.loc[df_bands['channel_idx'] == i3, 'Alpha_rel']
                a4 = df_bands.loc[df_bands['channel_idx'] == i4, 'Alpha_rel']
                if not a3.empty and not a4.empty:
                    try:
                        out["alpha_asym_F3_F4"] = float(a3.values[0] - a4.values[0])
                    except Exception:
                        out["alpha_asym_F3_F4"] = 0.0
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# ---------------- Topomap image generator ----------------
def generate_topomap_image(band_vals: np.ndarray, ch_names: Optional[list] = None, band_name: str = "Alpha") -> Optional[bytes]:
    """
    Create a topography image (PNG bytes) for given per-channel band values.
    Uses matplotlib + griddata for interpolation. If not available, returns placeholder path (string).
    """
    if band_vals is None:
        return None
    # If matplotlib or griddata unavailable, return placeholder filename or None
    if not HAS_MATPLOTLIB or griddata is None:
        return str(TOPO_PLACEHOLDER) if TOPO_PLACEHOLDER.exists() else None
    try:
        # basic 10-20 approximate coordinates for common channels (best-effort)
        coords = []
        labels = []
        if ch_names:
            names = [n.upper() for n in ch_names]
            approx = {
                "FP1": (-0.3, 0.9), "FP2": (0.3, 0.9),
                "F3": (-0.5, 0.5), "F4": (0.5, 0.5), "F7": (-0.8, 0.2), "F8": (0.8, 0.2),
                "C3": (-0.5, 0.0), "C4": (0.5, 0.0),
                "P3": (-0.5, -0.5), "P4": (0.5, -0.5),
                "O1": (-0.3, -0.9), "O2": (0.3, -0.9)
            }
            for v in names:
                placed = False
                for k, p in approx.items():
                    if k in v:
                        coords.append(p)
                        labels.append(v)
                        placed = True
                        break
                if not placed:
                    coords.append((np.random.uniform(-0.9, 0.9), np.random.uniform(-0.9, 0.9)))
                    labels.append(v)
        else:
            # distribute around circle
            nch = len(band_vals)
            thetas = np.linspace(0, 2 * np.pi, nch, endpoint=False)
            coords = [(0.8 * np.sin(t), 0.8 * np.cos(t)) for t in thetas]
            labels = [f"ch{i}" for i in range(len(coords))]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        vals = np.array(band_vals[:len(coords)])
        # grid
        xi = np.linspace(-1.0, 1.0, 160)
        yi = np.linspace(-1.0, 1.0, 160)
        XI, YI = np.meshgrid(xi, yi)
        Z = griddata((xs, ys), vals, (XI, YI), method="cubic", fill_value=np.nan)
        fig = plt.figure(figsize=(4, 4), dpi=120)
        ax = fig.add_subplot(111)
        cmap = cm.get_cmap("RdBu_r")
        im = ax.imshow(Z, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{band_name} topography", fontsize=9)
        circle = plt.Circle((0, 0), 0.95, color="k", fill=False, linewidth=1)
        ax.add_artist(circle)
        ax.scatter(xs, ys, s=20, c="k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("Topomap generation failed:", e)
        return str(TOPO_PLACEHOLDER) if TOPO_PLACEHOLDER.exists() else None

# ---------------- Connectivity ----------------
def compute_connectivity(data: np.ndarray, sf: Optional[float], ch_names: Optional[list] = None, band: Tuple[float, float] = (8.0, 13.0)) -> Tuple[Optional[np.ndarray], str]:
    """
    Compute connectivity matrix. Prefer mne_connectivity if available,
    otherwise compute pairwise coherence via scipy.signal.coherence.
    Returns (matrix, narrative)
    """
    try:
        nchan = int(data.shape[0])
        lo, hi = band
        # try mne/mne_connectivity path if available
        if HAS_MNE and HAS_MNE_CONN:
            try:
                info = mne.create_info(ch_names=ch_names if ch_names else [f"ch{i}" for i in range(nchan)], sfreq=sf or DEFAULT_SF, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                # Use mne_connectivity.spectral_connectivity if present
                try:
                    from mne_connectivity import spectral_connectivity
                    con = spectral_connectivity(raw, method=["wpli", "pli", "coh"], sfreq=sf or DEFAULT_SF, fmin=lo, fmax=hi, mt_adaptive=False)
                    # con is dict-like in some versions; attempt to extract mean matrix
                    if isinstance(con, dict) and "coh" in con:
                        mat = con["coh"].mean(axis=0)
                    else:
                        # try common attributes
                        try:
                            mat = np.mean([v.mean(axis=0) for v in con.values()], axis=0)
                        except Exception:
                            mat = np.eye(nchan)
                    return mat, f"Connectivity computed using mne_connectivity in {lo}-{hi} Hz."
                except Exception:
                    # fallback to mne.connectivity.spectral_connectivity if present
                    try:
                        from mne.connectivity import spectral_connectivity as sc
                        con = sc(raw, method='coh', sfreq=sf or DEFAULT_SF, fmin=lo, fmax=hi)
                        # depending on API, return dense matrix
                        if hasattr(con, 'get_data'):
                            mat = con.get_data(output='dense') if hasattr(con, 'get_data') else np.eye(nchan)
                        else:
                            mat = np.eye(nchan)
                        return mat, f"Connectivity computed using mne (coherence) in {lo}-{hi} Hz."
                    except Exception:
                        pass
            except Exception as e:
                print("mne connectivity path failed:", e)
        # fallback: pairwise coherence (scipy.signal.coherence)
        mat = np.zeros((nchan, nchan), dtype=float)
        for i in range(nchan):
            for j in range(i, nchan):
                try:
                    f, Cxy = coherence(data[i], data[j], fs=sf or DEFAULT_SF, nperseg=min(1024, max(256, data.shape[1])))
                    mask = (f >= lo) & (f <= hi)
                    if mask.sum() > 0:
                        val = float(np.nanmean(Cxy[mask]))
                        if not np.isfinite(val):
                            val = 0.0
                    else:
                        val = 0.0
                except Exception:
                    val = 0.0
                mat[i, j] = val; mat[j, i] = val
        return mat, f"Connectivity (mean coherence) in {lo}-{hi} Hz computed via scipy.signal.coherence."
    except Exception as e:
        print("Connectivity computation error:", e)
        return None, f"Connectivity computation failed: {str(e)}"

# ---------------- Part 2/3: UI, forms, processing entry points ----------------

# Header / top bar
st.markdown("""
<style>
.block-container { max-width:1200px; }
.header { background: linear-gradient(90deg,#0b3d91,#2451a6); color:white; padding:14px; border-radius:8px; }
.small { color:#cbd5e1; font-size:13px; }
.sidebar .stSelectbox { width: 100%; }
</style>
""", unsafe_allow_html=True)

col_main, col_logo = st.columns([4,1])
with col_main:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro – AI EEG Assistant</h2><div class='small'>EEG / QEEG + Explainable AI — Clinical support</div></div>", unsafe_allow_html=True)
with col_logo:
    if LOGO_PATH.exists():
        # use_container_width is the new parameter per Streamlit warnings
        try:
            st.image(str(LOGO_PATH), width=140, use_container_width=False)
        except Exception:
            st.image(str(LOGO_PATH), use_container_width=True)

# Sidebar: settings & patient info
with st.sidebar:
    st.header("Settings & Patient")
    lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)
    st.markdown("---")
    st.subheader("Patient information")
    patient_name = st.text_input("Name / الاسم")
    patient_id = st.text_input("ID")
    # limit DOB range wide enough
    dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1900,1,1), max_value=date.today())
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))
    st.markdown("---")
    st.subheader("Clinical & Labs")
    lab_options = ["Vitamin B12","Thyroid (TSH)","Vitamin D","Folate","Homocysteine","HbA1C","Cholesterol / Lipids"]
    selected_labs = st.multiselect("Available lab results", options=lab_options)
    lab_notes = st.text_area("Notes / lab values (optional)", help="Enter numeric values or comments for relevant labs")
    meds = st.text_area("Current medications (name + dose)")
    conditions = st.text_area("Comorbid conditions (e.g., diabetes, hypertension, anxiety)")
    st.markdown("---")
    st.write(f"Backends: mne={HAS_MNE} mne_conn={HAS_MNE_CONN} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

# Main area: upload + forms
st.markdown("### 1) Upload EDF file(s) (.edf) — you can upload multiple files to compare")
uploads = st.file_uploader("Drag & drop EDF files here", type=["edf"], accept_multiple_files=True)

# PHQ-9 (with corrected Q3/Q5/Q8 options)
st.markdown("### 2) PHQ-9 (Depression screening)")
PHQ_EN = [
 "Little interest or pleasure in doing things",
 "Feeling down, depressed, or hopeless",
 "Sleep changes (choose below)",
 "Feeling tired or having little energy",
 "Appetite changes (choose below)",
 "Feeling bad about yourself — or that you are a failure",
 "Trouble concentrating on things, such as reading or watching TV",
 "Moving or speaking slowly OR being fidgety/restless",
 "Thoughts that you would be better off dead or of harming yourself"
]
PHQ_AR = [
 "قلة الاهتمام أو المتعة بالأشياء",
 "الشعور بالحزن أو اليأس",
 "تغيرات النوم (اختيارات أدناه)",
 "الشعور بالتعب أو قلة الطاقة",
 "تغيرات الشهية (اختيارات أدناه)",
 "الشعور بالسوء تجاه النفس أو أنك فاشل",
 "صعوبة في التركيز",
 "تباطؤ في الحركة/التكلم أو التململ/القلق",
 "أفكار بإيذاء النفس"
]

phq_answers = {}
for i in range(1, 10):
    if lang == 'ar':
        label = reshape_ar(PHQ_AR[i-1])
        st.markdown(f"**{label}**")
    else:
        label = f"Q{i}. {PHQ_EN[i-1]}"
    # special options
    if i == 3:
        opts_display = [
            "0 — Not at all" if lang == 'en' else f"0 — {reshape_ar('لا')}",
            "1 — Insomnia (difficulty falling/staying asleep)" if lang == 'en' else f"1 — {reshape_ar('أرق (صعوبة في النوم)')}",
            "2 — Sleeping less" if lang == 'en' else f"2 — {reshape_ar('قلة النوم')}",
            "3 — Sleeping more" if lang == 'en' else f"3 — {reshape_ar('زيادة النوم')}"
        ]
    elif i == 5:
        opts_display = [
            "0 — Not at all" if lang == 'en' else f"0 — {reshape_ar('لا')}",
            "1 — Eating less" if lang == 'en' else f"1 — {reshape_ar('قلة الأكل')}",
            "2 — Eating more" if lang == 'en' else f"2 — {reshape_ar('زيادة الأكل')}",
            "3 — Both / variable" if lang == 'en' else f"3 — {reshape_ar('متغير / كلاهما')}"
        ]
    elif i == 8:
        opts_display = [
            "0 — Not at all" if lang == 'en' else f"0 — {reshape_ar('لا')}",
            "1 — Moving/speaking slowly" if lang == 'en' else f"1 — {reshape_ar('تباطؤ في الحركة/التكلم')}",
            "2 — Fidgety / restless" if lang == 'en' else f"2 — {reshape_ar('تململ/قلق')}",
            "3 — Both / variable" if lang == 'en' else f"3 — {reshape_ar('متغير / كلاهما')}"
        ]
    else:
        opts_display = [
            "0 — Not at all" if lang == 'en' else f"0 — {reshape_ar('لا شيء')}",
            "1 — Several days" if lang == 'en' else f"1 — {reshape_ar('عدة أيام')}",
            "2 — More than half the days" if lang == 'en' else f"2 — {reshape_ar('أكثر من نصف الأيام')}",
            "3 — Nearly every day" if lang == 'en' else f"3 — {reshape_ar('تقریباً كل يوم')}"
        ]

    key = f"phq_{i}_{lang}"
    if lang == 'en':
        ans = st.radio(label, options=opts_display, index=0, key=key, horizontal=True)
    else:
        ans = st.radio("", options=opts_display, index=0, key=key, horizontal=True)
    # parse numeric value
    try:
        val = int(str(ans).split("—")[0].strip())
    except Exception:
        s = str(ans).strip()
        val = int(s[0]) if s and s[0].isdigit() else 0
    phq_answers[f"Q{i}"] = val

phq_total = sum(phq_answers.values())
st.info(f"PHQ-9 total: {phq_total} (0–27)")

# AD8
st.markdown("### 3) AD8 (Cognitive screening)")
AD8_EN = [
 "Problems with judgment (bad decisions)",
 "Less interest in hobbies/activities",
 "Repeats questions/stories",
 "Trouble learning to use a tool or gadget",
 "Forgetting the correct month or year",
 "Difficulty handling complicated financial affairs",
 "Trouble remembering appointments",
 "Daily problems with thinking and memory"
]
AD8_AR = [
 "مشاكل في الحكم (مثل اتخاذ قرارات سيئة)",
 "قلة الاهتمام بالهوايات/الأنشطة",
 "تكرار الأسئلة/القصص",
 "صعوبة تعلم استخدام جهاز/أداة",
 "نسيان الشهر أو السنة الصحيحة",
 "صعوبة في التعامل مع الأمور المالية المعقدة",
 "صعوبة في تذكر المواعيد",
 "مشاكل يومية في التفكير والذاكرة"
]

ad8_answers = {}
for i, txt in enumerate(AD8_EN, start=1):
    if lang == 'en':
        label = f"A{i}. {txt}"
        val = st.radio(label, options=[0,1], index=0, key=f"ad8_{i}_en", horizontal=True)
    else:
        label = reshape_ar(AD8_AR[i-1])
        st.markdown(f"**{label}**")
        val = st.radio("", options=[0,1], index=0, key=f"ad8_{i}_ar", horizontal=True)
    ad8_answers[f"A{i}"] = int(val)

ad8_total = sum(ad8_answers.values())
st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

# Processing options
st.markdown("---")
st.header("Processing & Visualization Options")
use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
do_topomap = st.checkbox("Generate topography maps (if matplotlib available)", value=True)
do_connectivity = st.checkbox("Compute connectivity (Alpha band coherence)", value=True)
run_models = st.checkbox("Run ML models if provided (model_depression.pkl / model_alzheimer.pkl)", value=False)

# Prepare a button to trigger processing (if files uploaded)
if uploads and st.button("Process uploaded EDF(s)"):
    processing_placeholder = st.empty()
    results = []
    for up in uploads:
        processing_placeholder.info(f"Processing {up.name} ...")
        try:
            tmp = save_tmp_upload(up)
            edf = read_edf(tmp)
            data = edf.get("data")
            sf = edf.get("sfreq") or DEFAULT_SF
            ch_names = edf.get("ch_names")
            st.success(f"Loaded {up.name}: backend={edf.get('backend')} channels={data.shape[0]} sfreq={sf}")
            # preprocess
            cleaned = preprocess_data(data, sf, do_notch=use_notch)
            # compute PSD
            dfbands = compute_psd_bands(cleaned, sf)
            # aggregate features
            agg = aggregate_bands(dfbands, ch_names=ch_names)
            # ensure gamma exists
            if "gamma_rel_mean" not in agg:
                try:
                    agg["gamma_rel_mean"] = float(np.nanmean(dfbands["Gamma_rel"].values)) if not dfbands.empty else 0.0
                except Exception:
                    agg["gamma_rel_mean"] = 0.0

            # topomaps
            topo_images = {}
            if do_topomap:
                for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                    try:
                        vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(cleaned.shape[0])
                    except Exception:
                        vals = np.zeros(cleaned.shape[0])
                    img = generate_topomap_image(vals, ch_names=ch_names, band_name=band) if vals is not None else None
                    topo_images[band] = img

            # connectivity
            conn_img = None
            conn_mat = None
            conn_narr = None
            if do_connectivity:
                conn_mat, conn_narr = compute_connectivity(cleaned, sf, ch_names=ch_names, band=BANDS.get("Alpha", (8.0,13.0)))
                if conn_mat is not None and HAS_MATPLOTLIB:
                    try:
                        fig = plt.figure(figsize=(4,3)); plt.imshow(conn_mat, cmap='viridis'); plt.colorbar(); plt.title("Connectivity (Alpha)"); buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                    except Exception as e:
                        print("Connectivity image creation failed:", e)
                else:
                    conn_img = None

            results.append({
                "filename": up.name,
                "agg_features": agg,
                "df_bands": dfbands,
                "topo_images": topo_images,
                "connectivity_image": conn_img,
                "connectivity_matrix": conn_mat,
                "connectivity_narrative": conn_narr
            })
            processing_placeholder.success(f"Processed {up.name}")
        except Exception as e:
            processing_placeholder.error(f"Failed processing {up.name}: {e}")
            _trace(e)
    # show brief table of aggregated features
    if results:
        st.markdown("### Aggregated features (first file shown)")
        try:
            st.write(pd.Series(results[0]["agg_features"]))
        except Exception:
            st.write(results[0]["agg_features"])
else:
    results = []
# ---------------- Part 3/3: XAI, ML scoring, Exec summary, PDF generation & download ----------------

# helper: load shap json if exists
def load_shap_summary_json():
    if SHAP_JSON.exists():
        try:
            with open(SHAP_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Failed to read shap_summary.json:", e)
            return None
    return None

# Load shap summary if present
shap_summary = load_shap_summary_json()

# Build summary dict (patient + clinical + files)
summary = {
    "patient": {"name": patient_name or "-", "id": patient_id or "-", "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": phq_answers},
    "ad8": {"total": ad8_total, "items": ad8_answers},
    "clinical": {"labs": selected_labs, "lab_notes": lab_notes, "meds": meds, "conditions": conditions},
    "files": results,
    "xai": None,
    "ml_risk": None,
    "risk_category": None,
    "qeegh": None,
    "recommendations": []
}

# Heuristic QEEG interpretation (first file)
if results:
    agg0 = results[0].get("agg_features", {})
    ta = agg0.get("theta_alpha_ratio", None)
    asym = agg0.get("alpha_asym_F3_F4", None)
    if ta is not None:
        if ta > 1.4:
            summary["qeegh"] = "Elevated Theta/Alpha ratio consistent with early cognitive slowing."
        elif ta > 1.1:
            summary["qeegh"] = "Mild elevation of Theta/Alpha; correlate clinically."
        else:
            summary["qeegh"] = "Theta/Alpha within expected range."
    if results[0].get("connectivity_narrative"):
        summary["connectivity"] = results[0].get("connectivity_narrative")

# Attach XAI data
if shap_summary:
    summary["xai"] = shap_summary
else:
    # fallback: attempt to read feature_importances_ from pickled models if joblib available
    fallback_xai = {}
    try:
        if HAS_JOBLIB and MODEL_DEP.exists():
            md = joblib.load(str(MODEL_DEP))
            if hasattr(md, "feature_importances_"):
                names = list(md.feature_names_in_) if hasattr(md, "feature_names_in_") else [f"f{i}" for i in range(len(md.feature_importances_))]
                fallback_xai["depression_global"] = dict(zip(names, md.feature_importances_.tolist()))
        if HAS_JOBLIB and MODEL_AD.exists():
            ma = joblib.load(str(MODEL_AD))
            if hasattr(ma, "feature_importances_"):
                names = list(ma.feature_names_in_) if hasattr(ma, "feature_names_in_") else [f"f{i}" for i in range(len(ma.feature_importances_))]
                fallback_xai["alzheimers_global"] = dict(zip(names, ma.feature_importances_.tolist()))
    except Exception as e:
        print("Fallback XAI failed:", e)
    if fallback_xai:
        summary["xai"] = fallback_xai

# ML predictions
mlrisk = None
if run_models and HAS_JOBLIB and (MODEL_DEP.exists() or MODEL_AD.exists()) and results:
    try:
        Xdf = pd.DataFrame([r.get("agg_features", {}) for r in results]).fillna(0)
        preds = []
        if MODEL_DEP.exists():
            mdep = joblib.load(str(MODEL_DEP))
            if hasattr(mdep, "predict_proba"):
                p = mdep.predict_proba(Xdf)[:,1]
            else:
                p = mdep.predict(Xdf)
            summary.setdefault("predictions", {})["depression_prob"] = [float(x) for x in p]
            preds.append(np.mean(p))
        if MODEL_AD.exists():
            mad = joblib.load(str(MODEL_AD))
            if hasattr(mad, "predict_proba"):
                p2 = mad.predict_proba(Xdf)[:,1]
            else:
                p2 = mad.predict(Xdf)
            summary.setdefault("predictions", {})["alzheimers_prob"] = [float(x) for x in p2]
            preds.append(np.mean(p2))
        if preds:
            mlrisk = float(np.mean(preds) * 100.0)
    except Exception as e:
        st.warning("Model prediction failed: " + str(e))
        print("Model prediction exception:", e)

# fallback heuristic if models not run/present
if mlrisk is None:
    if results:
        ta_val = results[0]["agg_features"].get("theta_alpha_ratio", 1.0)
    else:
        ta_val = 1.0
    phq_norm = phq_total / 27.0
    ad8_norm = ad8_total / 8.0
    ta_norm = min(1.0, ta_val / 1.4)
    mlrisk = min(100.0, (ta_norm * 0.55 + phq_norm * 0.3 + ad8_norm * 0.15) * 100.0)

summary["ml_risk"] = mlrisk
if mlrisk >= 50:
    summary["risk_category"] = "High"
elif mlrisk >= 25:
    summary["risk_category"] = "Moderate"
else:
    summary["risk_category"] = "Low"

# Recommendations (structured)
if summary["phq9"]["total"] >= 10:
    summary["recommendations"].append("PHQ-9 suggests moderate/severe depression — consider psychiatric referral and treatment planning.")
if summary["ad8"]["total"] >= 2 or (results and results[0]["agg_features"].get("theta_alpha_ratio", 0) > 1.4):
    summary["recommendations"].append("AD8 elevated or Theta/Alpha increased — consider neurocognitive testing and neuroimaging (MRI/FDG-PET).")
summary["recommendations"].append("Correlate QEEG/connectivity findings with PHQ-9, AD8 and clinical interview.")
summary["recommendations"].append("Review medications that may affect EEG.")
if not summary["recommendations"]:
    summary["recommendations"].append("Clinical follow-up and re-evaluation in 3–6 months.")

# Executive Summary display
st.markdown("---")
st.subheader("Executive Summary")
st.metric(label="Final ML Risk Score", value=f"{summary['ml_risk']:.1f}%", delta=summary.get("risk_category", "-"))
st.write("Brief QEEG interpretation:"); st.write(summary.get("qeegh", "-"))

# Normative comparison charts (Theta/Alpha & Alpha asymmetry)
if results:
    agg0 = results[0]["agg_features"]
    tar = agg0.get("theta_alpha_ratio", None)
    asym = agg0.get("alpha_asym_F3_F4", None)
    if tar is not None and HAS_MATPLOTLIB:
        img = plot_norm_comparison("theta_alpha_ratio", tar, title="Theta/Alpha vs Norm")
        if img:
            st.image(img, caption="Theta/Alpha comparison", use_container_width=True)
    if asym is not None and HAS_MATPLOTLIB:
        img2 = plot_norm_comparison("alpha_asym_F3_F4", asym, title="Alpha Asymmetry (F3-F4)")
        if img2:
            st.image(img2, caption="Alpha Asymmetry comparison", use_container_width=True)

# XAI display area
st.markdown("---")
st.header("Explainable AI (XAI)")
if summary.get("xai"):
    st.write("Top contributors (from shap_summary.json or model importances):")
    st.json(summary["xai"])
else:
    st.info("XAI data not available. Upload shap_summary.json or model pickle to enable.")

# Detailed file cards
if results:
    st.markdown("---")
    st.header("Per-file details")
    for i, r in enumerate(results):
        st.subheader(f"[{i+1}] {r.get('filename')}")
        st.write("Aggregated features:")
        st.write(pd.Series(r.get("agg_features", {})))
        # display a few topomap images
        topo_imgs = r.get("topo_images", {})
        cols = st.columns(len(topo_imgs) if topo_imgs else 1)
        for j, (band, img) in enumerate(topo_imgs.items()):
            try:
                if isinstance(img, (bytes, bytearray)):
                    cols[j].image(img, caption=f"{band} topomap", use_container_width=True)
                elif isinstance(img, str) and Path(img).exists():
                    cols[j].image(str(img), caption=f"{band} topomap (placeholder)", use_container_width=True)
            except Exception:
                pass
        # connectivity matrix image
        if r.get("connectivity_image"):
            st.image(r.get("connectivity_image"), caption="Connectivity (Alpha)", use_container_width=True)
        elif r.get("connectivity_matrix") is not None:
            st.write("Connectivity matrix available (numeric).")

# ---------------- PDF generation ----------------
# register Amiri if available (reportlab functions were imported in part1)
def register_amiri_font(path: Optional[str] = None) -> str:
    if not HAS_REPORTLAB:
        return "Helvetica"
    try:
        ttf = path or (str(AMIRI_TTF) if AMIRI_TTF.exists() else None)
        if ttf and Path(ttf).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", ttf))
                return "Amiri"
            except Exception as e:
                print("Amiri register failed:", e)
        # fallback Helvetica
        return "Helvetica"
    except Exception as e:
        print("register_amiri_font failed:", e)
        return "Helvetica"

def generate_pdf_report(summary_obj: dict, lang: str = "en", amiri_path: Optional[str] = None) -> bytes:
    """
    Create a PDF (bytes) with ReportLab. Uses Amiri font if provided for Arabic.
    Layout: title, ML risk, patient meta, QEEG summary, per-file sections, XAI section, recommendations, footer/logo.
    """
    if not HAS_REPORTLAB:
        # fallback: return a JSON as bytes
        return json.dumps(summary_obj, indent=2, ensure_ascii=False).encode("utf-8")

    font_name = register_amiri_font(amiri_path)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 36
    x = margin
    y = H - margin

    # Title
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font_name, 16)
    if lang == "en":
        c.drawCentredString(W/2, y, title_en)
    else:
        c.drawCentredString(W/2, y, reshape_ar(title_ar))
    y -= 28

    # ML Risk
    ml = summary_obj.get("ml_risk", None)
    cat = summary_obj.get("risk_category", "-")
    c.setFont(font_name, 12)
    if ml is not None:
        if lang == "en":
            c.drawString(x, y, f"Final ML Risk Score: {format_for_pdf_value(ml)}%    Category: {cat}")
        else:
            c.drawRightString(W - margin, y, reshape_ar(f"معدل الخطر ML النهائي: {format_for_pdf_value(ml)}%    الفئة: {cat}"))
    y -= 18

    # Patient meta
    c.setFont(font_name, 10)
    p = summary_obj.get("patient", {})
    if lang == "en":
        c.drawString(x, y, f"Patient: {p.get('name','-')}    ID: {p.get('id','-')}    DOB: {p.get('dob','-')}")
    else:
        c.drawRightString(W - margin, y, reshape_ar(f"المريض: {p.get('name','-')}    المعرف: {p.get('id','-')}    الميلاد: {p.get('dob','-')}"))
    y -= 14

    # QEEG interpretation
    q = summary_obj.get("qeegh", "-")
    if lang == "en":
        c.drawString(x, y, f"QEEG Interpretation: {q}")
    else:
        c.drawRightString(W - margin, y, reshape_ar("تفسير QEEG: " + str(q)))
    y -= 16

    # Clinical context
    clinical = summary_obj.get("clinical", {})
    if lang == "en":
        c.drawString(x, y, "Clinical context:"); y -= 12
        c.drawString(x + 6, y, f"Labs: {', '.join(clinical.get('labs', [])) if clinical.get('labs') else 'None'}"); y -= 10
        c.drawString(x + 6, y, f"Medications: {clinical.get('meds', 'None')}"); y -= 10
        c.drawString(x + 6, y, f"Comorbidities: {clinical.get('conditions', 'None')}"); y -= 12
    else:
        c.drawRightString(W - margin, y, reshape_ar("السياق السريري:")); y -= 12
        c.drawRightString(W - margin, y, reshape_ar(f"التحاليل: {', '.join(clinical.get('labs', [])) if clinical.get('labs') else 'لا'}")); y -= 10
        c.drawRightString(W - margin, y, reshape_ar(f"الأدوية: {clinical.get('meds', 'لا')}")); y -= 10
        c.drawRightString(W - margin, y, reshape_ar(f"الأمراض المصاحبة: {clinical.get('conditions', 'لا')}")); y -= 12

    # Files details
    files = summary_obj.get("files", [])
    for idx, fd in enumerate(files):
        if y < 220:
            c.showPage(); y = H - margin
        fname = fd.get("filename", f"File_{idx+1}")
        agg = fd.get("agg_features", {})
        if lang == "en":
            c.drawString(x, y, f"[{idx+1}] File: {fname}"); y -= 12
        else:
            c.drawRightString(W - margin, y, reshape_ar(f"[{idx+1}] الملف: {fname}")); y -= 12

        # Bands table
        col1 = x + 6; col2 = x + 160; col3 = x + 300
        if lang == "en":
            c.drawString(col1, y, "Band"); c.drawString(col2, y, "Abs_mean"); c.drawString(col3, y, "Rel_mean")
        else:
            c.drawRightString(W - margin, y, reshape_ar("التردد")); c.drawRightString(W - margin - 120, y, reshape_ar("المطلق")); c.drawRightString(W - margin - 30, y, reshape_ar("النسبي"))
        y -= 12
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            a = agg.get(f"{band.lower()}_abs_mean", 0.0)
            r = agg.get(f"{band.lower()}_rel_mean", 0.0)
            if lang == "en":
                c.drawString(col1, y, f"{band}"); c.drawString(col2, y, f"{a:>12.4f}"); c.drawString(col3, y, f"{r:>12.4f}")
            else:
                c.drawRightString(W - margin, y, reshape_ar(f"{band}")); c.drawRightString(W - margin - 120, y, format_for_pdf_value(a)); c.drawRightString(W - margin - 30, y, format_for_pdf_value(r))
            y -= 10
        y -= 8

        # feature list
        c.setFont(font_name, 9)
        if lang == "en":
            c.drawString(col1, y, "Feature"); c.drawString(col2, y, "Value"); y -= 10
            feat_list = ["theta_alpha_ratio", "alpha_asym_F3_F4", "theta_beta_ratio", "beta_alpha_ratio", "gamma_rel_mean"]
            for fn in feat_list:
                val = agg.get(fn, None)
                c.drawString(col1, y, f"{fn}") ; c.drawString(col2, y, format_for_pdf_value(val)); y -= 9
        else:
            c.drawRightString(W - margin, y, reshape_ar("الميزة")); c.drawRightString(W - margin - 120, y, reshape_ar("القيمة")); y -= 10
            feat_list = ["theta_alpha_ratio", "alpha_asym_F3_F4", "theta_beta_ratio", "beta_alpha_ratio", "gamma_rel_mean"]
            for fn in feat_list:
                val = agg.get(fn, None)
                c.drawRightString(W - margin, y, reshape_ar(f"{fn}")); c.drawRightString(W - margin - 120, y, reshape_ar(format_for_pdf_value(val))); y -= 9
        y -= 12

        # topomaps images
        topo_imgs = fd.get("topo_images", {})
        if topo_imgs:
            img_w = 110; colcount = 0
            for band, img in topo_imgs.items():
                if img is None: continue
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else ImageReader(str(img))
                    xi = x + (colcount % 4) * (img_w + 8)
                    yi = y - img_w
                    c.drawImage(ir, xi, yi, width=img_w, height=img_w, mask='auto')
                    c.setFont(font_name, 8); c.drawString(xi, yi - 10, band)
                    colcount += 1
                    if colcount % 4 == 0:
                        y -= (img_w + 24)
                except Exception:
                    pass
            y -= (img_w + 12)
        else:
            if TOPO_PLACEHOLDER.exists():
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(str(TOPO_PLACEHOLDER)); c.drawImage(ir, x, y - 140, width=240, height=140, mask='auto'); y -= 160
                except Exception:
                    y -= 6

        # connectivity image
        if fd.get("connectivity_image"):
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(io.BytesIO(fd.get("connectivity_image")))
                c.drawImage(ir, x, y - 160, width=320, height=160, mask='auto')
                y -= 170
            except Exception:
                y -= 6
        else:
            if CONN_PLACEHOLDER.exists():
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(str(CONN_PLACEHOLDER)); c.drawImage(ir, x, y - 140, width=240, height=140, mask='auto'); y -= 160
                except Exception:
                    y -= 6
        y -= 6

    # XAI section
    if y < 180:
        c.showPage(); y = H - margin
    if lang == "en":
        c.drawString(x, y, "Explainable AI — Top contributors:"); y -= 12
    else:
        c.drawRightString(W - margin, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:")); y -= 12
    xai = summary_obj.get("xai", None)
    if xai:
        # xai might be dict of groups
        if isinstance(xai, dict):
            for gk, gv in xai.items():
                if isinstance(gv, dict):
                    if lang == "en":
                        c.drawString(x + 6, y, f"{gk}:"); y -= 10
                    else:
                        c.drawRightString(W - margin, y, reshape_ar(f"{gk}:")); y -= 10
                    for kk, vv in sorted(gv.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int, float)) else 0)[:20]:
                        if lang == "en":
                            c.drawString(x + 10, y, f"{kk}: {format_for_pdf_value(vv)}"); y -= 9
                        else:
                            c.drawRightString(W - margin, y, reshape_ar(f"{kk}: {format_for_pdf_value(vv)}")); y -= 9
                else:
                    if lang == "en":
                        c.drawString(x + 6, y, f"{gk}: {format_for_pdf_value(gv)}"); y -= 10
                    else:
                        c.drawRightString(W - margin, y, reshape_ar(f"{gk}: {format_for_pdf_value(gv)}")); y -= 10
    else:
        if lang == "en":
            c.drawString(x + 6, y, "XAI not available (no shap_summary.json or model)."); y -= 10
        else:
            c.drawRightString(W - margin, y, reshape_ar("لا توجد بيانات XAI (لا يوجد shap_summary.json أو نموذج).")); y -= 10

    y -= 8
    # Recommendations
    if lang == "en":
        c.drawString(x, y, "Structured Clinical Recommendations:"); y -= 12
    else:
        c.drawRightString(W - margin, y, reshape_ar("التوصيات السريرية المنظمة:")); y -= 12
    for r in summary_obj.get("recommendations", []):
        if lang == "en":
            c.drawString(x + 6, y, "- " + r); y -= 10
        else:
            c.drawRightString(W - margin, y, reshape_ar("- " + r)); y -= 10

    # Footer + logo
    c.setFont(font_name, 8)
    try:
        if LOGO_PATH.exists():
            from reportlab.lib.utils import ImageReader
            ir = ImageReader(str(LOGO_PATH))
            c.drawImage(ir, W - 160, 10, width=120, height=40, mask='auto')
    except Exception:
        pass

    footer_en = "Designed and developed by Golden Bird LLC — Vista Kaviani"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني")
    disc_en = "Research/demo only — Not a clinical diagnosis."
    disc_ar = reshape_ar("هذه لأغراض البحث/العرض فقط — ليست تشخیصًا طبیًا نهائیًا.")
    if lang == "en":
        c.drawCentredString(W/2, 30, footer_en); c.drawCentredString(W/2, 18, disc_en)
    else:
        c.drawCentredString(W/2, 30, footer_ar); c.drawCentredString(W/2, 18, disc_ar)

    c.save()
    buf.seek(0)
    return buf.read()

# UI: Generate PDF
st.markdown("---")
st.header("Generate report")
colA, colB = st.columns([3,1])
with colA:
    report_lang = st.selectbox("Report language / لغة التقرير", options=["en","ar"], index=0)
    amiri_path = st.text_input("Amiri TTF path (optional) — leave blank if Amiri-Regular.ttf in root", value="")
with colB:
    if st.button("Generate PDF report"):
        try:
            pdfb = generate_pdf_report(summary, lang=report_lang, amiri_path=(amiri_path or None))
            st.download_button("Download PDF", data=pdfb, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("Report generated. Footer includes Golden Bird LLC.")
        except Exception as e:
            st.error("PDF generation failed.")
            _trace(e)

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis.")

