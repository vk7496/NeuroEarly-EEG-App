# app.py — NeuroEarly Pro (Final Clinical Edition) — Part 1/3
# Medical Blue Theme, English default, Arabic optional
# Features: EDF load (mne/pyedflib), robust preprocessing, PSD, Focal Delta Index,
# Spectral Slowing, Connectivity (coherence/PLI/wPLI), Topomaps (incl Gamma), plotting utilities.

import os
import io
import json
import math
import tempfile
import traceback
import base64
import datetime
from datetime import date
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy libs (import if available)
HAS_MNE = False
HAS_PYEDF = False
HAS_MNE_CONN = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_ARABIC = False
HAS_SHAP = False
HAS_JOBLIB = False

try:
    import mne
    HAS_MNE = True
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
    from matplotlib import patches
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

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

# SciPy functions (assume installed via requirements)
from scipy.signal import welch, butter, filtfilt, iirnotch, coherence
from scipy.integrate import trapezoid

# ----------------- Project Files & Assets -----------------
ROOT = Path(".")
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"
TOPO_PLACEHOLDER = ASSETS_DIR / "topo_placeholder.png"
CONN_PLACEHOLDER = ASSETS_DIR / "conn_placeholder.png"
SHAP_JSON = ROOT / "shap_summary.json"
MODEL_DEP = ROOT / "model_depression.pkl"
MODEL_AD = ROOT / "model_alzheimer.pkl"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"

# ----------------- Bands & Defaults -----------------
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}
DEFAULT_SF = 256.0

# Normative ranges for bar charts (example values; can be adjusted)
NORM_RANGES = {
    "theta_alpha_ratio": {"healthy_low": 0.0, "healthy_high": 1.1, "at_risk_low": 1.1, "at_risk_high": 1.4},
    "alpha_asym_F3_F4": {"healthy_low": -0.05, "healthy_high": 0.05, "at_risk_low": -0.2, "at_risk_high": -0.05},
    "gamma_rel_mean": {"healthy_low": 0.02, "healthy_high": 0.06, "low_risk": 0.02, "high_risk": 0.10}
}

# ----------------- Streamlit page & styling (Medical Blue Theme) -----------------
st.set_page_config(page_title="NeuroEarly Pro – AI EEG Assistant", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{
  --main-blue: #0b63d6;
  --light-blue: #eaf3ff;
  --muted: #6b7280;
}
.header {
  background: linear-gradient(90deg, #0b63d6, #2b8cff);
  color: white; padding:14px; border-radius:10px;
}
.kv { color: var(--muted); font-size:13px; }
.card { background: white; border-radius:8px; padding:12px; box-shadow: 0 2px 6px rgba(11,99,214,0.08); }
</style>
""", unsafe_allow_html=True)

# ----------------- Utilities -----------------
def now_ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e: Exception) -> None:
    tb = traceback.format_exc()
    st.error("Internal error — see logs")
    st.code(tb)
    print(tb)

def reshape_ar(text: str) -> str:
    if not text:
        return ""
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def format_for_pdf_value(v: Any) -> str:
    try:
        if v is None:
            return "N/A"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "N/A"
            return f"{v:.4f}"
        if isinstance(v, dict):
            return "; ".join([f"{k}={format_for_pdf_value(val)}" for k, val in v.items()])
        if isinstance(v, (list, tuple)):
            return ", ".join([format_for_pdf_value(x) for x in v])
        return str(v)
    except Exception:
        return str(v)

# ----------------- EDF read utilities -----------------
def save_tmp_upload(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path: str) -> Dict[str, Any]:
    """
    Return: dict {backend, raw (if mne), data (channels x samples), ch_names, sfreq}
    """
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data()
            chs = raw.ch_names
            sf = raw.info.get("sfreq", None)
            return {"backend": "mne", "raw": raw, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            print("mne read failed:", e)
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
            nsamps = f.getNSamples()[0] if hasattr(f, "getNSamples") else None
            for i in range(n):
                try:
                    s = f.readSignal(i).astype(np.float64)
                    sigs.append(s)
                except Exception:
                    sigs.append(np.zeros(nsamps if nsamps else 1))
            f._close()
            data = np.vstack(sigs)
            return {"backend": "pyedflib", "raw": None, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            raise IOError(f"pyedflib failed: {e}")
    raise ImportError("No EDF backend found. Install mne or pyedflib.")

# ----------------- Filtering & Preprocess -----------------
def notch_filter(sig: np.ndarray, sf: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    if sf is None or sf <= 0:
        return sig
    try:
        b, a = iirnotch(freq, Q, sf)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig: np.ndarray, sf: float, low: float = 0.5, high: float = 45.0, order: int = 4) -> np.ndarray:
    if sf is None or sf <= 0:
        return sig
    try:
        nyq = 0.5 * sf
        low_n = max(low / nyq, 1e-6)
        high_n = min(high / nyq, 0.999)
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(raw_data: np.ndarray, sf: float, do_notch: bool = True) -> np.ndarray:
    cleaned = np.zeros_like(raw_data, dtype=np.float64)
    for i in range(raw_data.shape[0]):
        s = raw_data[i].astype(np.float64)
        if do_notch:
            s = notch_filter(s, sf)
        s = bandpass_filter(s, sf)
        cleaned[i, :] = s
    return cleaned

# ----------------- PSD & band powers (robust) -----------------
def compute_psd_bands(data: np.ndarray, sf: float, nperseg: int = 1024) -> pd.DataFrame:
    """
    Return DataFrame: each row per channel, columns {Delta_abs, Delta_rel, Theta_abs, Theta_rel, ...}
    Robust to short signals and NaNs.
    """
    rows = []
    nch = int(data.shape[0]) if data is not None and len(data.shape) >= 1 else 0
    for ch in range(nch):
        sig = data[ch]
        if sig is None or len(sig) < 8:
            row = {"channel_idx": ch}
            for b in BANDS.keys():
                row[f"{b}_abs"] = 0.0
                row[f"{b}_rel"] = 0.0
            rows.append(row)
            continue
        try:
            freqs, pxx = welch(sig, fs=sf if sf else DEFAULT_SF, nperseg=min(nperseg, max(256, len(sig))))
            if freqs is None or pxx is None or len(freqs) == 0:
                total = 0.0
            else:
                pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)
                total = float(trapezoid(pxx, freqs)) if freqs.size > 0 else 0.0
        except Exception as e:
            print("welch error:", e)
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
                        seg_freqs = freqs[mask]; seg_pxx = pxx[mask]
                        seg_pxx = np.nan_to_num(seg_pxx, nan=0.0, posinf=0.0, neginf=0.0)
                        abs_p = float(trapezoid(seg_pxx, seg_freqs)) if seg_freqs.size > 0 else 0.0
                rel = float(abs_p / total) if total > 0 else 0.0
            except Exception as e:
                print("band compute error:", e)
                abs_p = 0.0; rel = 0.0
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        rows.append(row)
    if not rows:
        cols = ["channel_idx"] + [f"{b}_abs" for b in BANDS.keys()] + [f"{b}_rel" for b in BANDS.keys()]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)

# ----------------- Aggregate & Derived Features -----------------
def aggregate_bands(df_bands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df_bands is None or df_bands.empty:
        for band in BANDS.keys():
            out[f"{band.lower()}_abs_mean"] = 0.0
            out[f"{band.lower()}_rel_mean"] = 0.0
        out["theta_alpha_ratio"] = 0.0
        out["theta_beta_ratio"] = 0.0
        out["beta_alpha_ratio"] = 0.0
        out["alpha_asym_F3_F4"] = 0.0
        out["gamma_rel_mean"] = 0.0
        return out
    for band in BANDS.keys():
        try:
            out[f"{band.lower()}_abs_mean"] = float(np.nanmean(df_bands[f"{band}_abs"].values))
        except Exception:
            out[f"{band.lower()}_abs_mean"] = 0.0
        try:
            out[f"{band.lower()}_rel_mean"] = float(np.nanmean(df_bands[f"{band}_rel"].values))
        except Exception:
            out[f"{band.lower()}_rel_mean"] = 0.0
    alpha_rel = out.get("alpha_rel_mean", 1e-9)
    beta_rel = out.get("beta_rel_mean", 1e-9)
    theta_rel = out.get("theta_rel_mean", 0.0)
    out["theta_alpha_ratio"] = float(theta_rel / alpha_rel) if alpha_rel > 0 else 0.0
    out["theta_beta_ratio"] = float(theta_rel / beta_rel) if beta_rel > 0 else 0.0
    out["beta_alpha_ratio"] = float(beta_rel / alpha_rel) if alpha_rel > 0 else 0.0
    out["gamma_rel_mean"] = float(out.get("gamma_rel_mean", out.get("gamma_rel_mean", 0.0)))
    # Alpha asymmetry F3-F4 (best-effort)
    out["alpha_asym_F3_F4"] = 0.0
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def find_index(token):
                for i, nm in enumerate(names):
                    if token in nm:
                        return i
                return None
            i3 = find_index("F3"); i4 = find_index("F4")
            if i3 is not None and i4 is not None:
                a3 = df_bands.loc[df_bands['channel_idx'] == i3, 'Alpha_rel']
                a4 = df_bands.loc[df_bands['channel_idx'] == i4, 'Alpha_rel']
                if not a3.empty and not a4.empty:
                    out["alpha_asym_F3_F4"] = float(a3.values[0] - a4.values[0])
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# ----------------- Focal Delta / Tumor Indicators -----------------
def compute_focal_delta_index(df_bands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute Focal Delta Index (FDI) per channel/region:
      FDI_ch = Delta_power(ch) / Global_mean_delta_power
    Also compute pairwise asymmetry ratios for symmetric pairs (e.g., T7/T8).
    """
    res = {"fdi": {}, "asymmetry": {}, "focal_alerts": []}
    try:
        # compute per-channel delta abs power
        if df_bands is None or df_bands.empty:
            return res
        delta_vals = df_bands[["channel_idx", "Delta_abs"]].set_index("channel_idx")["Delta_abs"].to_dict()
        global_mean = np.nanmean(list(delta_vals.values())) if delta_vals else 0.0
        if not np.isfinite(global_mean) or global_mean <= 0:
            global_mean = 1e-9
        # compute FDI
        for idx, val in delta_vals.items():
            fdi = float(val / global_mean) if global_mean > 0 else 0.0
            res["fdi"][idx] = fdi
            # alert if above threshold
            if fdi > 2.0:
                chname = ch_names[idx] if ch_names and idx < len(ch_names) else f"ch{idx}"
                res["focal_alerts"].append({"channel_idx": idx, "channel": chname, "fdi": fdi})
        # symmetry checks (best-effort mapping)
        # build dict name->idx
        name_idx = {}
        if ch_names:
            for i, n in enumerate(ch_names):
                name_idx[n.upper()] = i
        # common symmetric pairs
        pairs = [("T7","T8"), ("F3","F4"), ("P3","P4"), ("O1","O2"), ("C3","C4")]
        for L, R in pairs:
            li = None; ri = None
            for nm, idx in name_idx.items():
                if L in nm: li = idx
                if R in nm: ri = idx
            if li is not None and ri is not None:
                dl = delta_vals.get(li, 0.0); dr = delta_vals.get(ri, 0.0)
                if dl <= 0 and dr <= 0:
                    ratio = 1.0
                elif dl == 0:
                    ratio = float("inf") if dr>0 else 1.0
                else:
                    ratio = float(dr / dl)
                res["asymmetry"][f"{L}/{R}"] = ratio
                if ratio > 3.0 or (isinstance(ratio, float) and ratio < 0.33):
                    res["focal_alerts"].append({"pair": f"{L}/{R}", "ratio": ratio})
    except Exception as e:
        print("compute_focal_delta_index error:", e)
    return res

# ----------------- Connectivity computation (coherence fallback) -----------------
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: Optional[List[str]] = None, band: Tuple[float,float]=(8.0,13.0)) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns (matrix, narrative). Prefer mne_connectivity if available; fallback to scipy coherence pairwise mean.
    """
    try:
        nchan = int(data.shape[0])
        lo, hi = band
        if HAS_MNE and HAS_MNE_CONN:
            try:
                info = mne.create_info(ch_names=ch_names if ch_names else [f"ch{i}" for i in range(nchan)], sfreq=sf)
                raw = mne.io.RawArray(data, info)
                # mne_connectivity spectral_connectivity usage may vary by version; handle flexibly
                try:
                    from mne_connectivity import spectral_connectivity
                    con = spectral_connectivity(raw, method=["coh","pli","wpli"], sfreq=sf, fmin=lo, fmax=hi, mt_adaptive=False)
                    # con may be dict-like; attempt to extract coherence matrix if present
                    if isinstance(con, dict) and "coh" in con:
                        mat = con["coh"].mean(axis=0)
                    else:
                        # try to aggregate available arrays
                        mats = []
                        if isinstance(con, dict):
                            for v in con.values():
                                try:
                                    mats.append(v.mean(axis=0))
                                except Exception:
                                    pass
                        if mats:
                            mat = np.mean(mats, axis=0)
                        else:
                            mat = np.eye(nchan)
                    return mat, f"Connectivity computed via mne_connectivity ({lo}-{hi}Hz)."
                except Exception:
                    # fallback to mne.connectivity if available
                    try:
                        from mne.connectivity import spectral_connectivity as sc
                        con_out = sc(raw, method='coh', sfreq=sf, fmin=lo, fmax=hi)
                        # depending on API
                        try:
                            mat = con_out.get_data(output='dense') if hasattr(con_out, 'get_data') else np.eye(nchan)
                        except Exception:
                            mat = np.eye(nchan)
                        return mat, f"Connectivity computed via mne ({lo}-{hi}Hz)."
                    except Exception:
                        pass
            except Exception as e:
                print("mne connectivity branch failed:", e)
        # fallback: pairwise coherence using scipy.signal.coherence
        mat = np.zeros((nchan, nchan), dtype=float)
        for i in range(nchan):
            for j in range(i, nchan):
                try:
                    f, Cxy = coherence(data[i], data[j], fs=sf if sf else DEFAULT_SF, nperseg=min(1024, max(256, data.shape[1])))
                    mask = (f >= lo) & (f <= hi)
                    val = float(np.nanmean(Cxy[mask])) if mask.sum()>0 else 0.0
                    if not np.isfinite(val):
                        val = 0.0
                except Exception:
                    val = 0.0
                mat[i,j] = val; mat[j,i] = val
        return mat, f"Connectivity (mean coherence) in {lo}-{hi} Hz computed via scipy."
    except Exception as e:
        print("compute_connectivity_matrix error:", e)
        return None, f"Connectivity computation failed: {e}"

# ----------------- Topomap generator (incl Gamma) -----------------
def generate_topomap_image(band_vals: np.ndarray, ch_names: Optional[List[str]]=None, band_name: str="Alpha") -> Optional[bytes]:
    """
    Create PNG bytes of topomap for given per-channel values. If matplotlib not available, return placeholder path.
    """
    if band_vals is None:
        return None
    if not HAS_MATPLOTLIB:
        return str(TOPO_PLACEHOLDER) if Path(TOPO_PLACEHOLDER).exists() else None
    try:
        # approximate 10-20 coords for common channels
        if ch_names:
            names = [n.upper() for n in ch_names]
            approx = {
                "FP1":(-0.3,0.9),"FP2":(0.3,0.9),"F3":(-0.5,0.5),"F4":(0.5,0.5),
                "F7":(-0.8,0.2),"F8":(0.8,0.2),"C3":(-0.5,0.0),"C4":(0.5,0.0),
                "P3":(-0.5,-0.5),"P4":(0.5,-0.5),"O1":(-0.3,-0.9),"O2":(0.3,-0.9),
                "T7":(-0.7,0.0),"T8":(0.7,0.0)
            }
            coords = []; vals = []
            for i, nm in enumerate(names):
                pt = None
                for k,p in approx.items():
                    if k in nm:
                        pt = p; break
                if pt is None:
                    pt = (np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9))
                coords.append(pt)
                vals.append(float(band_vals[i]) if i < len(band_vals) else 0.0)
        else:
            nch = len(band_vals)
            thetas = np.linspace(0, 2*np.pi, nch, endpoint=False)
            coords = [(0.8*np.sin(t), 0.8*np.cos(t)) for t in thetas]
            vals = list(band_vals[:nch])
        xs = np.array([c[0] for c in coords]); ys = np.array([c[1] for c in coords]); zs = np.array(vals)
        # grid interpolation
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(-1.0,1.0,180); yi = np.linspace(-1.0,1.0,180)
            XI, YI = np.meshgrid(xi, yi)
            Z = griddata((xs, ys), zs, (XI, YI), method='cubic', fill_value=np.nan)
        except Exception:
            # fallback radial smoothing
            XI, YI = np.meshgrid(np.linspace(-1,1,180), np.linspace(-1,1,180))
            Z = np.zeros_like(XI)
            for i in range(XI.shape[0]):
                for j in range(XI.shape[1]):
                    dists = np.sqrt((XI[i,j]-xs)**2 + (YI[i,j]-ys)**2)
                    weights = 1.0/(dists+1e-3)
                    Z[i,j] = np.sum(weights*zs)/np.sum(weights)
        fig = plt.figure(figsize=(4,4), dpi=120)
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap('RdBu_r')
        im = ax.imshow(Z, origin='lower', extent=[-1,1,-1,1], cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        circle = plt.Circle((0,0),0.95, color='k', fill=False, linewidth=1)
        ax.add_artist(circle)
        ax.scatter(xs, ys, s=18, c='k')
        ax.set_title(f"{band_name} Topography", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("Topomap generation failed:", e)
        return str(TOPO_PLACEHOLDER) if Path(TOPO_PLACEHOLDER).exists() else None

# ----------------- Normative comparison bar chart (Theta/Alpha & Alpha Asymmetry) -----------------
def plot_norm_comparison_bar(metric_key: str, patient_value: float, title: Optional[str]=None) -> Optional[bytes]:
    """
    Returns PNG bytes of a vertical bar with healthy (white) zone and pathological (red) zone shown.
    """
    if not HAS_MATPLOTLIB:
        return None
    rng = NORM_RANGES.get(metric_key, None)
    fig, ax = plt.subplots(figsize=(3.8,2.2), dpi=120)
    try:
        # background ranges
        if rng:
            healthy_low = rng.get("healthy_low", 0.0); healthy_high = rng.get("healthy_high", 1.0)
            at_low = rng.get("at_risk_low", healthy_high); at_high = rng.get("at_risk_high", healthy_high*1.5)
            # draw healthy zone as light rectangle centered
            ax.bar(0, healthy_high, width=0.6, bottom=healthy_low, color='white', edgecolor='gray', alpha=0.8)
            ax.bar(0, at_high, width=0.6, bottom=at_low, color='red', alpha=0.25)
        # patient bar
        color = '#0b63d6' if rng is None or (rng and patient_value <= rng.get("healthy_high", 1.0)) else 'red'
        ax.bar(0, patient_value, width=0.4, color=color)
        ax.set_xlim(-0.8, 0.8)
        # y limits smart
        if rng:
            ymin = min(healthy_low, patient_value) - 0.2*abs(patient_value if patient_value!=0 else 1)
            ymax = max(at_high, patient_value) + 0.2*abs(patient_value if patient_value!=0 else 1)
            ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_ylabel(metric_key, fontsize=8)
        if title:
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("plot_norm_comparison_bar error:", e)
        plt.close(fig)
        return None

# End of Part 1/3
# ----------------- Part 2/3: Streamlit UI, Forms, Processing Entry Points -----------------

# --- Helper: safe image display wrapper (use new 'width' arg) ---
def st_image_bytes(img_bytes: bytes, caption: Optional[str] = None, use_stretch: bool = True):
    try:
        if img_bytes is None:
            return
        if use_stretch:
            st.image(img_bytes, caption=caption, width="stretch")
        else:
            st.image(img_bytes, caption=caption, width="content")
    except Exception:
        try:
            st.image(img_bytes, caption=caption)
        except Exception:
            pass

# --- Sidebar & Header ---
col_main, col_logo = st.columns([4,1])
with col_main:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical EEG Assistant</h2><div class='kv'>QEEG • Connectivity • XAI • Tumor screening</div></div>", unsafe_allow_html=True)
with col_logo:
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width="stretch")
        except Exception:
            st.image(str(LOGO_PATH))

with st.sidebar:
    st.header("Settings & Patient")
    lang_choice = st.selectbox("Language / اللغة", options=["English", "Arabic"], index=0)
    lang = "en" if lang_choice == "English" else "ar"
    st.markdown("---")
    st.subheader("Patient information")
    patient_name = st.text_input("Name / الاسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1900,1,1), max_value=date.today())
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))
    st.markdown("---")
    st.subheader("Clinical & Labs")
    lab_options = ["Vitamin B12","Thyroid (TSH)","Vitamin D","Folate","Homocysteine","HbA1C","Cholesterol"]
    selected_labs = st.multiselect("Available lab results", options=lab_options)
    lab_notes = st.text_area("Notes / lab values (optional)")
    meds = st.text_area("Current medications (name + dose)")
    conditions = st.text_area("Comorbid conditions (e.g., diabetes, hypertension)")
    st.markdown("---")
    st.write(f"Backends: mne={HAS_MNE} mne_conn={HAS_MNE_CONN} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

# --- Upload EDF files ---
st.markdown("## 1) Upload EEG (.edf) files — multi-upload supported")
uploads = st.file_uploader("Drag & drop EDF files", type=["edf"], accept_multiple_files=True)

# --- PHQ-9 (corrected Q3,Q5,Q8 options) ---
st.markdown("## 2) PHQ-9 (Depression screening)")
PHQ_EN = [
 "Little interest or pleasure in doing things",
 "Feeling down, depressed, or hopeless",
 "Sleep changes (select below)",
 "Feeling tired or having little energy",
 "Appetite changes (select below)",
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
for i in range(1,10):
    if lang == "ar":
        qlabel = reshape_ar(PHQ_AR[i-1])
        st.markdown(f"**{qlabel}**")
    else:
        qlabel = f"Q{i}. {PHQ_EN[i-1]}"
    # custom options for Q3, Q5, Q8
    if i == 3:
        opts = ["0 — Not at all","1 — Insomnia (difficulty falling/staying asleep)","2 — Sleeping less","3 — Sleeping more"]
        opts_ar = [reshape_ar(x) for x in ["0 — لا","1 — أرق (صعوبة في النوم)","2 — قلة النوم","3 — زيادة النوم"]]
        choices = opts_ar if lang=="ar" else opts
    elif i == 5:
        opts = ["0 — Not at all","1 — Eating less","2 — Eating more","3 — Both/variable"]
        opts_ar = [reshape_ar(x) for x in ["0 — لا","1 — قلة الأكل","2 — زيادة الأكل","3 — متغير"]]
        choices = opts_ar if lang=="ar" else opts
    elif i == 8:
        opts = ["0 — Not at all","1 — Moving/speaking slowly","2 — Fidgety/restless","3 — Both/variable"]
        opts_ar = [reshape_ar(x) for x in ["0 — لا","1 — تباطؤ في الحركة/التكلم","2 — تململ/قلق","3 — متغير"]]
        choices = opts_ar if lang=="ar" else opts
    else:
        opts = ["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"]
        opts_ar = [reshape_ar(x) for x in ["0 — لا شيء","1 — عدة أيام","2 — أكثر من نصف الأيام","3 — تقريباً كل يوم"]]
        choices = opts_ar if lang=="ar" else opts

    key = f"phq_{i}_{lang}"
    if lang == "ar":
        sel = st.radio("", options=choices, key=key, horizontal=True)
    else:
        sel = st.radio(qlabel, options=choices, key=key, horizontal=True)
    # parse numeric prefix
    try:
        val = int(str(sel).split("—")[0].strip())
    except Exception:
        s = str(sel).strip()
        val = int(s[0]) if s and s[0].isdigit() else 0
    phq_answers[f"Q{i}"] = val

phq_total = sum(phq_answers.values())
st.info(f"PHQ-9 total: {phq_total} (0–27)")

# --- AD8 (corrected) ---
st.markdown("## 3) AD8 (Cognitive screening)")
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
    if lang == "ar":
        st.markdown(f"**{reshape_ar(AD8_AR[i-1])}**")
        choice = st.radio("", options=[0,1], key=f"ad8_{i}_ar", horizontal=True)
    else:
        label = f"A{i}. {txt}"
        choice = st.radio(label, options=[0,1], key=f"ad8_{i}_en", horizontal=True)
    ad8_answers[f"A{i}"] = int(choice)

ad8_total = sum(ad8_answers.values())
st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

# --- Processing options ---
st.markdown("---")
st.header("Processing Options")
use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
do_topomap = st.checkbox("Generate topography maps (incl. Gamma)", value=True)
do_connectivity = st.checkbox("Compute connectivity (Coherence / PLI / wPLI)", value=True)
run_models = st.checkbox("Run ML models if provided (model pickles)", value=False)

# --- Process button ---
results = []
if uploads and st.button("Process uploaded EDF(s)"):
    processing_placeholder = st.empty()
    for up in uploads:
        processing_placeholder.info(f"Processing {up.name} ...")
        try:
            tmpfile = save_tmp_upload(up)
            edf = read_edf(tmpfile)
            data = edf.get("data"); sf = edf.get("sfreq") or DEFAULT_SF; ch_names = edf.get("ch_names")
            st.success(f"Loaded {up.name}: backend={edf.get('backend')} channels={data.shape[0]} sfreq={sf}")
            # preprocess
            cleaned = preprocess_data(data, sf, do_notch=use_notch)
            # PSD and bands
            dfbands = compute_psd_bands(cleaned, sf)
            agg = aggregate_bands(dfbands, ch_names=ch_names)
            # focal delta
            focal = compute_focal_delta_index(dfbands, ch_names=ch_names)
            # topomaps
            topo_imgs = {}
            if do_topomap:
                for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                    try:
                        vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(cleaned.shape[0])
                    except Exception:
                        vals = np.zeros(cleaned.shape[0])
                    img = generate_topomap_image(vals, ch_names=ch_names, band_name=band)
                    topo_imgs[band] = img
            # connectivity
            conn_mat = None; conn_narr = None; conn_img = None
            if do_connectivity:
                conn_mat, conn_narr = compute_connectivity_matrix(cleaned, sf, ch_names=ch_names, band=BANDS.get("Alpha",(8.0,13.0)))
                if conn_mat is not None and HAS_MATPLOTLIB:
                    try:
                        fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
                        im = ax.imshow(conn_mat, cmap='viridis'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title("Connectivity (Alpha)")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                    except Exception as e:
                        print("conn image failed:", e)
            # ===== Store EEG and clinical metrics for PDF report =====
st.session_state["theta_alpha_ratio"] = theta_alpha_ratio if "theta_alpha_ratio" in locals() else None
st.session_state["alpha_asymmetry"] = alpha_asymmetry if "alpha_asymmetry" in locals() else None
st.session_state["mean_connectivity"] = mean_connectivity if "mean_connectivity" in locals() else None
st.session_state["focal_delta_index"] = focal_delta_index if "focal_delta_index" in locals() else None
st.session_state["focal_delta_ratio"] = focal_delta_ratio if "focal_delta_ratio" in locals() else None
st.session_state["mean_gamma"] = mean_gamma_power if "mean_gamma_power" in locals() else None
st.session_state["ml_score"] = ml_risk_score if "ml_risk_score" in locals() else None

# store patient info for report
st.session_state["patient_name"] = patient_name
st.session_state["patient_age"] = patient_age
st.session_state["patient_gender"] = patient_gender
st.session_state["patient_meds"] = selected_medications
st.session_state["patient_conditions"] = selected_conditions

            processing_placeholder.success(f"Processed {up.name}")
        except Exception as e:
            processing_placeholder.error(f"Failed processing {up.name}: {e}")
            _trace(e)
    # show brief table for first file
    if results:
        st.markdown("### Aggregated features (first file)")
        try:
            st.write(pd.Series(results[0]["agg_features"]))
        except Exception:
            st.write(results[0]["agg_features"])
else:
    if not uploads:
        st.info("Upload EDF(s) to enable processing.")

# --- If results exist, show visualizations and XAI ---
if results:
    st.markdown("---")
    st.header("Results Overview (First file)")

    r0 = results[0]
    agg0 = r0["agg_features"]
    focal0 = r0["focal"]

    # Executive ML Risk (heuristic if no models)
    # normalized inputs
    ta = agg0.get("theta_alpha_ratio", 1.0)
    phq_norm = phq_total/27.0
    ad8_norm = ad8_total/8.0
    ta_norm = min(1.0, ta/1.4)
    ml_risk = min(100.0, (ta_norm*0.55 + phq_norm*0.3 + ad8_norm*0.15)*100.0)
    st.metric(label="Final ML Risk Score", value=f"{ml_risk:.1f}%")

    # QEEG key metrics table
    st.subheader("QEEG Key Metrics")
    st.table(pd.DataFrame([{
        "Theta/Alpha Ratio": agg0.get("theta_alpha_ratio",0),
        "Theta/Beta Ratio": agg0.get("theta_beta_ratio",0),
        "Alpha mean (rel)": agg0.get("alpha_rel_mean",0),
        "Theta mean (rel)": agg0.get("theta_rel_mean",0),
        "Alpha Asymmetry (F3-F4)": agg0.get("alpha_asym_F3_F4",0)
    }]).T.rename(columns={0:"Value"}))

    # Normative comparison bars
    st.subheader("Normative Comparison")
    ta_img = plot_norm_comparison_bar("theta_alpha_ratio", agg0.get("theta_alpha_ratio",0), title="Theta/Alpha vs Norm")
    asym_img = plot_norm_comparison_bar("alpha_asym_F3_F4", agg0.get("alpha_asym_F3_F4",0), title="Alpha Asymmetry (F3-F4)")
    col1, col2 = st.columns(2)
    with col1:
        if ta_img:
            st_image_bytes(ta_img, caption="Theta/Alpha comparison", use_stretch=True)
    with col2:
        if asym_img:
            st_image_bytes(asym_img, caption="Alpha Asymmetry", use_stretch=True)

    # Show focal delta alerts
    st.subheader("Focal Delta / Tumor indicators")
    if focal0 and focal0.get("focal_alerts"):
        for alert in focal0["focal_alerts"]:
            if "channel" in alert:
                st.warning(f"Focal Delta Alert — {alert['channel']} : FDI={alert['fdi']:.2f}")
            else:
                st.warning(f"Extreme Asymmetry — {alert.get('pair')} : ratio={alert.get('ratio')}")
    else:
        st.success("No focal delta alerts detected.")

    # Show Topomaps (Gamma included)
    st.subheader("Topography Maps (first file)")
    topo_imgs = r0.get("topo_images", {})
    if topo_imgs:
        cols = st.columns(5)
        for i, (band, img) in enumerate(topo_imgs.items()):
            try:
                if isinstance(img, (bytes, bytearray)):
                    cols[i].image(img, caption=f"{band} topomap", width="stretch")
                elif isinstance(img, str) and Path(img).exists():
                    cols[i].image(str(img), caption=f"{band} topomap (placeholder)", width="stretch")
            except Exception:
                pass

    # Connectivity matrix & functional disconnection
    st.subheader("Functional Connectivity")
    if r0.get("connectivity_image"):
        st.image(r0.get("connectivity_image"), caption="Connectivity (Alpha) Map", width="stretch")
    elif r0.get("connectivity_matrix") is not None:
        st.write("Connectivity matrix computed — numeric matrix available.")
        # compute mean connectivity and flag functional disconnection if low
        mean_conn = float(np.nanmean(r0["connectivity_matrix"]))
        st.write(f"Mean connectivity (alpha): {mean_conn:.3f}")
        if mean_conn < 0.15:
            st.warning("Functional disconnection suspected (mean connectivity low). Correlate clinically.")
        else:
            st.success("Connectivity within expected range.")
    else:
        st.info("Connectivity not available (mne_connectivity not installed or computation failed).")

    # XAI (SHAP)
    st.subheader("Explainable AI (XAI)")
    shap_data = None
    try:
        if SHAP_JSON.exists():
            with open(SHAP_JSON, "r", encoding="utf-8") as f:
                shap_data = json.load(f)
        if shap_data:
            # choose key by heuristic
            model_key = "depression_global"
            if agg0.get("theta_alpha_ratio",0) > 1.3:
                model_key = "alzheimers_global"
            features = shap_data.get(model_key, {})
            if features:
                st.write("Top contributors (SHAP):")
                # convert to series sorted absolute importance
                s = pd.Series(features).abs().sort_values(ascending=False)
                # show top 10
                st.bar_chart(s.head(10), use_container_width=True)
            else:
                st.info("SHAP file present but no matching model key.")
        else:
            st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
    except Exception as e:
        st.warning(f"XAI load error: {e}")

    # Downloadable simple CSV of metrics
    st.markdown("---")
    st.subheader("Export")
    try:
        df_export = pd.DataFrame([res["agg_features"] for res in results])
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass
    # Generate complete bilingual PDF report
    try:
        if results:
            summary_data = results[0]["agg_features"]
            ml_score = summary_data.get("final_ml_risk_score", 0)
            pdf_bytes = generate_pdf_report(
                summary=summary_data,
                lang="en",
                amiri_path="fonts/Amiri-Regular.ttf",
                output_path=f"NeuroEarly_Report_{now_ts()}.pdf"
            )
            st.download_button(
                "📄 Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"NeuroEarly_Report_{now_ts()}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("No processed EEG data found. Please upload and process EEG first.")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

# End of Part 2/3
# ----------------- Part 3/3: PDF Report Generator, Footer, and Main Run -----------------
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import io

def generate_pdf_report(summary, lang="en", amiri_path=None, output_path=None):
    """Generate a bilingual clinical PDF report with EEG metrics and visuals."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from datetime import datetime
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=40, bottomMargin=40, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()

    # فونت و استایل سفارشی
    styles.add(ParagraphStyle(
        name="Title",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0077B6"),  # آبی روشن
        alignment=1,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.black
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0096C7"),
        spaceBefore=12,
        spaceAfter=6
    ))

    # ---- محتوا ----
    elements = []
    elements.append(Paragraph("NeuroEarly EEG Analysis Report", styles["Title"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Body"]))
    elements.append(Spacer(1, 12))

    # خلاصه‌ی بالینی
    elements.append(Paragraph("Clinical Summary", styles["SectionHeader"]))
    ml_score = summary.get("final_ml_risk_score", 0)
    risk_label = "Low" if ml_score < 0.3 else "Moderate" if ml_score < 0.7 else "High"
    elements.append(Paragraph(f"Final ML Risk Score: <b>{ml_score:.2f}</b> ({risk_label} risk of abnormal EEG pattern)", styles["Body"]))
    elements.append(Spacer(1, 10))

    # شاخص‌های کلیدی
    elements.append(Paragraph("EEG Quantitative Metrics", styles["SectionHeader"]))
    metrics_table = [
        ["Feature", "Value", "Clinical Note"],
        ["Theta/Alpha Ratio", f"{summary.get('theta_alpha_ratio', 'N/A'):.2f}", "High values may suggest cognitive slowing."],
        ["Alpha Asymmetry", f"{summary.get('alpha_asymmetry', 'N/A'):.2f}", "Asymmetry linked to mood disorders."],
        ["Focal Delta Index", f"{summary.get('focal_delta_index', 'N/A'):.2f}", "Elevated values indicate focal slowing or tumor lesion."],
        ["Gamma Power", f"{summary.get('gamma_power', 'N/A'):.2f}", "Reduced gamma may indicate disconnection."],
        ["Functional Connectivity", f"{summary.get('mean_connectivity', 'N/A'):.2f}", "Reflects inter-regional neural coherence."]
    ]
    table = Table(metrics_table, colWidths=[150, 100, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#CAF0F8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.gray),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 15))

    # تفسیر بالینی نهایی
    elements.append(Paragraph("Clinical Interpretation", styles["SectionHeader"]))
    interpretation = []
    if summary.get("focal_delta_index", 0) > 2.0:
        interpretation.append("⚠️ Focal Delta activity suggests possible localized cortical lesion or tumor focus.")
    if summary.get("theta_alpha_ratio", 0) > 1.3:
        interpretation.append("🧠 Theta/Alpha ratio indicates global slowing, consistent with early Alzheimer’s signs.")
    if summary.get("alpha_asymmetry", 0) > 0.5:
        interpretation.append("⚠️ Marked Alpha Asymmetry may indicate depressive or emotional dysregulation.")
    if summary.get("mean_connectivity", 0) < 0.3:
        interpretation.append("🔴 Functional disconnection observed, reduced inter-hemispheric coherence.")
    if not interpretation:
        interpretation.append("✅ EEG patterns are within normal clinical range.")
    for line in interpretation:
        elements.append(Paragraph(line, styles["Body"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Report generated by Golden Bird LLC | NeuroEarly System</b>", styles["Body"]))

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_data)
    return pdf_data
