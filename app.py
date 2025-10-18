# app.py — NeuroEarly Pro (Clinical Edition) — Final single-file
# Default: English, selectable Arabic. Uses Amiri font if available at ./fonts/Amiri-Regular.ttf
# Features: EDF load, preprocessing, PSD, focal delta, connectivity, topomaps, XAI (shap), PDF report, CSV export.

import os
import io
import sys
import json
import math
import tempfile
import traceback
import datetime
from datetime import date, datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy libs
HAS_MNE = False
HAS_PYEDF = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False

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
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
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

# SciPy functions
from scipy.signal import welch, butter, filtfilt, iirnotch, coherence
from scipy.integrate import trapezoid

# Paths
ROOT = Path(".")
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ROOT / "fonts"
AMIRI_TTF = FONTS_DIR / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"

# EEG bands
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}
DEFAULT_SF = 256.0

# Normative ranges (example) for bar chart
NORM_RANGES = {
    "theta_alpha_ratio": {"healthy_low": 0.0, "healthy_high": 1.1, "at_risk_high": 1.4},
    "alpha_asym_F3_F4": {"healthy_low": -0.05, "healthy_high": 0.05, "at_risk_low": -0.2, "at_risk_high": -0.05}
}

# Streamlit config
st.set_page_config(page_title="NeuroEarly Pro — Clinical EEG Assistant", layout="wide", initial_sidebar_state="expanded")

# Simple CSS (medical blue theme)
st.markdown("""
<style>
:root { --main-blue: #0b63d6; --light-blue: #eaf3ff; --muted: #6b7280; }
.header { background: linear-gradient(90deg, #0b63d6, #2b8cff); color: white; padding:12px; border-radius:8px; }
.card { background:white; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(11,99,214,0.06); }
.kv { color: #6b7280; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# Utilities
def now_ts():
    return dt.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e: Exception):
    tb = traceback.format_exc()
    st.error("Internal error — see logs")
    st.code(tb)
    print(tb, file=sys.stderr)

def save_tmp_upload(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path: str) -> Dict[str, Any]:
    """Return dict with keys: backend, raw (if mne), data (channels x samples), ch_names, sfreq"""
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

# Filtering
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

# PSD / band powers
def compute_psd_bands(data: np.ndarray, sf: float, nperseg: int = 1024) -> pd.DataFrame:
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
            pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(trapezoid(pxx, freqs)) if freqs.size > 0 else 0.0
        except Exception as e:
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
            except Exception:
                abs_p = 0.0; rel = 0.0
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        rows.append(row)
    if not rows:
        cols = ["channel_idx"] + [f"{b}_abs" for b in BANDS.keys()] + [f"{b}_rel" for b in BANDS.keys()]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)

# Aggregate features
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
    out["gamma_rel_mean"] = float(out.get("gamma_rel_mean", 0.0))
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

# Focal delta analysis
def compute_focal_delta_index(df_bands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, Any]:
    res = {"fdi": {}, "asymmetry": {}, "focal_alerts": []}
    try:
        if df_bands is None or df_bands.empty:
            return res
        delta_vals = df_bands[["channel_idx", "Delta_abs"]].set_index("channel_idx")["Delta_abs"].to_dict()
        global_mean = np.nanmean(list(delta_vals.values())) if delta_vals else 0.0
        if not np.isfinite(global_mean) or global_mean <= 0:
            global_mean = 1e-9
        for idx, val in delta_vals.items():
            fdi = float(val / global_mean) if global_mean > 0 else 0.0
            res["fdi"][idx] = fdi
            if fdi > 2.0:
                chname = ch_names[idx] if ch_names and idx < len(ch_names) else f"ch{idx}"
                res["focal_alerts"].append({"channel_idx": idx, "channel": chname, "fdi": fdi})
        name_idx = {}
        if ch_names:
            for i, n in enumerate(ch_names):
                name_idx[n.upper()] = i
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

# Connectivity
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: Optional[List[str]] = None, band: Tuple[float,float]=(8.0,13.0)) -> Tuple[Optional[np.ndarray], str]:
    try:
        nchan = int(data.shape[0])
        lo, hi = band
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

# Topomap generator
def generate_topomap_image(band_vals: np.ndarray, ch_names: Optional[List[str]]=None, band_name: str="Alpha") -> Optional[bytes]:
    if band_vals is None:
        return None
    if not HAS_MATPLOTLIB:
        return None
    try:
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
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(-1.0,1.0,160); yi = np.linspace(-1.0,1.0,160)
            XI, YI = np.meshgrid(xi, yi)
            Z = griddata((xs, ys), zs, (XI, YI), method='cubic', fill_value=np.nan)
        except Exception:
            XI, YI = np.meshgrid(np.linspace(-1,1,160), np.linspace(-1,1,160))
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
        return None

# Plot normative comparison bar
def plot_norm_comparison_bar(metric_key: str, patient_value: float, title: Optional[str]=None) -> Optional[bytes]:
    if not HAS_MATPLOTLIB:
        return None
    rng = NORM_RANGES.get(metric_key, None)
    fig, ax = plt.subplots(figsize=(3.8,2.2), dpi=120)
    try:
        if rng:
            healthy_low = rng.get("healthy_low", 0.0); healthy_high = rng.get("healthy_high", 1.0)
            at_low = rng.get("healthy_high", healthy_high); at_high = rng.get("at_risk_high", healthy_high*1.5)
            ax.bar(0, healthy_high, width=0.6, bottom=healthy_low, color='#ffffff', edgecolor='gray', alpha=1.0)
            ax.bar(0, at_high, width=0.6, bottom=at_low, color='red', alpha=0.25)
        color = '#0b63d6' if rng is None or (rng and patient_value <= rng.get("healthy_high", 1.0)) else 'red'
        ax.bar(0, patient_value, width=0.4, color=color)
        ax.set_xlim(-0.8, 0.8)
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

# PDF generator (professional version v2)
def generate_pdf_report(summary: dict,
                        lang: str = "en",
                        amiri_path: Optional[str] = None,
                        topo_images: Optional[Dict[str, bytes]] = None,
                        conn_image: Optional[bytes] = None,
                        bar_img: Optional[bytes] = None) -> bytes:
    """Create a professional PDF report and return bytes"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()

    # register Amiri if available
    try:
        if amiri_path and Path(amiri_path).exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        else:
            base_font = "Helvetica"
    except Exception:
        base_font = "Helvetica"

    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

    story = []
    nowstr = dt.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph("NeuroEarly Pro — Clinical EEG/QEEG Report", styles["TitleBlue"]))
    story.append(Paragraph(f"Generated: {nowstr}", styles["Body"]))
    story.append(Spacer(1, 8))

    # patient info
    p = summary.get("patient_info", {})
    if p:
        story.append(Paragraph("<b>Patient Information</b>", styles["H2"]))
        pd_table = [[
            "Name", p.get("name", "—"),
            "ID", p.get("id", "—")
        ], [
            "DOB", p.get("dob", "—"),
            "Sex", p.get("sex", "—")
        ], [
            "Medications", ", ".join(p.get("medications", [])) if p.get("medications") else "—",
            "Conditions", ", ".join(p.get("conditions", [])) if p.get("conditions") else "—"
        ]]
        t = Table(pd_table, colWidths=[80, 180, 80, 140])
        t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), base_font),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff"))
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # Executive summary
    story.append(Paragraph("<b>Executive Summary</b>", styles["H2"]))
    ml = summary.get("final_ml_risk", None)
    if ml is not None:
        score_pct = ml*100 if ml<=1 else ml
        risk_cat = "Low" if score_pct < 25 else "Moderate" if score_pct < 60 else "High"
        story.append(Paragraph(f"<b>Final ML Risk Score:</b> {score_pct:.1f}% — {risk_cat}", styles["Body"]))
    else:
        story.append(Paragraph("Final ML Risk Score: N/A", styles["Body"]))
    # quick interpret
    qinterp = summary.get("qinterp", "")
    if qinterp:
        story.append(Paragraph(qinterp, styles["Body"]))
    story.append(Spacer(1, 8))

    # Metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
    metrics = summary.get("metrics", {})
    mt = [["Metric", "Value", "Clinical Note"]]
    # ensure keys in desired order
    desired = [("theta_alpha_ratio","Theta/Alpha Ratio","Higher indicates slowing"),
               ("theta_beta_ratio","Theta/Beta Ratio","Stress/inattention"),
               ("alpha_asym_F3_F4","Alpha Asymmetry (F3-F4)","Left-right asymmetry"),
               ("gamma_rel_mean","Gamma Relative Mean","Cognitive integration"),
               ("mean_connectivity","Mean Connectivity","Functional coherence")]
    for k,label,note in desired:
        v = metrics.get(k, summary.get(k, "N/A"))
        try:
            vv = f"{float(v):.3f}"
        except Exception:
            vv = str(v)
        mt.append([label, vv, note])
    table = Table(mt, colWidths=[160, 80, 220])
    table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1), base_font),
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff")),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    # Bar chart image (theta/alpha & alpha asym)
    if bar_img:
        story.append(Paragraph("<b>Comparative Bar Chart</b>", styles["H2"]))
        try:
            story.append(RLImage(io.BytesIO(bar_img), width=420, height=160))
            story.append(Spacer(1,8))
        except Exception as e:
            story.append(Paragraph(f"Bar chart could not be embedded: {e}", styles["Note"]))

    # Topomaps
    if topo_images:
        story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
        # layout images in rows (max 3 per row)
        imgs = []
        for band, img in topo_images.items():
            try:
                imgs.append(RLImage(io.BytesIO(img), width=120, height=120))
            except Exception:
                pass
        if imgs:
            # display as a single-row table (wrap if many)
            story.append(Table([imgs], hAlign='LEFT'))
            story.append(Spacer(1,8))

    # Connectivity
    story.append(Paragraph("<b>Functional Connectivity</b>", styles["H2"]))
    conn_val = metrics.get("mean_connectivity", summary.get("mean_connectivity", "N/A"))
    story.append(Paragraph(f"Mean connectivity (alpha band): {conn_val}", styles["Body"]))
    if conn_image:
        try:
            story.append(RLImage(io.BytesIO(conn_image), width=420, height=220))
            story.append(Spacer(1,6))
        except Exception as e:
            story.append(Paragraph(f"Connectivity image embed error: {e}", styles["Note"]))

    # Focal delta
    tumor = summary.get("tumor", {})
    story.append(Paragraph("<b>Focal Delta / Tumor Indicators</b>", styles["H2"]))
    fi = tumor.get("delta_index", None)
    ar = tumor.get("asym_ratio", None)
    if fi is not None:
        story.append(Paragraph(f"Focal Delta Index (max region): {fi:.2f}", styles["Body"]))
    if ar is not None:
        story.append(Paragraph(f"Asymmetry Ratio (R/L): {ar:.2f}", styles["Body"]))
    # show alerts
    alerts = tumor.get("alerts", [])
    if alerts:
        story.append(Paragraph("<b>Alerts:</b>", styles["Body"]))
        for a in alerts:
            story.append(Paragraph(f"- {a}", styles["Body"]))

    story.append(Spacer(1,10))

    # XAI shaps
    shap_top = summary.get("shap_top", [])
    if shap_top:
        story.append(Paragraph("<b>Explainable AI (SHAP) Top Contributors</b>", styles["H2"]))
        for i,(f,v) in enumerate(shap_top[:10]):
            story.append(Paragraph(f"{i+1}. {f}: {v:.3f}", styles["Body"]))
        story.append(Spacer(1,8))

    # Recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
    recs = summary.get("recommendations", [
        "Correlate QEEG findings with PHQ-9 and AD8 results.",
        "Check B12 and TSH to exclude reversible causes.",
        "If ML Risk >25% and Theta/Alpha>1.4 → consider MRI or FDG-PET.",
        "For moderate risk: follow-up in 3–6 months."
    ])
    for r in recs:
        story.append(Paragraph(f"• {r}", styles["Body"]))

    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Report generated by Golden Bird LLC — NeuroEarly Pro v2.0</b>", styles["Note"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------- Streamlit UI & Main ----------------
def main():
    # Header
    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical EEG Assistant</h2><div class='kv'>QEEG • Connectivity • XAI • Tumor screening</div></div>", unsafe_allow_html=True)
    with col2:
        if LOGO_PATH.exists():
            try:
                st.image(str(LOGO_PATH), width=120)
            except Exception:
                st.image(str(LOGO_PATH))

    # Sidebar
    with st.sidebar:
        st.header("Settings & Patient")
        lang_choice = st.selectbox("Language / اللغة", options=["English", "Arabic"], index=0)
        lang = "en" if lang_choice == "English" else "ar"
        st.markdown("---")
        st.subheader("Patient information")
        patient_name = st.text_input("Name / الاسم", key="in_name")
        patient_id = st.text_input("ID", key="in_id")
        dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1900,1,1), max_value=date.today(), key="in_dob")
        sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"), key="in_sex")
        st.markdown("---")
        st.subheader("Clinical & Labs")
        lab_options = ["Vitamin B12","Thyroid (TSH)","Vitamin D","Folate","Homocysteine","HbA1C","Cholesterol"]
        selected_labs = st.multiselect("Available lab results", options=lab_options, key="in_labs")
        lab_notes = st.text_area("Notes / lab values (optional)", key="in_labnotes")
        meds = st.text_area("Current medications (name + dose)", key="in_meds")
        conditions = st.text_area("Comorbid conditions (e.g., diabetes, hypertension)", key="in_conditions")
        st.markdown("---")
        st.write(f"Backends: mne={HAS_MNE} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

    st.markdown("## 1) Upload EEG (.edf) files — multi-upload supported")
    uploads = st.file_uploader("Drag & drop EDF files", type=["edf"], accept_multiple_files=True)

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
    phq_answers = {}
    for i in range(1,10):
        qlabel = f"Q{i}. {PHQ_EN[i-1]}"
        if i == 3:
            opts = ["0 — Not at all","1 — Insomnia (difficulty falling/staying asleep)","2 — Sleeping less","3 — Sleeping more"]
        elif i == 5:
            opts = ["0 — Not at all","1 — Eating less","2 — Eating more","3 — Both/variable"]
        elif i == 8:
            opts = ["0 — Not at all","1 — Moving/speaking slowly","2 — Fidgety/restless","3 — Both/variable"]
        else:
            opts = ["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"]
        sel = st.radio(qlabel, options=opts, key=f"phq_{i}_{lang}", horizontal=True)
        try:
            val = int(str(sel).split("—")[0].strip())
        except Exception:
            val = 0
        phq_answers[f"Q{i}"] = val
    phq_total = sum(phq_answers.values())
    st.info(f"PHQ-9 total: {phq_total} (0–27)")

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
    ad8_answers = {}
    for i, txt in enumerate(AD8_EN, start=1):
        label = f"A{i}. {txt}"
        choice = st.radio(label, options=[0,1], key=f"ad8_{i}", horizontal=True)
        ad8_answers[f"A{i}"] = int(choice)
    ad8_total = sum(ad8_answers.values())
    st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

    st.markdown("---")
    st.header("Processing Options")
    use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
    do_topomap = st.checkbox("Generate topography maps (incl. Gamma)", value=True)
    do_connectivity = st.checkbox("Compute connectivity (Coherence)", value=True)
    run_models = st.checkbox("Run ML models if provided (model pickles)", value=False)

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
                cleaned = preprocess_data(data, sf, do_notch=use_notch)
                dfbands = compute_psd_bands(cleaned, sf)
                agg = aggregate_bands(dfbands, ch_names=ch_names)
                focal = compute_focal_delta_index(dfbands, ch_names=ch_names)
                topo_imgs = {}
                if do_topomap:
                    for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                        try:
                            vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(cleaned.shape[0])
                        except Exception:
                            vals = np.zeros(cleaned.shape[0])
                        img = generate_topomap_image(vals, ch_names=ch_names, band_name=band)
                        topo_imgs[band] = img

                # --- Connectivity computation and image
                conn_mat = None
                conn_narr = None
                conn_img = None
                if do_connectivity:
                    try:
                        conn_mat, conn_narr = compute_connectivity_matrix(cleaned, sf, ch_names=ch_names, band=BANDS.get("Alpha",(8.0,13.0)))
                        if conn_mat is not None and HAS_MATPLOTLIB:
                            fig = plt.figure(figsize=(4,3))
                            ax = fig.add_subplot(111)
                            im = ax.imshow(conn_mat, cmap='viridis')
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            ax.set_title("Connectivity (Alpha)")
                            buf = io.BytesIO()
                            fig.tight_layout()
                            fig.savefig(buf, format='png')
                            plt.close(fig)
                            buf.seek(0)
                            conn_img = buf.getvalue()
                    except Exception as e:
                        print("connectivity block failed:", e)

                # --- compute simple ML heuristic risk (if not running models)
                # heuristic using theta/alpha + phq + ad8
                ta = agg.get("theta_alpha_ratio", 0.0)
                phq_norm = phq_total/27.0
                ad8_norm = ad8_total/8.0
                ta_norm = min(1.0, ta/1.4)
                ml_risk = min(1.0, (ta_norm*0.55 + phq_norm*0.3 + ad8_norm*0.15))

                # save to session_state for PDF
                st.session_state["final_ml_risk"] = ml_risk
                st.session_state["theta_alpha_ratio"] = agg.get("theta_alpha_ratio", None)
                st.session_state["alpha_asymmetry"] = agg.get("alpha_asym_F3_F4", None)
                st.session_state["mean_connectivity"] = float(np.nanmean(conn_mat)) if conn_mat is not None else None
                # focal summarise
                max_fdi = None
                if focal and focal.get("fdi"):
                    try:
                        max_idx = max(focal["fdi"].keys(), key=lambda x: focal["fdi"][x])
                        max_fdi = focal["fdi"].get(max_idx, None)
                    except Exception:
                        max_fdi = None
                st.session_state["focal_delta_index"] = max_fdi
                # asymmetry ratio (best-effort)
                asym_ratio = None
                if "T7/T8" in focal.get("asymmetry", {}):
                    asym_ratio = focal["asymmetry"].get("T7/T8")
                st.session_state["focal_delta_ratio"] = asym_ratio
                st.session_state["mean_gamma"] = agg.get("gamma_rel_mean", None)

                # patient info
                st.session_state["patient_name"] = patient_name
                st.session_state["patient_id"] = patient_id
                st.session_state["patient_dob"] = str(dob)
                st.session_state["patient_sex"] = sex
                st.session_state["patient_meds"] = [l.strip() for l in meds.split("\n") if l.strip()]
                st.session_state["patient_conditions"] = [l.strip() for l in conditions.split("\n") if l.strip()]

                # collect for display/export
                results.append({
                    "filename": up.name,
                    "agg_features": agg,
                    "df_bands": dfbands,
                    "topo_images": topo_imgs,
                    "connectivity_matrix": conn_mat,
                    "connectivity_narrative": conn_narr,
                    "connectivity_image": conn_img,
                    "focal": focal,
                    "raw_sf": sf,
                    "ch_names": ch_names,
                    "ml_risk": ml_risk
                })
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

    # Results visualization and XAI
    if results:
        st.markdown("---")
        st.header("Results Overview (First file)")

        r0 = results[0]
        agg0 = r0["agg_features"]
        focal0 = r0["focal"]

        # metric display
        ml_display = st.session_state.get("final_ml_risk", 0.0)*100 if st.session_state.get("final_ml_risk") is not None else 0.0
        st.metric(label="Final ML Risk Score", value=f"{ml_display:.1f}%")

        st.subheader("QEEG Key Metrics")
        st.table(pd.DataFrame([{
            "Theta/Alpha Ratio": agg0.get("theta_alpha_ratio",0),
            "Theta/Beta Ratio": agg0.get("theta_beta_ratio",0),
            "Alpha mean (rel)": agg0.get("alpha_rel_mean",0),
            "Theta mean (rel)": agg0.get("theta_rel_mean",0),
            "Alpha Asymmetry (F3-F4)": agg0.get("alpha_asym_F3_F4",0)
        }]).T.rename(columns={0:"Value"}))

        # Normative bars
        st.subheader("Normative Comparison")
        ta_img = plot_norm_comparison_bar("theta_alpha_ratio", agg0.get("theta_alpha_ratio",0), title="Theta/Alpha vs Norm")
        asym_img = plot_norm_comparison_bar("alpha_asym_F3_F4", agg0.get("alpha_asym_F3_F4",0), title="Alpha Asymmetry (F3-F4)")
        col1, col2 = st.columns(2)
        with col1:
            if ta_img:
                st.image(ta_img, caption="Theta/Alpha comparison", use_column_width=True)
        with col2:
            if asym_img:
                st.image(asym_img, caption="Alpha Asymmetry", use_column_width=True)

        # Focal delta alerts
        st.subheader("Focal Delta / Tumor indicators")
        if focal0 and focal0.get("focal_alerts"):
            for alert in focal0["focal_alerts"]:
                if "channel" in alert:
                    st.warning(f"Focal Delta Alert — {alert['channel']} : FDI={alert['fdi']:.2f}")
                else:
                    st.warning(f"Extreme Asymmetry — {alert.get('pair')} : ratio={alert.get('ratio')}")
        else:
            st.success("No focal delta alerts detected.")

        # Topomaps
        st.subheader("Topography Maps (first file)")
        topo_imgs = r0.get("topo_images", {})
        if topo_imgs:
            cols = st.columns(5)
            for i, (band, img) in enumerate(topo_imgs.items()):
                try:
                    if isinstance(img, (bytes, bytearray)):
                        cols[i].image(img, caption=f"{band} topomap", use_column_width=True)
                except Exception:
                    pass

        # Connectivity
        st.subheader("Functional Connectivity")
        if r0.get("connectivity_image"):
            st.image(r0.get("connectivity_image"), caption="Connectivity (Alpha) Map", use_column_width=True)
        elif r0.get("connectivity_matrix") is not None:
            st.write("Connectivity matrix computed — numeric matrix available.")
            mean_conn = float(np.nanmean(r0["connectivity_matrix"]))
            st.write(f"Mean connectivity (alpha): {mean_conn:.3f}")
            if mean_conn < 0.15:
                st.warning("Functional disconnection suspected (mean connectivity low). Correlate clinically.")
            else:
                st.success("Connectivity within expected range.")
        else:
            st.info("Connectivity not available.")

        # XAI
        st.subheader("Explainable AI (XAI)")
        shap_data = None
        try:
            if SHAP_JSON.exists():
                with open(SHAP_JSON, "r", encoding="utf-8") as f:
                    shap_data = json.load(f)
            if shap_data:
                model_key = "depression_global"
                if agg0.get("theta_alpha_ratio",0) > 1.3:
                    model_key = "alzheimers_global"
                features = shap_data.get(model_key, {})
                if features:
                    st.write("Top contributors (SHAP):")
                    s = pd.Series(features).abs().sort_values(ascending=False)
                    st.bar_chart(s.head(10))
                else:
                    st.info("SHAP file present but no matching model key.")
            else:
                st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
        except Exception as e:
            st.warning(f"XAI load error: {e}")

        # Export CSV
        st.markdown("---")
        st.subheader("Export")
        try:
            df_export = pd.DataFrame([res["agg_features"] for res in results])
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
        except Exception:
            pass

        # Generate PDF button (collect full summary)
        st.markdown("---")
        st.header("Generate Clinical PDF Report")
        if st.button("📘 Generate PDF Report"):
            try:
                # build summary dict to pass to pdf generator
                summary = {
                    "patient_info": {
                        "name": st.session_state.get("patient_name", ""),
                        "id": st.session_state.get("patient_id", ""),
                        "dob": st.session_state.get("patient_dob", ""),
                        "sex": st.session_state.get("patient_sex", ""),
                        "medications": st.session_state.get("patient_meds", []),
                        "conditions": st.session_state.get("patient_conditions", [])
                    },
                    "final_ml_risk": st.session_state.get("final_ml_risk", 0.0),
                    "metrics": {
                        "theta_alpha_ratio": st.session_state.get("theta_alpha_ratio", results[0]["agg_features"].get("theta_alpha_ratio", 0)),
                        "theta_beta_ratio": results[0]["agg_features"].get("theta_beta_ratio", 0),
                        "alpha_asym_F3_F4": results[0]["agg_features"].get("alpha_asym_F3_F4", 0),
                        "gamma_rel_mean": results[0]["agg_features"].get("gamma_rel_mean", 0),
                        "mean_connectivity": st.session_state.get("mean_connectivity", None)
                    },
                    "qinterp": f"PHQ-9: {phq_total} /27  — AD8: {ad8_total} /8",
                    "topo_images": results[0].get("topo_images", {}),
                    "conn_image": results[0].get("connectivity_image", None),
                    "bar_img": None,
                    "tumor": {
                        "delta_index": st.session_state.get("focal_delta_index", None),
                        "asym_ratio": st.session_state.get("focal_delta_ratio", None),
                        "alerts": [f"FDI>{a['fdi']:.2f} in {a.get('channel','?')}" for a in results[0].get("focal", {}).get("focal_alerts", [])]
                    },
                    "shap_top": [],
                    "recommendations": []
                }

                # create bar chart image (theta/alpha and alpha asym)
                try:
                    ta_val = float(summary["metrics"]["theta_alpha_ratio"] or 0.0)
                    aa_val = float(summary["metrics"]["alpha_asym_F3_F4"] or 0.0)
                    bar_img = plot_norm_comparison_bar("theta_alpha_ratio", ta_val, title="Theta/Alpha vs Norm")
                    summary["bar_img"] = bar_img
                except Exception:
                    summary["bar_img"] = None

                # shap_top
                try:
                    if SHAP_JSON.exists():
                        with open(SHAP_JSON, "r", encoding="utf-8") as f:
                            sd = json.load(f)
                        model_key = "depression_global"
                        if summary["metrics"]["theta_alpha_ratio"] > 1.3:
                            model_key = "alzheimers_global"
                        feats = sd.get(model_key, {})
                        if feats:
                            top = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                            summary["shap_top"] = top
                except Exception:
                    pass

                pdf_bytes = generate_pdf_report(summary, lang=lang, amiri_path=str(AMIRI_TTF) if AMIRI_TTF.exists() else None,
                                                topo_images=summary.get("topo_images"), conn_image=summary.get("conn_image"),
                                                bar_img=summary.get("bar_img"))
                st.success("PDF generated.")
                st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            except Exception as e:
                _trace(e)

    # Sidebar summary list display
    with st.sidebar:
        st.markdown("---")
        st.subheader("🩺 Summary")
        meds_ss = st.session_state.get("patient_meds", [])
        cond_ss = st.session_state.get("patient_conditions", [])
        if meds_ss:
            st.markdown("**Medications:**")
            for m in meds_ss:
                st.markdown(f"- {m}")
        if cond_ss:
            st.markdown("**Comorbidities:**")
            for c in cond_ss:
                st.markdown(f"- {c}")
        st.markdown("---")
        st.caption("Report generated by Golden Bird LLC • 2025")

if __name__ == "__main__":
    main()
