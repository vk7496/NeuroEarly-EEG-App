# app.py — NeuroEarly Pro (Final Professional Version)
# Default language: English (Arabic selectable)
# Requires: streamlit, numpy, pandas, scipy, matplotlib, mne or pyedflib, reportlab, arabic-reshaper, python-bidi, shap (optional)

import os
import io
import sys
import json
import math
import tempfile
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ----------------- Optional heavy libs (graceful fallback) -----------------
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_RESHAPER = False
HAS_BIDI = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    import pyedflib
    HAS_PYEDFLIB = True
except Exception:
    HAS_PYEDFLIB = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

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

try:
    import scipy.signal as sps
    from scipy.signal import iirnotch, filtfilt, butter, welch, coherence, hilbert
except Exception:
    sps = None

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_RESHAPER = True
    HAS_BIDI = True
except Exception:
    HAS_ARABIC_RESHAPER = False
    HAS_BIDI = False

# ----------------- Paths / constants -----------------
ROOT = Path(".")
AMIRI_FONT_PATH = Path("Amiri-Regular.ttf")  # you confirmed path is correct
SHAP_JSON = Path("shap_summary.json")
LOGO_PATH = Path("GoldenBird_logo.png")  # optional
BLUE_HEX = "#0b63d6"
DEFAULT_SF = 256.0

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Normative example ranges for bar comparison (tweakable)
NORM_RANGES = {
    "theta_alpha_ratio": {"healthy_low": 0.0, "healthy_high": 1.1, "at_risk_high": 1.4},
    "alpha_asym_F3_F4": {"healthy_low": -0.05, "healthy_high": 0.05, "at_risk_low": -0.2, "at_risk_high": -0.05}
}

# Streamlit page config
st.set_page_config(page_title="NeuroEarly Pro — Clinical EEG", layout="wide", initial_sidebar_state="expanded")

# Simple CSS
st.markdown(f"""
<style>
.header {{ background: linear-gradient(90deg, {BLUE_HEX}, #2b8cff); color: white; padding:12px; border-radius:8px; }}
.card {{ background:white; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(11,99,214,0.06); }}
.small {{ font-size:12px; color:#666; }}
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e: Exception):
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    st.error("Internal error — check logs")
    st.code(tb)

def safe_plot_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def reshape_ar(text: str) -> str:
    if HAS_ARABIC_RESHAPER and HAS_BIDI:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

# ----------------- EDF reading (mne preferred, fallback to pyedflib) -----------------
def read_edf(path: str) -> Dict[str, Any]:
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data()
            chs = raw.ch_names
            sf = raw.info.get("sfreq", DEFAULT_SF)
            return {"backend": "mne", "raw": raw, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            print("mne read failed:", e)
    if HAS_PYEDF:
        try:
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            chs = f.getSignalLabels()
            sf = int(f.getSampleFrequencies()[0]) if hasattr(f, "getSampleFrequencies") else DEFAULT_SF
            sigs = []
            for i in range(n):
                try:
                    s = f.readSignal(i).astype(np.float64)
                except Exception:
                    s = np.zeros(1)
                sigs.append(s)
            f._close()
            data = np.vstack(sigs)
            return {"backend": "pyedflib", "raw": None, "data": data, "ch_names": chs, "sfreq": sf}
        except Exception as e:
            raise IOError(f"pyedflib failed: {e}")
    raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ----------------- Filtering & Preprocessing -----------------
def notch_filter(sig: np.ndarray, sf: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    try:
        b, a = iirnotch(freq, Q, sf)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig: np.ndarray, sf: float, low: float = 0.5, high: float = 45.0, order: int = 4) -> np.ndarray:
    try:
        nyq = 0.5*sf
        low_n = max(low/nyq, 1e-6)
        high_n = min(high/nyq, 0.9999)
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess(raw_data: np.ndarray, sf: float, do_notch: bool=True) -> np.ndarray:
    cleaned = np.zeros_like(raw_data, dtype=np.float64)
    for i in range(raw_data.shape[0]):
        s = raw_data[i].astype(np.float64)
        if do_notch:
            s = notch_filter(s, sf)
        s = bandpass_filter(s, sf)
        cleaned[i, :] = s
    return cleaned

# ----------------- PSD / Band power -----------------
def compute_psd_bands(data: np.ndarray, sf: float, nperseg: int = 1024) -> pd.DataFrame:
    # returns DataFrame rows per channel with columns: Delta_abs, Delta_rel, Theta_abs, Theta_rel, ...
    rows = []
    nch = int(data.shape[0]) if data is not None and data.ndim >= 2 else 0
    for ch in range(nch):
        sig = data[ch]
        try:
            freqs, pxx = welch(sig, fs=sf, nperseg=min(nperseg, max(256, len(sig))))
            total = float(np.trapz(pxx, freqs)) if freqs.size>0 else 0.0
        except Exception:
            freqs = np.array([]); pxx = np.array([]); total = 0.0
        row = {"channel_idx": ch}
        for band,(lo,hi) in BANDS.items():
            try:
                if freqs.size == 0:
                    abs_p = 0.0
                else:
                    mask = (freqs>=lo)&(freqs<=hi)
                    seg_freqs = freqs[mask]
                    seg_pxx = pxx[mask]
                    abs_p = float(np.trapz(seg_pxx, seg_freqs)) if seg_freqs.size>0 else 0.0
                rel = float(abs_p/total) if total>0 else 0.0
            except Exception:
                abs_p = 0.0; rel = 0.0
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        rows.append(row)
    if not rows:
        cols = ["channel_idx"] + [f"{b}_abs" for b in BANDS.keys()] + [f"{b}_rel" for b in BANDS.keys()]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)

# ----------------- Aggregate features -----------------
def aggregate_features(dfbands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str,float]:
    out = {}
    if dfbands is None or dfbands.empty:
        for band in BANDS:
            out[f"{band.lower()}_rel_mean"] = 0.0
        out["theta_alpha_ratio"] = 0.0
        out["alpha_asym_F3_F4"] = 0.0
        out["gamma_rel_mean"] = 0.0
        return out
    for band in BANDS:
        try:
            out[f"{band.lower()}_rel_mean"] = float(np.nanmean(dfbands[f"{band}_rel"].values))
        except Exception:
            out[f"{band.lower()}_rel_mean"] = 0.0
    alpha = out.get("alpha_rel_mean", 1e-9)
    beta = out.get("beta_rel_mean", 1e-9)
    theta = out.get("theta_rel_mean", 0.0)
    out["theta_alpha_ratio"] = float(theta/alpha) if alpha>0 else 0.0
    out["theta_beta_ratio"] = float(theta/beta) if beta>0 else 0.0
    out["beta_alpha_ratio"] = float(beta/alpha) if alpha>0 else 0.0
    out["gamma_rel_mean"] = float(out.get("gamma_rel_mean",0.0))
    out["alpha_asym_F3_F4"] = 0.0
    if ch_names is not None:
        try:
            names = [n.upper() for n in ch_names]
            def find_idx(tok):
                for i,nm in enumerate(names):
                    if tok in nm: return i
                return None
            i3 = find_idx("F3"); i4 = find_idx("F4")
            if i3 is not None and i4 is not None:
                a3 = dfbands.loc[dfbands['channel_idx']==i3, 'Alpha_rel']
                a4 = dfbands.loc[dfbands['channel_idx']==i4, 'Alpha_rel']
                if not a3.empty and not a4.empty:
                    out["alpha_asym_F3_F4"] = float(a3.values[0] - a4.values[0])
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# ----------------- Focal Delta Index (Tumor indicator) -----------------
def compute_focal_delta_index(dfbands: pd.DataFrame, ch_names: Optional[List[str]]=None) -> Dict[str,Any]:
    res = {"fdi":{}, "asymmetry":{}, "alerts":[]}
    try:
        if dfbands is None or dfbands.empty:
            return res
        delta = dfbands.set_index("channel_idx")["Delta_abs"].to_dict()
        global_mean = np.nanmean(list(delta.values())) if delta else 0.0
        if not np.isfinite(global_mean) or global_mean<=0: global_mean = 1e-9
        for idx,val in delta.items():
            fdi = float(val/global_mean)
            res["fdi"][idx] = fdi
            if fdi>2.0:
                ch = ch_names[idx] if ch_names and idx<len(ch_names) else f"ch{idx}"
                res["alerts"].append({"channel_idx":idx,"channel":ch,"fdi":fdi})
        # pairwise asymmetry checks
        pairs = [("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        name_idx = {}
        if ch_names:
            for i,n in enumerate(ch_names):
                name_idx[n.upper()] = i
        for L,R in pairs:
            lidx = None; ridx=None
            for nm,idx in name_idx.items():
                if L in nm: lidx = idx
                if R in nm: ridx = idx
            if lidx is not None and ridx is not None:
                dl = delta.get(lidx,0.0); dr = delta.get(ridx,0.0)
                ratio = float(dr/(dl+1e-9)) if dl>0 else float("inf") if dr>0 else 1.0
                res["asymmetry"][f"{L}/{R}"] = ratio
                if (isinstance(ratio, float) and (ratio>3.0 or ratio<0.33)) or (ratio==float("inf")):
                    res["alerts"].append({"pair":f"{L}/{R}","ratio":ratio})
    except Exception as e:
        print("compute_focal_delta_index error:", e)
    return res

# ----------------- Connectivity (Coherence) -----------------
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: Optional[List[str]]=None, band: Tuple[float,float]=(8.0,13.0)) -> Tuple[Optional[np.ndarray], str]:
    try:
        nchan = int(data.shape[0])
        lo, hi = band
        # If MNE and spectral_connectivity available, use it
        if HAS_MNE:
            try:
                info = mne.create_info(ch_names, sf, ch_types="eeg")
                raw = mne.io.RawArray(data, info)
                raw.filter(l_freq=lo, h_freq=hi, verbose=False)
                con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity([raw], method='coh', sfreq=sf, fmin=lo, fmax=hi, faverage=True, verbose=False)
                conn = np.squeeze(con.get_data(output='dense')) if hasattr(con, "get_data") else np.squeeze(con)
                return conn, f"Coherence {lo}-{hi}Hz (MNE)"
            except Exception as e:
                print("mne connectivity failed:", e)
        # fallback: compute mean coherence per pair using scipy.signal.coherence
        mat = np.zeros((nchan,nchan))
        for i in range(nchan):
            for j in range(i,nchan):
                try:
                    f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                    mask = (f>=lo)&(f<=hi)
                    val = float(np.nanmean(Cxy[mask])) if mask.sum()>0 else 0.0
                except Exception:
                    val = 0.0
                mat[i,j]=val; mat[j,i]=val
        return mat, f"Coherence {lo}-{hi}Hz (scipy)"
    except Exception as e:
        print("compute_connectivity_matrix error:", e)
        return None, "Connectivity failed"

# ----------------- Topomap (approximate) -----------------
def generate_topomap_image(vals: np.ndarray, ch_names: Optional[List[str]] = None, band_name: str="Alpha") -> Optional[bytes]:
    if not HAS_MATPLOTLIB:
        return None
    try:
        n = len(vals)
        side = int(np.ceil(np.sqrt(n)))
        grid = np.zeros((side, side))
        flat = np.array(vals).flatten()
        grid.flat[:len(flat)] = flat
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap='RdBu_r', origin='lower', aspect='auto')
        ax.set_title(f"{band_name} Topography")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return safe_plot_to_bytes(fig)
    except Exception as e:
        print("generate_topomap_image error:", e)
        return None

# ----------------- Normative Bar Chart -----------------
def plot_norm_comparison_bar(metric_key: str, patient_value: float, title: str="") -> Optional[bytes]:
    if not HAS_MATPLOTLIB:
        return None
    try:
        rng = NORM_RANGES.get(metric_key, None)
        fig, ax = plt.subplots(figsize=(4.5,2.2))
        # draw background healthy and risk zones
        if rng:
            healthy_low = rng.get("healthy_low", 0.0); healthy_high = rng.get("healthy_high", 1.0)
            at_low = rng.get("healthy_high", healthy_high); at_high = rng.get("at_risk_high", healthy_high*1.5)
            ax.bar(0, healthy_high-healthy_low, bottom=healthy_low, width=0.6, color='white', edgecolor='gray')
            ax.bar(0, at_high-at_low, bottom=at_low, width=0.6, color='red', alpha=0.25)
        color = BLUE_HEX if rng is None or (rng and patient_value <= rng.get("healthy_high", 1.0)) else 'red'
        ax.bar(0, patient_value, width=0.4, color=color)
        ax.set_xlim(-0.8,0.8)
        ax.set_xticks([])
        if title:
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        return safe_plot_to_bytes(fig)
    except Exception as e:
        print("plot_norm_comparison_bar error:", e)
        return None

# ----------------- SHAP plot image -----------------
def shap_bar_image(shap_dict: Dict[str,float], top_n: int=10) -> Optional[bytes]:
    if not HAS_MATPLOTLIB or not shap_dict:
        return None
    try:
        items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        labels = [i[0] for i in items][::-1]
        values = [i[1] for i in items][::-1]
        fig, ax = plt.subplots(figsize=(5,2.5))
        ax.barh(labels, values, color=BLUE_HEX)
        ax.set_xlabel("SHAP value (impact)")
        fig.tight_layout()
        return safe_plot_to_bytes(fig)
    except Exception as e:
        print("shap_bar_image error:", e)
        return None

# ----------------- PDF generation (bilingual) -----------------
def generate_pdf_report(summary: dict,
                        lang: str = "en",
                        amiri_path: Optional[str] = None,
                        topo_images: Optional[Dict[str, bytes]] = None,
                        conn_image: Optional[bytes] = None,
                        bar_img: Optional[bytes] = None,
                        shap_img: Optional[bytes] = None) -> bytes:
    """Create a bilingual professional PDF and return bytes"""
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")
    if amiri_path is None:
        amiri_path = str(AMIRI_FONT_PATH) if AMIRI_FONT_PATH.exists() else None
    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri registration failed:", e)
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE_HEX), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE_HEX), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        nowstr = datetime.now().strftime("%Y-%m-%d %H:%M")
        # header
        if lang == "ar" and HAS_ARABIC_RESHAPER and HAS_BIDI:
            story.append(Paragraph(reshape_ar("تقرير NeuroEarly الاحترافي"), styles["TitleBlue"]))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("NeuroEarly Pro — Clinical EEG/QEEG Report", styles["TitleBlue"]))
            story.append(Paragraph(f"Generated: {nowstr}", styles["Note"]))
            story.append(Spacer(1, 6))
        # Patient info
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
            t = Table(pd_table, colWidths=[80,160,80,140])
            t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),base_font),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff"))]))
            story.append(t)
            story.append(Spacer(1,8))
        # Executive summary + ML score
        story.append(Paragraph("<b>Executive Summary</b>", styles["H2"]))
        ml = summary.get("final_ml_risk", None)
        if ml is not None:
            score_pct = ml*100 if isinstance(ml, float) and ml<=1 else ml
            try:
                score_str = f"{float(score_pct):.1f}%"
            except Exception:
                score_str = str(score_pct)
            risk_cat = "Low" if float(score_pct)<25 else "Moderate" if float(score_pct)<60 else "High"
            story.append(Paragraph(f"<b>Final ML Risk Score:</b> {score_str} — {risk_cat}", styles["Body"]))
        else:
            story.append(Paragraph("Final ML Risk Score: N/A", styles["Body"]))
        qinterp = summary.get("qinterp", "")
        if qinterp:
            story.append(Paragraph(qinterp, styles["Body"]))
        story.append(Spacer(1,8))
        # Key metrics table
        story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
        metrics = summary.get("metrics", {})
        mt = [["Metric","Value","Clinical Note"]]
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
        table = Table(mt, colWidths=[160,80,220])
        table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),base_font),("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),9)]))
        story.append(table)
        story.append(Spacer(1,10))
        # Bar chart
        if bar_img:
            story.append(Paragraph("<b>Comparative Power Ratios</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(bar_img), width=420, height=160))
            except Exception as e:
                story.append(Paragraph(f"Bar chart embed error: {e}", styles["Note"]))
            story.append(Spacer(1,8))
        # Topomaps
        if topo_images:
            story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
            imgs = []
            for band,img in topo_images.items():
                try:
                    imgs.append(RLImage(io.BytesIO(img), width=120, height=120))
                except Exception:
                    pass
            if imgs:
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
        # Focal delta / tumor
        tumor = summary.get("tumor", {})
        story.append(Paragraph("<b>Focal Delta / Tumor Indicators</b>", styles["H2"]))
        fi = tumor.get("delta_index", None)
        ar = tumor.get("asym_ratio", None)
        if fi is not None:
            story.append(Paragraph(f"Focal Delta Index (max region): {fi}", styles["Body"]))
        if ar is not None:
            story.append(Paragraph(f"Asymmetry Ratio (R/L): {ar}", styles["Body"]))
        alerts = tumor.get("alerts", [])
        if alerts:
            story.append(Paragraph("<b>Alerts:</b>", styles["Body"]))
            for a in alerts:
                story.append(Paragraph(f"- {a}", styles["Body"]))
        story.append(Spacer(1,8))
        # SHAP
        if shap_img:
            story.append(Paragraph("<b>Explainable AI (SHAP) — Top Contributors</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(shap_img), width=420, height=160))
                story.append(Spacer(1,6))
            except Exception as e:
                story.append(Paragraph(f"SHAP image embed error: {e}", styles["Note"]))
        elif summary.get("shap_top"):
            story.append(Paragraph("<b>Explainable AI (SHAP) — Top Contributors</b>", styles["H2"]))
            for i,(f,v) in enumerate(summary.get("shap_top",[])[:10]):
                story.append(Paragraph(f"{i+1}. {f}: {v:.3f}", styles["Body"]))
            story.append(Spacer(1,6))
        # Recommendations
        story.append(Spacer(1,10))
        story.append(Paragraph("<b>Expert Recommendations & Model Transparency</b>", styles["H2"]))
        story.append(Paragraph("1) Connectivity Upgrade: Enable functional connectivity metrics (coherence/PLI/wPLI) to elevate to Dynamic Network Biomarkers.", styles["Body"]))
        story.append(Paragraph("2) Explainability: Provide SHAP bar chart visualizations to show which QEEG features drive predictions (theta/alpha, alpha asymmetry, connectivity).", styles["Body"]))
        story.append(Paragraph("3) Research Roadmap: Consider adding Microstate Analysis and non-linear features (Entropy) to improve early AD/MCI sensitivity.", styles["Body"]))
        story.append(Spacer(1,12))
        story.append(Paragraph("<b>Report generated by Golden Bird LLC — NeuroEarly Pro</b>", styles["Note"]))
        # build
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        buffer.close()
        print("PDF generation error:", e)
        traceback.print_exc()
        return b""

# ----------------- Streamlit UI -----------------
def main():
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical EEG Assistant</h2><div class='small'>QEEG • Connectivity • XAI • Tumor screening</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([4,1])
    with col2:
        if LOGO_PATH.exists():
            try:
                st.image(str(LOGO_PATH), width=120)
            except Exception:
                pass

    # Sidebar: language + patient info
    with st.sidebar:
        st.header("Settings & Patient")
        lang_choice = st.selectbox("Language / اللغة", options=["English","Arabic"], index=0)
        lang = "en" if lang_choice=="English" else "ar"
        st.session_state["lang"] = lang

        st.markdown("---")
        st.subheader("Patient information")
        patient_name = st.text_input("Name / الاسم", key="in_name")
        patient_id = st.text_input("ID", key="in_id")
        dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1900,1,1), max_value=date.today(), key="in_dob")
        sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"), key="in_sex")
        st.markdown("---")
        st.subheader("Clinical & Labs")
        lab_options = ["Vitamin B12","TSH","Vitamin D","Folate","Homocysteine","HbA1C","Cholesterol"]
        selected_labs = st.multiselect("Available lab results", options=lab_options, key="in_labs")
        lab_notes = st.text_area("Notes / lab values (optional)", key="in_labnotes")
        meds = st.text_area("Current medications (one per line)", key="in_meds")
        conditions = st.text_area("Comorbid conditions (one per line)", key="in_conditions")
        st.markdown("---")
        st.caption("Golden Bird LLC • NeuroEarly Pro")

    # Main panel: uploads and questionnaires
    st.markdown("## 1) Upload EEG (.edf) — Multi-upload supported")
    uploads = st.file_uploader("Drag & drop EDF files", type=["edf"], accept_multiple_files=True)

    # PHQ-9 with modifications
    st.markdown("## 2) PHQ-9 (Depression screening)")
    PHQ_EN = [
     "Little interest or pleasure in doing things",
     "Feeling down, depressed, or hopeless",
     "Sleep changes (Insomnia / Short sleep / Hypersomnia)",
     "Feeling tired or having little energy",
     "Appetite changes (Eating less / Eating more)",
     "Feeling bad about yourself — or that you are a failure",
     "Trouble concentrating on things, such as reading or watching TV",
     "Moving or speaking slowly OR being fidgety/restless",
     "Thoughts that you would be better off dead or of harming yourself"
    ]
    phq_answers = {}
    for i in range(1,10):
        q = PHQ_EN[i-1]
        if i==3:
            opts = ["0 — Not at all","1 — Insomnia (difficulty falling/staying asleep)","2 — Sleeping less","3 — Sleeping more"]
        elif i==5:
            opts = ["0 — Not at all","1 — Eating less","2 — Eating more","3 — Both/variable"]
        elif i==8:
            opts = ["0 — Not at all","1 — Moving/speaking slowly","2 — Fidgety/restless","3 — Both/variable"]
        else:
            opts = ["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"]
        sel = st.radio(f"Q{i}. {q}", options=opts, key=f"phq_{i}", horizontal=True)
        try:
            val = int(str(sel).split("—")[0].strip())
        except Exception:
            val = 0
        phq_answers[f"Q{i}"] = val
    phq_total = sum(phq_answers.values())
    st.info(f"PHQ-9 total: {phq_total} (0–27)")

    # AD8
    st.markdown("## 3) AD8 (Cognitive screening)")
    AD8_QS = [
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
    for i, txt in enumerate(AD8_QS, start=1):
        choice = st.radio(f"A{i}. {txt}", options=[0,1], key=f"ad8_{i}", horizontal=True)
        ad8_answers[f"A{i}"] = int(choice)
    ad8_total = sum(ad8_answers.values())
    st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

    # Processing options
    st.markdown("---")
    st.header("Processing Options")
    use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
    do_topomap = st.checkbox("Generate topography maps (approx)", value=True)
    do_connectivity = st.checkbox("Compute connectivity (Coherence)", value=True)
    run_models = st.checkbox("Run ML models if provided (joblib pickle)", value=False)

    # results storage
    results = []

    if uploads and st.button("Process uploaded EDF(s)"):
        processing_placeholder = st.empty()
        for up in uploads:
            processing_placeholder.info(f"Processing {up.name} ...")
            try:
                # save tmp
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                tmpfile.write(up.getbuffer())
                tmpfile.flush(); tmpfile.close()
                edf = read_edf(tmpfile.name)
                data = edf.get("data"); sf = edf.get("sfreq") or DEFAULT_SF; ch_names = edf.get("ch_names")
                processing_placeholder.success(f"Loaded {up.name}: backend={edf.get('backend')} channels={data.shape[0]} sfreq={sf}")
                # preprocess
                cleaned = preprocess(data, sf, do_notch=use_notch)
                # psd bands
                dfbands = compute_psd_bands(cleaned, sf)
                agg = aggregate_features(dfbands, ch_names=ch_names)
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
                            conn_img = safe_plot_to_bytes(fig)
                        except Exception as e:
                            print("conn image failed:", e)
                # simple ML heuristic (transparent)
                ta = agg.get("theta_alpha_ratio",0.0)
                phq_norm = phq_total/27.0
                ad8_norm = ad8_total/8.0
                ta_norm = min(1.0, ta/1.4)
                ml_risk = min(1.0, (ta_norm*0.55 + phq_norm*0.3 + ad8_norm*0.15))
                # session state stores
                st.session_state["final_ml_risk"] = ml_risk
                st.session_state["theta_alpha_ratio"] = agg.get("theta_alpha_ratio", None)
                st.session_state["alpha_asymmetry"] = agg.get("alpha_asym_F3_F4", None)
                st.session_state["mean_connectivity"] = float(np.nanmean(conn_mat)) if conn_mat is not None else None
                # focal
                max_fdi = None
                try:
                    if focal and focal.get("fdi"):
                        max_idx = max(focal["fdi"].keys(), key=lambda x: focal["fdi"][x])
                        max_fdi = focal["fdi"].get(max_idx, None)
                except Exception:
                    max_fdi = None
                st.session_state["focal_delta_index"] = max_fdi
                # collect result
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

        if results:
            st.markdown("### Aggregated features (first file)")
            try:
                st.write(pd.Series(results[0]["agg_features"]))
            except Exception:
                st.write(results[0]["agg_features"])
    else:
        if not uploads:
            st.info("Upload EDF(s) to enable processing.")

    # present results
    if results:
        st.markdown("---")
        st.header("Results Overview (First file)")
        r0 = results[0]
        agg0 = r0["agg_features"]
        focal0 = r0["focal"]
        ml_display = st.session_state.get("final_ml_risk", 0.0)*100 if st.session_state.get("final_ml_risk") is not None else 0.0
        st.metric(label="Final ML Risk Score", value=f"{ml_display:.1f}%")
        st.subheader("QEEG Key Metrics")
        try:
            st.table(pd.DataFrame([{
                "Theta/Alpha Ratio": agg0.get("theta_alpha_ratio",0),
                "Theta/Beta Ratio": agg0.get("theta_beta_ratio",0),
                "Alpha mean (rel)": agg0.get("alpha_rel_mean",0),
                "Theta mean (rel)": agg0.get("theta_rel_mean",0),
                "Alpha Asymmetry (F3-F4)": agg0.get("alpha_asym_F3_F4",0)
            }]).T.rename(columns={0:"Value"}))
        except Exception:
            st.write(agg0)
        # normative bars
        st.subheader("Normative Comparison")
        ta_img = plot_norm_comparison_bar("theta_alpha_ratio", agg0.get("theta_alpha_ratio",0), title="Theta/Alpha vs Norm")
        asym_img = plot_norm_comparison_bar("alpha_asym_F3_F4", agg0.get("alpha_asym_F3_F4",0), title="Alpha Asymmetry (F3-F4)")
        c1,c2 = st.columns(2)
        with c1:
            if ta_img:
                st.image(ta_img, caption="Theta/Alpha comparison")
        with c2:
            if asym_img:
                st.image(asym_img, caption="Alpha Asymmetry")
        # focal alerts
        st.subheader("Focal Delta / Tumor indicators")
        if focal0 and focal0.get("alerts"):
            for alert in focal0["alerts"]:
                if "channel" in alert:
                    st.warning(f"Focal Delta Alert — {alert['channel']} : FDI={alert['fdi']:.2f}")
                else:
                    st.warning(f"Extreme Asymmetry — {alert.get('pair')} : ratio={alert.get('ratio')}")
        else:
            st.success("No focal delta alerts detected.")
        # topomaps
        st.subheader("Topography Maps (first file)")
        topo_imgs = r0.get("topo_images", {})
        if topo_imgs:
            cols = st.columns(min(5, len(topo_imgs)))
            for i,(band,img) in enumerate(topo_imgs.items()):
                try:
                    if isinstance(img,(bytes,bytearray)):
                        cols[i].image(img, caption=f"{band} topomap")
                except Exception:
                    pass
        # connectivity
        st.subheader("Functional Connectivity")
        if r0.get("connectivity_image"):
            st.image(r0.get("connectivity_image"), caption="Connectivity (Alpha)")
        elif r0.get("connectivity_matrix") is not None:
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
                    # show as matplotlib horizontal bar for better styling
                    if HAS_MATPLOTLIB:
                        fig,ax = plt.subplots(figsize=(6,3))
                        s.head(10).sort_values().plot.barh(ax=ax, color=BLUE_HEX)
                        ax.set_xlabel("abs SHAP value")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        st.image(buf.getvalue(), caption="SHAP Top contributors")
                    else:
                        st.bar_chart(s.head(10))
                else:
                    st.info("SHAP file present but no matching model key.")
            else:
                st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
        except Exception as e:
            st.warning(f"XAI load error: {e}")
        # export
        st.markdown("---")
        st.subheader("Export")
        try:
            df_export = pd.DataFrame([res["agg_features"] for res in results])
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
        except Exception:
            pass
        # Generate PDF
        st.markdown("---")
        st.header("Generate Clinical PDF Report")
        report_lang = st.selectbox("Report language", options=["English","Arabic"], index=0)
        if st.button("📘 Generate PDF Report"):
            try:
                summary = {
                    "patient_info":{
                        "name": st.session_state.get("patient_name", patient_name),
                        "id": st.session_state.get("patient_id", patient_id),
                        "dob": str(dob),
                        "sex": sex,
                        "medications": [l.strip() for l in meds.split("\n") if l.strip()],
                        "conditions": [l.strip() for l in conditions.split("\n") if l.strip()]
                    },
                    "final_ml_risk": st.session_state.get("final_ml_risk", r0.get("ml_risk",0.0)),
                    "metrics": {
                        "theta_alpha_ratio": st.session_state.get("theta_alpha_ratio", agg0.get("theta_alpha_ratio",0)),
                        "theta_beta_ratio": agg0.get("theta_beta_ratio",0),
                        "alpha_asym_F3_F4": agg0.get("alpha_asym_F3_F4",0),
                        "gamma_rel_mean": agg0.get("gamma_rel_mean",0),
                        "mean_connectivity": st.session_state.get("mean_connectivity", None)
                    },
                    "qinterp": f"PHQ-9: {phq_total} /27  — AD8: {ad8_total} /8",
                    "topo_images": r0.get("topo_images", {}),
                    "conn_image": r0.get("connectivity_image", None),
                    "bar_img": None,
                    "tumor": {
                        "delta_index": st.session_state.get("focal_delta_index", None),
                        "asym_ratio": None,
                        "alerts": [f"FDI>{a['fdi']:.2f} in {a.get('channel','?')}" for a in r0.get("focal", {}).get("alerts", [])]
                    },
                    "shap_top": [],
                    "recommendations": []
                }
                # build bar_img
                try:
                    ta_val = float(summary["metrics"]["theta_alpha_ratio"] or 0.0)
                    aa_val = float(summary["metrics"]["alpha_asym_F3_F4"] or 0.0)
                    bar_img = plot_norm_comparison_bar("theta_alpha_ratio", ta_val, title="Theta/Alpha vs Norm")
                    summary["bar_img"] = bar_img
                except Exception:
                    summary["bar_img"] = None
                # shap image
                shap_img = None
                try:
                    if SHAP_JSON.exists():
                        sd = json.load(open(SHAP_JSON,"r",encoding="utf-8"))
                        model_key = "depression_global" if summary["metrics"]["theta_alpha_ratio"]<=1.3 else "alzheimers_global"
                        feats = sd.get(model_key,{})
                        if feats:
                            top = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                            summary["shap_top"] = top
                            shap_img = shap_bar_image(dict(top))
                except Exception:
                    shap_img = None
                pdf_bytes = generate_pdf_report(summary, lang=("ar" if report_lang=="Arabic" else "en"), amiri_path=str(AMIRI_FONT_PATH) if AMIRI_FONT_PATH.exists() else None, topo_images=summary.get("topo_images"), conn_image=summary.get("conn_image"), bar_img=summary.get("bar_img"), shap_img=shap_img)
                if pdf_bytes:
                    st.success("PDF generated.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation returned empty result.")
            except Exception as e:
                _trace(e)

    # sidebar summary listing medications and conditions
    with st.sidebar:
        st.markdown("---")
        st.subheader("🩺 Summary")
        meds_ss = [l.strip() for l in meds.split("\n") if l.strip()]
        cond_ss = [l.strip() for l in conditions.split("\n") if l.strip()]
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
    try:
        main()
    except Exception as e:
        _trace(e)
