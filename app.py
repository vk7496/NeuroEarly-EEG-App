# app.py — NeuroEarly Pro — Clinical Final (v4)
# Single-file Streamlit app
# Designed for: EEG/QEEG analysis, Depression (PHQ-9) & Alzheimer (AD8) screening, Tumor FDI detection,
# Connectivity, SHAP-based XAI, Topography maps, Professional bilingual PDF (English / Arabic).
#
# Requirements (use requirements.txt previously provided):
# streamlit, numpy, pandas, matplotlib, scipy, mne or pyedflib, reportlab, shap, arabic-reshaper, python-bidi, pillow
#
# Place assets/goldenbird_logo.png and assets/Amiri-Regular.ttf in the repo (or adjust paths below).
# Place optional shap_summary.json in repo root.

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
import matplotlib.pyplot as plt
import streamlit as st

# --------- Optional heavy imports with graceful fallback ----------
HAS_MNE = False
HAS_PYEDF = False
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
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

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

# SciPy functions
try:
    from scipy.signal import welch, coherence, iirnotch, filtfilt, butter
    from scipy import integrate
except Exception:
    welch = None
    coherence = None
    iirnotch = None
    filtfilt = None
    butter = None
    integrate = None

matplotlib.rcParams['font.size'] = 10

# --------- Paths & constants ----------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"  # user-confirmed path
AMIRI_PATH = ASSETS / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

BLUE = "#3FA9F5"            # primary bright blue
DARK_BLUE = "#0b63d6"
LIGHT_BG = "#f7fbff"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Normative example ranges (for bar comparison)
NORM = {
    "theta_alpha": {"healthy": (0.0, 1.1), "at_risk": (1.1, 1.6)},
    "alpha_asym": {"healthy": (-0.05, 0.05), "at_risk": (-0.2, -0.05)}
}

# --------- Helpers ----------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def safe_trace(e: Exception) -> str:
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    return tb

def reshape_ar(text: str) -> str:
    if not text:
        return text
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

# --------- EDF reading ----------
def read_edf_file(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Returns (data ndarray shape (n_channels, n_samples), sampling_freq, ch_names)
    Uses mne if available, otherwise pyedflib.
    """
    path = str(path)
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data()
            sf = float(raw.info.get("sfreq", 256.0))
            chs = raw.ch_names
            return data, sf, chs
        except Exception as e:
            raise RuntimeError(f"mne read error: {e}")
    elif HAS_PYEDF:
        try:
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            chs = f.getSignalLabels()
            sf = float(f.getSampleFrequency(0))
            sigs = []
            for i in range(n):
                s = f.readSignal(i).astype(float)
                sigs.append(s)
            f._close()
            data = np.vstack(sigs)
            return data, sf, chs
        except Exception as e:
            raise RuntimeError(f"pyedflib read error: {e}")
    else:
        raise RuntimeError("No EDF reader available. Install mne or pyedflib.")

# --------- Filtering ----------
def apply_notch(sig: np.ndarray, sf: float, freq=50.0, Q=30.0) -> np.ndarray:
    try:
        b, a = iirnotch(freq, Q, sf)
        out = filtfilt(b, a, sig)
        return out
    except Exception:
        return sig

def bandpass(sig: np.ndarray, sf: float, low=0.5, high=45.0, order=4) -> np.ndarray:
    try:
        nyq = 0.5 * sf
        low_n = max(low / nyq, 1e-8)
        high_n = min(high / nyq, 0.9999)
        b, a = butter(order, [low_n, high_n], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(data: np.ndarray, sf: float, do_notch=True) -> np.ndarray:
    cleaned = np.copy(data).astype(float)
    nchan = cleaned.shape[0]
    for i in range(nchan):
        s = cleaned[i, :]
        if do_notch:
            # try both 50 and 60
            for f0 in (50.0, 60.0):
                try:
                    s = apply_notch(s, sf, freq=f0)
                except Exception:
                    pass
        try:
            s = bandpass(s, sf)
        except Exception:
            pass
        cleaned[i, :] = s
    return cleaned

# --------- PSD & bandpowers ----------
def compute_bandpowers(data: np.ndarray, sf: float, nperseg: int = 2048) -> pd.DataFrame:
    """Return DataFrame with columns like 'Alpha_abs','Alpha_rel' for each channel row."""
    rows = []
    nchan = data.shape[0]
    for i in range(nchan):
        sig = data[i, :]
        try:
            freqs, pxx = welch(sig, fs=sf, nperseg=min(nperseg, len(sig)))
        except Exception:
            freqs = np.array([0.0])
            pxx = np.array([0.0])
        total = np.trapz(pxx, freqs) if freqs.size else 1.0
        r = {}
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum() else 0.0
            rel = abs_p / (total if total > 0 else 1.0)
            r[f"{band}_abs"] = abs_p
            r[f"{band}_rel"] = rel
        rows.append(r)
    df = pd.DataFrame(rows)
    return df

# --------- Aggregation ----------
def aggregate_features(dfbands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, float]:
    out = {}
    if dfbands is None or dfbands.empty:
        for band in BANDS:
            out[f"{band.lower()}_rel_mean"] = 0.0
        out["theta_alpha_ratio"] = 0.0
        out["alpha_asym_F3_F4"] = 0.0
        return out
    for band in BANDS:
        col = f"{band}_rel"
        out[f"{band.lower()}_rel_mean"] = float(dfbands[col].mean()) if col in dfbands.columns else 0.0
    alpha = out.get("alpha_rel_mean", 1e-9)
    theta = out.get("theta_rel_mean", 0.0)
    beta = out.get("beta_rel_mean", 1e-9)
    out["theta_alpha_ratio"] = float(theta / alpha) if alpha > 0 else 0.0
    out["theta_beta_ratio"] = float(theta / beta) if beta > 0 else 0.0
    out["alpha_asym_F3_F4"] = 0.0
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            if "F3" in names and "F4" in names:
                i3 = names.index("F3")
                i4 = names.index("F4")
                a3 = float(dfbands.iloc[i3].get("Alpha_rel", 0.0))
                a4 = float(dfbands.iloc[i4].get("Alpha_rel", 0.0))
                out["alpha_asym_F3_F4"] = float(a3 - a4)
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# --------- Focal Delta Index (FDI) and asymmetry for tumor detection ----------
def compute_focal_delta(dfbands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, Any]:
    res = {"fdi": {}, "alerts": [], "max_idx": None, "max_val": None, "asymmetry": {}}
    try:
        if dfbands is None or dfbands.empty:
            return res
        if "Delta_abs" in dfbands.columns:
            delta = dfbands["Delta_abs"].values
        else:
            delta = np.zeros(dfbands.shape[0])
        global_mean = float(np.nanmean(delta)) if delta.size > 0 else 1e-9
        for idx, val in enumerate(delta):
            fdi = float(val / (global_mean if global_mean > 0 else 1e-9))
            res["fdi"][idx] = fdi
            if fdi > 2.0:
                ch = ch_names[idx] if ch_names and idx < len(ch_names) else f"Ch{idx}"
                res["alerts"].append({"type": "FDI", "channel": ch, "value": fdi})
        pairs = [("T7", "T8"), ("F3", "F4"), ("P3", "P4"), ("O1", "O2"), ("C3", "C4")]
        names_map = {}
        if ch_names:
            for i, n in enumerate(ch_names):
                names_map[n.upper()] = i
        for L, R in pairs:
            if L in names_map and R in names_map:
                lidx = names_map[L]; ridx = names_map[R]
                dl = float(delta[lidx]) if lidx < len(delta) else 0.0
                dr = float(delta[ridx]) if ridx < len(delta) else 0.0
                ratio = float(dr / (dl + 1e-9)) if dl > 0 else float("inf") if dr > 0 else 1.0
                res["asymmetry"][f"{L}/{R}"] = ratio
                if (isinstance(ratio, float) and (ratio > 3.0 or ratio < 0.33)) or (ratio == float("inf")):
                    res["alerts"].append({"type": "asymmetry", "pair": f"{L}/{R}", "ratio": ratio})
        max_idx = int(np.argmax(list(res["fdi"].values()))) if res["fdi"] else None
        max_val = res["fdi"].get(max_idx, None) if max_idx is not None else None
        res["max_idx"] = max_idx; res["max_val"] = max_val
    except Exception as e:
        print("compute_focal_delta err:", e)
    return res

# --------- Connectivity (coherence fallback to scipy) ----------
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: Optional[List[str]] = None, band: Tuple[float, float] = (8.0, 13.0)) -> Tuple[Optional[np.ndarray], str, Optional[bytes], Optional[float]]:
    try:
        nchan = data.shape[0]
    except Exception:
        return None, "(no data)", None, None
    narration = ""
    mat = np.zeros((nchan, nchan))
    mean_conn = 0.0
    if HAS_MNE:
        try:
            info = mne.create_info(ch_names, sf, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity([raw], method='coh', sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
            conn = np.squeeze(con)
            if conn.ndim == 2 and conn.shape[0] == nchan:
                mat = conn
            narration = f"Coherence {band[0]}-{band[1]} Hz (MNE)"
        except Exception as e:
            print("mne connectivity failed:", e)
            narration = "(mne connectivity failed)"
    else:
        if coherence is None:
            narration = "(connectivity not available)"
        else:
            try:
                for i in range(nchan):
                    for j in range(i, nchan):
                        try:
                            f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                            mask = (f >= band[0]) & (f <= band[1])
                            val = float(np.nanmean(Cxy[mask])) if mask.sum() else 0.0
                        except Exception:
                            val = 0.0
                        mat[i, j] = val; mat[j, i] = val
                narration = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
            except Exception as e:
                print("scipy connectivity error:", e)
                narration = "(connectivity computation error)"
    try:
        mean_conn = float(np.nanmean(mat)) if mat.size else 0.0
    except Exception:
        mean_conn = 0.0
    conn_img = None
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        ax.set_title("Connectivity Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        conn_img = buf.getvalue()
    except Exception:
        conn_img = None
    return mat, narration, conn_img, mean_conn

# --------- Topomap plotting (approx on 10-20 positions) ----------
def generate_topomap_image(vals: np.ndarray, ch_names: List[str], band_name: str) -> Optional[bytes]:
    try:
        pos = {
            'Fp1': (-0.5, 1), 'Fp2': (0.5, 1), 'F3': (-0.8, 0.3), 'F4': (0.8, 0.3),
            'C3': (-0.8, -0.3), 'C4': (0.8, -0.3), 'P3': (-0.5, -0.8), 'P4': (0.5, -0.8),
            'O1': (-0.2, -1), 'O2': (0.2, -1), 'F7': (-1, 0.6), 'F8': (1, 0.6), 'T7': (-1, -0.3), 'T8': (1, -0.3)
        }
        xs = []; ys = []; vplot = []
        for i, ch in enumerate(ch_names):
            up = ch.upper()
            if up in pos:
                x, y = pos[up]; xs.append(x); ys.append(y); vplot.append(float(vals[i]))
        if len(xs) >= 3:
            fig, ax = plt.subplots(figsize=(3.2, 2.2))
            sc = ax.scatter(xs, ys, c=vplot, s=260, cmap='RdBu_r')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{band_name} topography")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
        else:
            fig, ax = plt.subplots(figsize=(3.2, 2.2))
            n = min(len(ch_names), len(vals))
            ax.bar(range(n), vals[:n]); ax.set_xticks(range(n)); ax.set_xticklabels(ch_names[:n], rotation=60, fontsize=7)
            ax.set_title(f"{band_name} (bar)")
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
    except Exception as e:
        print("topomap err", e)
        return None

# --------- SHAP helpers ----------
def load_shap_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def shap_bar_image_from_dict(shap_dict: Dict[str, float], top_n=10) -> Optional[bytes]:
    if not shap_dict:
        return None
    try:
        s = pd.Series(shap_dict).abs().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, 2.2))
        s.sort_values().plot.barh(ax=ax, color=BLUE)
        ax.set_xlabel("SHAP (abs impact)")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# --------- PDF generation (ReportLab) ----------
def generate_pdf_report(summary: Dict[str, Any], lang: str = "en", amiri_path: Optional[str] = None, logo_path: Optional[str] = None) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed in environment.")
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
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(DARK_BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        # header with logo on right
        left = Paragraph("NeuroEarly Pro — Clinical QEEG Report", styles["TitleBlue"])
        if logo_path and Path(logo_path).exists():
            try:
                img = RLImage(str(logo_path), width=1.2 * inch, height=1.2 * inch)
                header_table = Table([[left, img]], colWidths=[4.7 * inch, 1.4 * inch])
                header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
                story.append(header_table)
            except Exception:
                story.append(left)
        else:
            story.append(left)
        story.append(Spacer(1, 6))

        # Executive summary
        ml_display = summary.get("final_ml_risk_display", "N/A")
        story.append(Paragraph("<b>Executive Summary</b>", styles["H2"]))
        story.append(Paragraph(f"Final ML Risk Score: <b>{ml_display}</b>", styles["Body"]))
        story.append(Spacer(1, 6))

        # Patient metadata
        pat = summary.get("patient_info", {})
        ptab = [["Field", "Value"], ["ID", pat.get("id", "—")], ["DOB", pat.get("dob", "—")], ["Sex", pat.get("sex", "—")]]
        t = Table(ptab, colWidths=[1.4 * inch, 4.7 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf4ff")), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
        story.append(t)
        story.append(Spacer(1, 8))

        # QEEG Key Metrics
        story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
        metrics = summary.get("metrics", {})
        mt = [["Metric", "Value", "Note"]]
        desired = [
            ("theta_alpha_ratio", "Theta/Alpha Ratio", "Slowing indicator"),
            ("theta_beta_ratio", "Theta/Beta Ratio", "Stress/inattention"),
            ("alpha_asym_F3_F4", "Alpha Asymmetry (F3-F4)", "Left-right asymmetry"),
            ("gamma_rel_mean", "Gamma Relative Mean", "Cognition"),
            ("mean_connectivity", "Mean Connectivity (alpha)", "Functional coherence")
        ]
        for k, label, note in desired:
            v = metrics.get(k, summary.get(k, "N/A"))
            try:
                vv = f"{float(v):.4f}"
            except Exception:
                vv = str(v)
            mt.append([label, vv, note])
        table = Table(mt, colWidths=[2.6 * inch, 1.2 * inch, 2.3 * inch])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf4ff")), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story.append(table)
        story.append(Spacer(1, 8))

        # Bar image for normative comparison
        if summary.get("bar_img"):
            try:
                story.append(Paragraph("<b>Power Ratio Comparison</b>", styles["H2"]))
                story.append(RLImage(io.BytesIO(summary["bar_img"]), width=5.6 * inch, height=1.6 * inch))
                story.append(Spacer(1, 6))
            except Exception:
                pass

        # Topography maps
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
            imgs = []
            for band, buf in summary["topo_images"].items():
                if buf:
                    try:
                        imgs.append(RLImage(io.BytesIO(buf), width=2.4 * inch, height=1.5 * inch))
                    except Exception:
                        pass
            if imgs:
                rows = []
                row = []
                for i, img in enumerate(imgs):
                    row.append(img)
                    if len(row) == 2:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
                for r in rows:
                    story.append(Table([r], colWidths=[3 * inch] * len(r)))
                    story.append(Spacer(1, 4))

        # Connectivity
        if summary.get("conn_image"):
            story.append(Paragraph("<b>Connectivity (Alpha)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["conn_image"]), width=5.6 * inch, height=2.4 * inch))
                story.append(Spacer(1, 6))
            except Exception:
                pass

        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("<b>Explainable AI — SHAP top contributors</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.6 * inch, height=1.8 * inch))
                story.append(Spacer(1, 6))
            except Exception:
                pass
        elif summary.get("shap_table"):
            story.append(Paragraph("<b>Explainable AI — Top contributors</b>", styles["H2"]))
            stbl = [["Feature", "Importance"]]
            for k, v in summary["shap_table"].items():
                try:
                    stbl.append([k, f"{float(v):.4f}"])
                except Exception:
                    stbl.append([k, str(v)])
            t2 = Table(stbl, colWidths=[3.5 * inch, 2 * inch])
            t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
            story.append(t2)
            story.append(Spacer(1, 6))

        # Tumor / focal
        if summary.get("tumor"):
            story.append(Paragraph("<b>Focal Delta / Tumor Indicators</b>", styles["H2"]))
            ttxt = summary["tumor"].get("narrative", "")
            story.append(Paragraph(ttxt, styles["Body"]))
            if summary["tumor"].get("alerts"):
                for a in summary["tumor"]["alerts"]:
                    story.append(Paragraph(f"- {a}", styles["Body"]))
            story.append(Spacer(1, 6))

        # Clinical recommendations
        story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
        for r in summary.get("recommendations", []):
            story.append(Paragraph(r, styles["Body"]))
        story.append(Spacer(1, 12))

        # Footer mention + doctor signature line
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        story.append(Spacer(1, 18))
        story.append(Paragraph("Doctor signature: ___________________________", styles["Body"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        raise

# --------- Streamlit UI ----------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide", initial_sidebar_state="expanded")

# Top header
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; background:linear-gradient(90deg,{BLUE}, #2b8cff); padding:12px; border-radius:8px; color:white;">
  <div style="font-size:20px; font-weight:600;">🧠 NeuroEarly Pro — Clinical AI</div>
  <div style="display:flex; align-items:center;">
    <div style="font-size:12px; margin-right:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:40px;">' if Path(LOGO_PATH).exists() else ''}
  </div>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### Settings")
    lang_choice = st.selectbox("Language / اللغة", options=["English", "Arabic"], index=0)
    is_ar = (lang_choice == "Arabic")
    st.markdown("---")
    st.subheader("Patient")
    patient_id = st.text_input("ID", key="pid")
    patient_dob = st.date_input("DOB", min_value=date(1900, 1, 1), max_value=date.today(), value=date(1980, 1, 1))
    patient_sex = st.selectbox("Sex", ["Unknown", "Male", "Female", "Other"])
    st.markdown("---")
    st.subheader("Clinical / Labs")
    labs = st.multiselect("Relevant labs (select)", options=["B12", "TSH", "Vitamin D", "Folate", "Homocysteine", "HbA1C", "Cholesterol"])
    meds = st.text_area("Current medications (one per line)")
    comorbid = st.text_area("Comorbid conditions (one per line)")
    st.markdown("---")
    st.write("Backends status:")
    st.write(f"mne={HAS_MNE} pyedflib={HAS_PYEDF} reportlab={HAS_REPORTLAB} shap={HAS_SHAP} arabic={HAS_ARABIC}")

with col_main:
    st.markdown("## 1) Upload EDF files")
    uploads = st.file_uploader("Upload EDF (.edf) — multiple allowed", type=["edf"], accept_multiple_files=True)
    st.markdown("## 2) Questionnaires")
    st.markdown("### PHQ-9 (Depression)")
    PHQ_QS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Sleep: choose insomnia / short sleep / hypersomnia",
        "Feeling tired or having little energy",
        "Appetite: overeating or undereating",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking slowly OR being fidgety/restless",
        "Thoughts that you would be better off dead or harming yourself"
    ]
    phq_vals = {}
    for i, q in enumerate(PHQ_QS, start=1):
        if i == 3:
            opts = ["0 — Not at all", "1 — Insomnia (difficulty falling/staying asleep)", "2 — Sleeping less", "3 — Sleeping more"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        elif i == 5:
            opts = ["0 — Not at all", "1 — Eating less", "2 — Eating more", "3 — Both/variable"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        elif i == 8:
            opts = ["0 — Not at all", "1 — Moving/speaking slowly", "2 — Fidgety/restless", "3 — Both/variable"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        else:
            sel = st.radio(f"Q{i}. {q}", options=["0 — Not at all", "1 — Several days", "2 — More than half the days", "3 — Nearly every day"], horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
    phq_total = sum(phq_vals.values())
    st.info(f"PHQ-9 total: {phq_total} (0–27)")

    st.markdown("### AD8 (Cognitive screening)")
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
    ad8_vals = {}
    cols = st.columns(2)
    for i, q in enumerate(AD8_QS, start=1):
        col = cols[(i - 1) % 2]
        sel = col.radio(f"A{i}. {q}", options=[0, 1], key=f"ad8_{i}", horizontal=True)
        ad8_vals[f"A{i}"] = int(sel)
    ad8_total = sum(ad8_vals.values())
    st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

    st.markdown("---")
    st.markdown("### Processing options")
    apply_notch = st.checkbox("Apply notch filter (50/60Hz)", value=True)
    compute_conn = st.checkbox("Compute connectivity (coherence)", value=True)
    generate_topos = st.checkbox("Generate topography maps (approx)", value=True)
    use_shap = st.checkbox("Enable XAI (SHAP) visuals (requires shap_summary.json)", value=True)

    # Process button
    if st.button("Process files"):
        if not uploads:
            st.error("Upload at least one EDF file before processing.")
        else:
            processing = st.empty()
            results = []
            for up in uploads:
                processing.info(f"Processing {up.name} ...")
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                    tmp.write(up.getbuffer()); tmp.flush(); tmp.close()
                    data, sf, ch_names = read_edf_file(tmp.name)  # data shape: (nchan, nsamples)
                    if data is None:
                        processing.error(f"Could not read {up.name}")
                        continue
                    cleaned = preprocess_data(data, sf, do_notch=apply_notch)
                    dfbands = compute_bandpowers(cleaned, sf)
                    try:
                        if len(ch_names) == dfbands.shape[0]:
                            dfbands.index = ch_names
                        else:
                            dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    except Exception:
                        dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    agg = aggregate_features(dfbands, ch_names=ch_names)
                    focal = compute_focal_delta(dfbands, ch_names=ch_names)
                    topo_imgs = {}
                    if generate_topos:
                        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                            try:
                                vals = dfbands[f"{band}_rel"].values if f"{band}_rel" in dfbands.columns else np.zeros(dfbands.shape[0])
                                topo_imgs[band] = generate_topomap_image(vals, ch_names=list(dfbands.index), band_name=band)
                            except Exception:
                                topo_imgs[band] = None
                    conn_mat = None; conn_narr = None; conn_img = None; mean_conn = None
                    if compute_conn:
                        try:
                            conn_mat, conn_narr, conn_img, mean_conn = compute_connectivity_matrix(cleaned, sf, ch_names=list(dfbands.index), band=BANDS.get("Alpha",(8.0,13.0)))
                        except Exception:
                            conn_mat, conn_narr, conn_img, mean_conn = (None, "(connectivity failed)", None, None)
                    # ML heuristic (transparent): combine theta/alpha, PHQ, AD8, mean connectivity (lower -> higher risk)
                    ta = agg.get("theta_alpha_ratio", 0.0)
                    phq_norm = phq_total / 27.0
                    ad8_norm = ad8_total / 8.0
                    mc = float(mean_conn) if mean_conn is not None else 0.0
                    mc_norm = max(0.0, 1.0 - mc)
                    ta_norm = min(1.0, ta / 1.6)
                    ml_risk = min(1.0, (ta_norm * 0.55 + phq_norm * 0.25 + ad8_norm * 0.15 + mc_norm * 0.05))
                    res = {
                        "filename": up.name,
                        "sf": sf,
                        "ch_names": list(dfbands.index),
                        "dfbands": dfbands,
                        "agg": agg,
                        "focal": focal,
                        "topo_imgs": topo_imgs,
                        "connectivity_mat": conn_mat,
                        "connectivity_narr": conn_narr,
                        "connectivity_img": conn_img,
                        "mean_connectivity": mean_conn,
                        "ml_risk": ml_risk
                    }
                    results.append(res)
                    processing.success(f"Processed {up.name}")
                except Exception as e:
                    processing.error(f"Failed processing {up.name}: {e}")
                    st.error(safe_trace(e))
            if results:
                st.markdown("## Results (first file)")
                r0 = results[0]
                st.metric("Final ML Risk Score", f"{r0['ml_risk']*100:.1f}%")
                st.markdown("### QEEG Key Metrics")
                try:
                    st.table(pd.DataFrame([{
                        "Theta/Alpha Ratio": r0["agg"].get("theta_alpha_ratio", 0),
                        "Theta/Beta Ratio": r0["agg"].get("theta_beta_ratio", 0),
                        "Alpha mean (rel)": r0["agg"].get("alpha_rel_mean", 0),
                        "Theta mean (rel)": r0["agg"].get("theta_rel_mean", 0),
                        "Alpha Asymmetry (F3-F4)": r0["agg"].get("alpha_asym_F3_F4", 0)
                    }]).T.rename(columns={0:"Value"}))
                except Exception:
                    st.write(r0["agg"])
                st.markdown("### Normative Comparison")
                try:
                    bar_img = None
                    ta_val = r0["agg"].get("theta_alpha_ratio", 0.0)
                    fig, ax = plt.subplots(figsize=(4.5, 1.8))
                    rng = NORM["theta_alpha"]
                    ax.barh([0], [rng["healthy"][1] - rng["healthy"][0]], left=rng["healthy"][0], height=0.6, color='white', edgecolor='gray')
                    ax.barh([0], [ta_val], left=0, height=0.4, color=BLUE)
                    ax.set_yticks([])
                    ax.set_title("Theta/Alpha comparison")
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); bar_img = buf.getvalue()
                    st.image(bar_img, use_column_width=False)
                except Exception:
                    pass
                st.markdown("### Topography (Alpha/Gamma/…)")
                cols = st.columns(5)
                bands_show = ["Delta","Theta","Alpha","Beta","Gamma"]
                for i, band in enumerate(bands_show):
                    img = r0["topo_imgs"].get(band)
                    if img:
                        cols[i].image(img, caption=band, use_column_width=True)
                st.markdown("### Connectivity")
                if r0.get("connectivity_img"):
                    st.image(r0["connectivity_img"], caption="Connectivity (Alpha)", use_container_width=True)
                else:
                    st.info("Connectivity image not available.")
                st.markdown("### Focal Delta / Tumor Indicators")
                st.json(r0["focal"])
                if use_shap:
                    shap_data = load_shap_summary(SHAP_JSON) if SHAP_JSON.exists() else None
                    if shap_data:
                        model_key = "depression_global" if r0["agg"].get("theta_alpha_ratio", 0) <= 1.3 else "alzheimers_global"
                        features = shap_data.get(model_key, {})
                        if features:
                            st.markdown("### SHAP Top contributors")
                            s = pd.Series(features).abs().sort_values(ascending=False)
                            fig, ax = plt.subplots(figsize=(6, 2.2))
                            s.head(10).sort_values().plot.barh(ax=ax, color=BLUE); ax.set_xlabel("abs SHAP")
                            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                            st.image(buf.getvalue(), caption="SHAP Top contributors", use_container_width=True)
                        else:
                            st.info("SHAP file present but no matching model key.")
                    else:
                        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
            st.session_state["ne_results"] = results

    # Generate PDF
    st.markdown("---")
    st.markdown("## Generate Clinical PDF Report")
    report_lang = st.selectbox("Report language for PDF", options=["English", "Arabic"], index=0)
    if st.button("Generate & Download PDF"):
        try:
            if "ne_results" not in st.session_state or not st.session_state["ne_results"]:
                st.error("No processed results available — process EDF(s) first.")
            else:
                r0 = st.session_state["ne_results"][0]
                summary = {}
                summary["patient_info"] = {"id": patient_id or "—", "dob": str(patient_dob), "sex": patient_sex}
                summary["final_ml_risk_display"] = f"{r0['ml_risk']*100:.1f}%"
                summary["metrics"] = {
                    "theta_alpha_ratio": r0["agg"].get("theta_alpha_ratio", 0.0),
                    "theta_beta_ratio": r0["agg"].get("theta_beta_ratio", 0.0),
                    "alpha_asym_F3_F4": r0["agg"].get("alpha_asym_F3_F4", 0.0),
                    "gamma_rel_mean": r0["agg"].get("gamma_rel_mean", 0.0) if "gamma_rel_mean" in r0["agg"] else 0.0,
                    "mean_connectivity": r0.get("mean_connectivity", None)
                }
                summary["topo_images"] = {k: v for k, v in (r0.get("topo_imgs") or {}).items()}
                summary["conn_image"] = r0.get("connectivity_img")
                shap_table = {}
                shap_img = None
                if SHAP_JSON.exists():
                    try:
                        sd = json.load(open(SHAP_JSON, "r", encoding="utf-8"))
                        model_key = "depression_global" if summary["metrics"]["theta_alpha_ratio"] <= 1.3 else "alzheimers_global"
                        feats = sd.get(model_key, {})
                        if feats:
                            shap_table = feats
                            shap_img = shap_bar_image_from_dict(feats, top_n=10)
                    except Exception:
                        shap_table = {}
                summary["shap_table"] = shap_table
                summary["shap_img"] = shap_img
                summary["tumor"] = {"narrative": f"FDI max {r0['focal'].get('max_val')} at {r0['focal'].get('max_channel')}", "alerts": [str(a) for a in r0['focal'].get('alerts', [])]}
                recs = []
                recs.append("Correlate QEEG findings with clinical exam and PHQ-9/AD8 scores.")
                if summary["metrics"]["theta_alpha_ratio"] > 1.4 or (r0["focal"].get("max_val", 0) and r0["focal"].get("max_val", 0) > 2.0):
                    recs.append("Recommend neuroimaging (MRI) and neurology referral for further evaluation.")
                else:
                    recs.append("Clinical follow-up and repeat EEG in 3-6 months.")
                recs.append("Check reversible causes: B12, TSH, metabolic panel.")
                summary["recommendations"] = recs
                try:
                    fig, ax = plt.subplots(figsize=(5.6, 1.6))
                    ta = summary["metrics"]["theta_alpha_ratio"] or 0.0
                    aa = summary["metrics"]["alpha_asym_F3_F4"] or 0.0
                    ax.bar([0, 1], [ta, aa], width=0.6, color=BLUE)
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["Theta/Alpha", "Alpha Asym (F3-F4)"])
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    summary["bar_img"] = buf.getvalue()
                except Exception:
                    summary["bar_img"] = None
                summary["final_ml_risk_num"] = r0["ml_risk"]
                amiri = str(AMIRI_PATH) if AMIRI_PATH.exists() else None
                pdf_bytes = generate_pdf_report(summary, lang=("ar" if report_lang == "Arabic" else "en"), amiri_path=amiri, logo_path=str(LOGO_PATH))
                if pdf_bytes:
                    st.success("PDF generated.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation returned empty content.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.exception(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("<small>Designed & Prepared by Golden Bird LLC — NeuroEarly Pro</small>", unsafe_allow_html=True)
