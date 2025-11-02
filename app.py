# app.py — NeuroEarly Pro (Final)
# Bilingual Streamlit app (English default, Arabic optional)
# Sidebar left: Language, Patient Info, Labs, Meds, EDF upload, Questionnaires.
# Outputs: Topomaps (5 bands), Connectivity (PLI/coherence), FDI (tumor), SHAP, PDF report.
# Requirements: streamlit, numpy, pandas, matplotlib, scipy, mne or pyedflib, reportlab, shap (optional), arabic-reshaper & python-bidi (optional)

import io
import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Optional heavy libs
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

# scipy helpers
try:
    from scipy.signal import welch, coherence, iirnotch, filtfilt, butter
except Exception:
    welch = None; coherence = None; iirnotch = None; filtfilt = None; butter = None

# constants & paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

BLUE = "#4DB6E2"
DARK_BLUE = "#0b63d6"
LIGHT_BG = "#f7fbff"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

NORM = {
    "theta_alpha": {"healthy": (0.0, 1.1), "at_risk": (1.1, 1.6)},
    "alpha_asym": {"healthy": (-0.05, 0.05), "at_risk": (-0.2, -0.05)}
}

# utils
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

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# EDF reading
def read_edf_file(path: str) -> Tuple[np.ndarray, float, List[str]]:
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
            sigs = [f.readSignal(i).astype(float) for i in range(n)]
            f._close()
            data = np.vstack(sigs)
            return data, sf, chs
        except Exception as e:
            raise RuntimeError(f"pyedflib read error: {e}")
    else:
        raise RuntimeError("No EDF backend available. Install mne or pyedflib.")

# Filtering & preprocessing
def apply_notch(sig: np.ndarray, sf: float, freq=50.0, Q=30.0) -> np.ndarray:
    try:
        b, a = iirnotch(freq, Q, sf)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass(sig: np.ndarray, sf: float, low=0.5, high=45.0, order=4) -> np.ndarray:
    try:
        nyq = 0.5 * sf
        low_n = max(low / nyq, 1e-8)
        high_n = min(high / nyq, 0.9999)
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(data: np.ndarray, sf: float, do_notch=True) -> np.ndarray:
    cleaned = np.copy(data).astype(float)
    nchan = cleaned.shape[0]
    for i in range(nchan):
        s = cleaned[i, :]
        if do_notch:
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

# PSD & bandpowers
def compute_bandpowers(data: np.ndarray, sf: float, nperseg: int = 2048) -> pd.DataFrame:
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

# Aggregation
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
                # careful about dfbands label case
                a3 = dfbands.iloc[i3].get("Alpha_rel", dfbands.iloc[i3].get("alpha_rel", 0.0))
                a4 = dfbands.iloc[i4].get("Alpha_rel", dfbands.iloc[i4].get("alpha_rel", 0.0))
                out["alpha_asym_F3_F4"] = float(a3 - a4)
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# Focal Delta Index (FDI) for tumor indicators
def compute_focal_delta(dfbands: pd.DataFrame, ch_names: Optional[List[str]] = None) -> Dict[str, Any]:
    res = {"fdi": {}, "alerts": [], "max_idx": None, "max_val": None, "asymmetry": {}}
    try:
        if dfbands is None or dfbands.empty:
            return res
        delta = dfbands["Delta_abs"].values if "Delta_abs" in dfbands.columns else np.zeros(dfbands.shape[0])
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

# Connectivity computation (tries PLI via mne, falls back to coherence)
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: Optional[List[str]] = None, band: Tuple[float, float] = (8.0, 13.0)):
    try:
        nchan = data.shape[0]
    except Exception:
        return None, "(no data)", None, 0.0
    mat = np.zeros((nchan, nchan))
    narration = ""
    mean_conn = 0.0
    # MNE spectral_connectivity (PLI) if available
    if HAS_MNE:
        try:
            info = mne.create_info(ch_names, sf, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            try:
                from mne.connectivity import spectral_connectivity
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method='pli', mode='multitaper', sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
                conn = np.squeeze(con)
                if conn.ndim == 2 and conn.shape[0] == nchan:
                    mat = conn
                narration = f"PLI {band[0]}-{band[1]} Hz (MNE)"
            except Exception:
                narration = "(mne connectivity module unavailable — fallback to coherence)"
        except Exception as e:
            narration = "(mne connectivity failed)"
    # fallback: coherence (scipy)
    if mat.sum() == 0:
        if coherence is None:
            narration = narration or "(connectivity not available)"
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
                narration = "(connectivity computation error)"
    try:
        mean_conn = float(np.nanmean(mat)) if mat.size else 0.0
    except Exception:
        mean_conn = 0.0
    conn_img = None
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        ax.set_title("Functional Connectivity")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        conn_img = fig_to_bytes(fig)
    except Exception:
        conn_img = None
    return mat, narration, conn_img, mean_conn

# Topomap approx (scatter on 2D head layout or bars)
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
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{band_name} Topography")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            return fig_to_bytes(fig)
        else:
            fig, ax = plt.subplots(figsize=(3.2, 2.2))
            n = min(len(ch_names), len(vals))
            ax.bar(range(n), vals[:n]); ax.set_xticks(range(n)); ax.set_xticklabels(ch_names[:n], rotation=60, fontsize=7)
            ax.set_title(f"{band_name} (bar)")
            return fig_to_bytes(fig)
    except Exception as e:
        print("topomap err", e)
        return None

# SHAP helpers
def load_shap_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def shap_summary_image(shap_dict: Dict[str, float], top_n=10) -> Optional[bytes]:
    if not shap_dict:
        return None
    try:
        s = pd.Series(shap_dict).abs().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, 2.2))
        s.sort_values().plot.barh(ax=ax, color=BLUE)
        ax.set_xlabel("SHAP (abs impact)")
        return fig_to_bytes(fig)
    except Exception:
        return None

# PDF generation (reportlab)
def generate_pdf_report(summary: Dict[str, Any], lang: str = "en", amiri_path: Optional[str] = None, logo_path: Optional[str] = None) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=28, bottomMargin=28, leftMargin=36, rightMargin=36)
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
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=13))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

    story = []
    title = Paragraph("NeuroEarly Pro — Clinical QEEG Report", styles["TitleBlue"])
    if logo_path and Path(logo_path).exists():
        try:
            logo = RLImage(str(logo_path), width=1.1 * inch, height=1.1 * inch)
            header = Table([[title, logo]], colWidths=[4.8 * inch, 1.4 * inch])
            header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(header)
        except Exception:
            story.append(title)
    else:
        story.append(title)
    story.append(Spacer(1, 6))

    # Executive + ML Risk
    ml_display = summary.get("final_ml_risk_display", "N/A")
    story.append(Paragraph("<b>Executive Summary</b>", styles["H2"]))
    story.append(Paragraph(f"Final ML Risk Score: <b>{ml_display}</b>", styles["Body"]))
    story.append(Spacer(1, 6))

    # Patient table
    pat = summary.get("patient_info", {})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    ptab = [["Field", "Value"], ["ID", pat.get("id", "—")], ["DOB", pat.get("dob", "—")], ["Report Date", today]]
    t = Table(ptab, colWidths=[1.5 * inch, 4.5 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf6ff")), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Metrics table
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
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf4ff")), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.append(table)
    story.append(Spacer(1, 8))

    # Normative bar image
    if summary.get("bar_img"):
        try:
            story.append(Paragraph("<b>Normative Comparison</b>", styles["H2"]))
            story.append(RLImage(io.BytesIO(summary["bar_img"]), width=5.6 * inch, height=1.6 * inch))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # Topographies
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
            rows = []; row = []
            for i, im in enumerate(imgs):
                row.append(im)
                if len(row) == 2:
                    rows.append(row); row = []
            if row: rows.append(row)
            for r in rows:
                story.append(Table([r], colWidths=[3 * inch] * len(r)))
                story.append(Spacer(1, 4))

    # Connectivity image
    if summary.get("conn_image"):
        story.append(Paragraph("<b>Functional Connectivity (Alpha)</b>", styles["H2"]))
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

    # Tumor / FDI narrative
    if summary.get("tumor"):
        story.append(Paragraph("<b>Focal Delta / Tumor Indicators</b>", styles["H2"]))
        story.append(Paragraph(summary["tumor"].get("narrative", ""), styles["Body"]))
        if summary["tumor"].get("alerts"):
            for a in summary["tumor"]["alerts"]:
                story.append(Paragraph(f"- {a}", styles["Body"]))
        story.append(Spacer(1, 6))

    # Recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(r, styles["Body"]))
    story.append(Spacer(1, 12))

    # Footer
    footer_text = "Prepared and designed by Golden Bird LLC — Oman | 2025"
    story.append(Paragraph(footer_text, styles["Note"]))
    story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide", initial_sidebar_state="expanded")

# Header
header_html = f"""
<div style="display:flex;align-items:center;justify-content:space-between;
background: linear-gradient(90deg,{BLUE}, #2b8cff); padding:12px; border-radius:8px; color:white;">
  <div style="font-weight:700; font-size:18px;">🧠 NeuroEarly Pro — Clinical & Research</div>
  <div style="display:flex;align-items:center;">
    <div style="font-size:12px;margin-right:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:44px;">' if Path(LOGO_PATH).exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<br/>", unsafe_allow_html=True)

# Layout: left controls, right visuals
col_left, col_right = st.columns([1.0, 2.2])

with col_left:
    st.markdown("## Patient / Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English", "Arabic"], index=0)
    is_ar = (lang_choice == "Arabic")
    patient_id = st.text_input("Patient ID")
    patient_dob = st.date_input("DOB", value=date(1980, 1, 1), min_value=date(1900, 1, 1), max_value=date.today())
    patient_sex = st.selectbox("Sex", ["Unknown", "Male", "Female", "Other"])
    st.markdown("### Blood Tests (summary)")
    labs_text = st.text_area("Enter labs (one per line) e.g. B12: 250 pg/mL\nTSH: 2.1 µIU/mL")
    st.markdown("### Medications")
    meds_text = st.text_area("Current medications (one per line)")
    st.markdown("---")
    st.subheader("Processing Options")
    apply_notch = st.checkbox("Apply notch (50/60Hz)", value=True)
    compute_conn = st.checkbox("Compute connectivity (PLI/coherence)", value=True)
    gen_topos = st.checkbox("Generate topographies (5 bands)", value=True)
    enable_shap = st.checkbox("Enable XAI (SHAP visuals)", value=True)
    st.markdown("---")
    st.markdown("### Upload EDF files")
    uploads = st.file_uploader("Upload .edf files (multiple)", type=["edf"], accept_multiple_files=True)
    st.markdown("---")
    st.subheader("Questionnaires")
    st.write("PHQ-9 (Depression)")
    PHQ_QS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Sleep",  # special
        "Feeling tired or having little energy",
        "Appetite",  # special
        "Feeling bad about yourself",
        "Trouble concentrating",
        "Moving/speaking slowly OR fidgety/restless",  # special
        "Thoughts that you would be better off dead"
    ]
    phq_vals = {}
    for i, q in enumerate(PHQ_QS, start=1):
        if i == 3:
            sel = st.selectbox(f"Q{i}. Sleep", ["0 — Not at all", "1 — Insomnia", "2 — Short sleep", "3 — Hypersomnia"], key=f"phq{i}")
        elif i == 5:
            sel = st.selectbox(f"Q{i}. Appetite", ["0 — Not at all", "1 — Less eating", "2 — More eating", "3 — Variable"], key=f"phq{i}")
        elif i == 8:
            sel = st.selectbox(f"Q{i}. Motor/Activity", ["0 — Not at all", "1 — Slow speech/move", "2 — Fidgety/restless", "3 — Both"], key=f"phq{i}")
        else:
            sel = st.radio(f"Q{i}. {q}", ["0 — Not at all", "1 — Several days", "2 — More than half the days", "3 — Nearly every day"], key=f"phq{i}", horizontal=True)
        try:
            phq_vals[f"Q{i}"] = int(str(sel).split("—")[0].strip())
        except Exception:
            phq_vals[f"Q{i}"] = 0
    phq_total = sum(phq_vals.values())
    st.info(f"PHQ-9 total: {phq_total} / 27")

    st.markdown("### AD8 (Cognitive screening)")
    AD8_QS = [
        "Problems with judgment", "Less interest in hobbies", "Repeats questions/stories",
        "Trouble learning to use a tool", "Forgetting month/year", "Difficulty handling finances",
        "Trouble remembering appointments", "Daily problems with thinking and memory"
    ]
    ad8_vals = {}
    for i, q in enumerate(AD8_QS, start=1):
        sel = st.selectbox(f"A{i}. {q}", options=[0, 1], key=f"ad8_{i}")
        ad8_vals[f"A{i}"] = int(sel)
    ad8_total = sum(ad8_vals.values())
    st.info(f"AD8 total: {ad8_total} / 8")
    st.markdown("---")
    if st.button("Process EDF(s)"):
        if not uploads:
            st.error("Upload at least one EDF file.")
        else:
            processing = st.empty()
            results = []
            for up in uploads:
                processing.info(f"Processing {up.name} ...")
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                    tmp.write(up.getbuffer()); tmp.flush(); tmp.close()
                    data, sf, ch_names = read_edf_file(tmp.name)
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
                    agg = aggregate_features(dfbands, ch_names=list(dfbands.index))
                    focal = compute_focal_delta(dfbands, ch_names=list(dfbands.index))
                    topo_imgs = {}
                    if gen_topos:
                        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                            try:
                                vals = dfbands[f"{band}_rel"].values if f"{band}_rel" in dfbands.columns else np.zeros(dfbands.shape[0])
                                topo_imgs[band] = generate_topomap_image(vals, ch_names=list(dfbands.index), band_name=band)
                            except Exception:
                                topo_imgs[band] = None
                    conn_mat = None; conn_narr = None; conn_img = None; mean_conn = 0.0
                    if compute_conn:
                        try:
                            conn_mat, conn_narr, conn_img, mean_conn = compute_connectivity_matrix(cleaned, sf, ch_names=list(dfbands.index), band=BANDS.get("Alpha", (8.0, 13.0)))
                        except Exception:
                            conn_mat, conn_narr, conn_img, mean_conn = (None, "(connectivity failed)", None, 0.0)
                    # ML risk heuristic (weighted)
                    ta = agg.get("theta_alpha_ratio", 0.0)
                    phq_norm = (phq_total / 27.0) if phq_total else 0.0
                    ad8_norm = (ad8_total / 8.0) if ad8_total else 0.0
                    mc = float(mean_conn) if mean_conn is not None else 0.0
                    mc_norm = max(0.0, 1.0 - mc)  # heuristic: lower connectivity -> higher risk
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
                st.session_state["ne_results"] = results
                st.success(f"{len(results)} file(s) processed.")

with col_right:
    st.markdown("## Console / Visualization")
    if "ne_results" in st.session_state and st.session_state["ne_results"]:
        r0 = st.session_state["ne_results"][0]
        st.metric("Final ML Risk Score", f"{r0['ml_risk']*100:.1f}%")
        st.markdown("### QEEG Key Metrics")
        try:
            df_metrics = pd.DataFrame([{
                "Theta/Alpha Ratio": r0["agg"].get("theta_alpha_ratio", 0),
                "Theta/Beta Ratio": r0["agg"].get("theta_beta_ratio", 0),
                "Alpha mean (rel)": r0["agg"].get("alpha_rel_mean", 0),
                "Theta mean (rel)": r0["agg"].get("theta_rel_mean", 0),
                "Alpha Asymmetry (F3-F4)": r0["agg"].get("alpha_asym_F3_F4", 0),
                "Mean Connectivity (alpha)": r0.get("mean_connectivity", 0.0)
            }]).T.rename(columns={0: "Value"})
            st.table(df_metrics)
        except Exception:
            st.write(r0["agg"])

        # Normative comparison - Theta/Alpha + Alpha Asymmetry bar
        try:
            ta_val = r0["agg"].get("theta_alpha_ratio", 0.0)
            aa_val = r0["agg"].get("alpha_asym_F3_F4", 0.0)
            fig, ax = plt.subplots(figsize=(6, 1.6))
            # healthy zone
            rng = NORM["theta_alpha"]["healthy"]
            ax.barh([0], [rng[1] - rng[0]], left=rng[0], height=0.6, color='white', edgecolor='gray')
            # patient value overlay
            ax.barh([0], [ta_val], left=0, height=0.4, color=BLUE)
            ax.set_yticks([]); ax.set_title("Theta/Alpha comparison (normative)")
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            st.image(buf.getvalue(), width=680)
        except Exception:
            st.info("Normative comparison failed.")

        st.markdown("### Topography maps")
        cols = st.columns(5)
        bands_show = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        for i, band in enumerate(bands_show):
            img = r0["topo_imgs"].get(band)
            if img:
                cols[i].image(img, caption=band, width=150)
            else:
                cols[i].text("—")

        st.markdown("### Connectivity")
        if r0.get("connectivity_img"):
            st.image(r0["connectivity_img"], caption=r0.get("connectivity_narr", ""), width=640)
        else:
            st.info("Connectivity image not available.")

        st.markdown("### Focal Delta / Tumor Indicators")
        st.json(r0["focal"])

        if enable_shap:
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
                    st.image(buf.getvalue(), width=680)
                else:
                    st.info("SHAP file present but no matching model key.")
            else:
                st.info("No shap_summary.json found. Upload to enable XAI visuals.")
    else:
        st.info("No processed results yet. Upload EDF and press 'Process EDF(s)' in the left panel.")

# Export & PDF
st.markdown("---")
st.markdown("## Export & Reports")
col_export_left, col_export_right = st.columns([1, 2])

with col_export_left:
    if st.button("Download metrics CSV (all processed)"):
        if "ne_results" not in st.session_state or not st.session_state["ne_results"]:
            st.error("No results to export.")
        else:
            rows = []
            for r in st.session_state["ne_results"]:
                rows.append({
                    "file": r["filename"],
                    "ml_risk": r["ml_risk"],
                    "theta_alpha": r["agg"].get("theta_alpha_ratio"),
                    "theta_beta": r["agg"].get("theta_beta_ratio"),
                    "alpha_asym_F3_F4": r["agg"].get("alpha_asym_F3_F4"),
                    "mean_connectivity": r.get("mean_connectivity")
                })
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")

with col_export_right:
    pdf_lang = st.selectbox("PDF language", options=["English", "Arabic"], index=0)
    if st.button("Generate & Download Clinical PDF"):
        try:
            if "ne_results" not in st.session_state or not st.session_state["ne_results"]:
                st.error("No processed results — run Process EDF(s) first.")
            else:
                r0 = st.session_state["ne_results"][0]
                summary = {}
                summary["patient_info"] = {"id": patient_id or "—", "dob": str(patient_dob)}
                summary["final_ml_risk_display"] = f"{r0['ml_risk']*100:.1f}%"
                summary["metrics"] = {
                    "theta_alpha_ratio": r0["agg"].get("theta_alpha_ratio", 0.0),
                    "theta_beta_ratio": r0["agg"].get("theta_beta_ratio", 0.0),
                    "alpha_asym_F3_F4": r0["agg"].get("alpha_asym_F3_F4", 0.0),
                    "gamma_rel_mean": r0["agg"].get("gamma_rel_mean", 0.0) if "gamma_rel_mean" in r0["agg"] else 0.0,
                    "mean_connectivity": r0.get("mean_connectivity", 0.0)
                }
                summary["topo_images"] = {k: v for k, v in (r0.get("topo_imgs") or {}).items()}
                summary["conn_image"] = r0.get("connectivity_img")
                # bar image
                try:
                    fig, ax = plt.subplots(figsize=(5.6, 1.6))
                    ta = summary["metrics"]["theta_alpha_ratio"] or 0.0
                    aa = summary["metrics"]["alpha_asym_F3_F4"] or 0.0
                    ax.bar([0, 1], [ta, aa], width=0.6, color=BLUE)
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["Theta/Alpha", "Alpha Asym"])
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    summary["bar_img"] = buf.getvalue()
                except Exception:
                    summary["bar_img"] = None
                # SHAP
                shap_table = {}
                shap_img = None
                if SHAP_JSON.exists():
                    sd = load_shap_summary(SHAP_JSON)
                    key = "depression_global" if summary["metrics"]["theta_alpha_ratio"] <= 1.3 else "alzheimers_global"
                    feats = sd.get(key, {}) if sd else {}
                    if feats:
                        shap_table = feats
                        shap_img = shap_summary_image(feats, top_n=10)
                summary["shap_table"] = shap_table
                summary["shap_img"] = shap_img
                # tumor narrative
                summary["tumor"] = {"narrative": f"FDI max {r0['focal'].get('max_val')} at index {r0['focal'].get('max_idx')}", "alerts": [str(a) for a in r0['focal'].get('alerts', [])]}
                recs = []
                recs.append("Correlate QEEG findings with clinical exam and PHQ-9/AD8 scores.")
                if summary["metrics"]["theta_alpha_ratio"] > 1.4 or (r0["focal"].get("max_val", 0) and r0["focal"].get("max_val", 0) > 2.0):
                    recs.append("Recommend neuroimaging (MRI) and neurology referral for further evaluation.")
                else:
                    recs.append("Clinical follow-up and repeat EEG in 3-6 months.")
                recs.append("Check reversible causes: B12, TSH, metabolic panel.")
                summary["recommendations"] = recs
                amiri = str(AMIRI_PATH) if AMIRI_PATH.exists() else None
                pdf_bytes = generate_pdf_report(summary, lang=("ar" if pdf_lang == "Arabic" else "en"), amiri_path=amiri, logo_path=str(LOGO_PATH))
                if pdf_bytes:
                    st.success("PDF generated.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation returned empty.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.text(safe_trace(e))

st.markdown("---")
st.markdown(f"<small style='color:gray'>Prepared and designed by Golden Bird LLC — Oman | 2025</small>", unsafe_allow_html=True)
