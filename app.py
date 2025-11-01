# app.py — NeuroEarly Pro — Final Professional (Clinical) v3.0
# IMPORTANT: Put assets/goldenbird_logo.png and assets/Amiri-Regular.ttf in repo root (or adjust paths below).
# Requires: streamlit, numpy, pandas, matplotlib, scipy, pyedflib or mne, reportlab, shap (optional), arabic-reshaper/python-bidi (for Arabic shaping)
# This file is intended to be a single-file deployable Streamlit app.

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
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# ---------- Optional heavy libs (graceful fallback) ----------
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_RESHAPER = False
HAS_BIDI = False

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
    HAS_ARABIC_RESHAPER = True
    HAS_BIDI = True
except Exception:
    HAS_ARABIC_RESHAPER = False
    HAS_BIDI = False

try:
    import scipy.signal as sps
    from scipy.signal import welch, coherence, iirnotch, filtfilt, butter
except Exception:
    sps = None

matplotlib.rcParams['font.size'] = 10

# ---------- Paths / constants ----------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ASSETS / "Amiri-Regular.ttf"  # adjust if your font path differs
SHAP_JSON = ROOT / "shap_summary.json"

BLUE_HEX = "#0b63d6"
LIGHT_GRAY = "#f3f7fb"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Normative ranges for the bar chart comparison (example)
NORM = {
    "theta_alpha": {"healthy": (0.0, 1.1), "at_risk": (1.1, 1.6)},
    "alpha_asym": {"healthy": (-0.05, 0.05), "at_risk": (-0.2, -0.05)}
}

# ---------- Helpers ----------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e: Exception):
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    st.error("Internal error — check logs (developer).")
    st.code(tb)

def reshape_ar(text: str) -> str:
    if HAS_ARABIC_RESHAPER and HAS_BIDI:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def safe_fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------- EDF reading ----------
def read_edf_path(path: str):
    """Return data: ndarray shape (n_channels, n_samples), sf, ch_names"""
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        sf = float(raw.info.get("sfreq", 256.0))
        chs = raw.ch_names
        return data, sf, chs
    elif HAS_PYEDF:
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
    else:
        raise RuntimeError("No EDF backend installed (install mne or pyedflib)")

# ---------- Filtering ----------
def apply_notch(sig, sf, freq=50.0, Q=30.0):
    try:
        b,a = iirnotch(freq, Q, sf)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass(sig, sf, low=0.5, high=45.0, order=4):
    try:
        nyq = 0.5*sf
        low_n = max(low/nyq, 1e-6)
        high_n = min(high/nyq, 0.9999)
        b,a = butter(order, [low_n, high_n], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(data: np.ndarray, sf: float, do_notch=True):
    """data: (n_channels, n_samples)"""
    cleaned = np.copy(data)
    nch = cleaned.shape[0]
    for i in range(nch):
        s = cleaned[i,:].astype(float)
        if do_notch:
            # try 50 and 60
            for f0 in (50.0, 60.0):
                try:
                    s = apply_notch(s, sf, freq=f0)
                except Exception:
                    pass
        try:
            s = bandpass(s, sf)
        except Exception:
            pass
        cleaned[i,:] = s
    return cleaned

# ---------- PSD and band powers ----------
def compute_bandpowers(data: np.ndarray, sf: float, nperseg=2048):
    """data: (n_channels, n_samples) -> DataFrame index channels, columns like 'Alpha_abs','Alpha_rel'"""
    nchan = data.shape[0]
    results = []
    for ch in range(nchan):
        sig = data[ch]
        try:
            freqs, pxx = welch(sig, fs=sf, nperseg=min(nperseg, len(sig)))
        except Exception:
            freqs = np.array([0.]); pxx = np.array([0.])
        total = np.trapz(pxx, freqs) if freqs.size else 1.0
        row = {}
        for band,(lo,hi) in BANDS.items():
            mask = (freqs>=lo) & (freqs<=hi)
            abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum()>0 else 0.0
            rel = float(abs_p / (total if total>0 else 1.0))
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        results.append(row)
    df = pd.DataFrame(results)
    return df

# ---------- Aggregate features ----------
def aggregate_features(dfbands: pd.DataFrame, ch_names: List[str]=None):
    out = {}
    if dfbands.empty:
        for band in BANDS:
            out[f"{band.lower()}_rel_mean"] = 0.0
        out["theta_alpha_ratio"] = 0.0
        out["alpha_asym_F3_F4"] = 0.0
        return out
    for band in BANDS:
        out[f"{band.lower()}_rel_mean"] = float(np.nanmean(dfbands[f"{band}_rel"].values)) if f"{band}_rel" in dfbands.columns else 0.0
    alpha = out.get("alpha_rel_mean", 1e-9)
    theta = out.get("theta_rel_mean", 0.0)
    beta = out.get("beta_rel_mean", 1e-9)
    out["theta_alpha_ratio"] = float(theta/alpha) if alpha>0 else 0.0
    out["theta_beta_ratio"] = float(theta/beta) if beta>0 else 0.0
    # alpha asymmetry F3-F4
    out["alpha_asym_F3_F4"] = 0.0
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            if "F3" in names and "F4" in names:
                i3 = names.index("F3"); i4 = names.index("F4")
                a3 = dfbands.iloc[i3][f"Alpha_rel"] if f"Alpha_rel" in dfbands.columns else 0.0
                a4 = dfbands.iloc[i4][f"Alpha_rel"] if f"Alpha_rel" in dfbands.columns else 0.0
                out["alpha_asym_F3_F4"] = float(a3 - a4)
        except Exception:
            out["alpha_asym_F3_F4"] = out.get("alpha_asym_F3_F4", 0.0)
    return out

# ---------- Focal Delta Index and tumor indicators ----------
def compute_focal_delta_index(dfbands: pd.DataFrame, ch_names: List[str]=None):
    res = {"fdi": {}, "alerts": [], "max_idx": None, "max_val": None, "asymmetry": {}}
    try:
        if dfbands is None or dfbands.empty:
            return res
        # use Delta_abs
        if "Delta_abs" in dfbands.columns:
            delta = dfbands["Delta_abs"].values
        else:
            delta = np.zeros(dfbands.shape[0])
        global_mean = float(np.nanmean(delta)) if delta.size>0 else 1e-9
        for idx, val in enumerate(delta):
            fdi = float(val / (global_mean if global_mean>0 else 1e-9))
            res["fdi"][idx] = fdi
            if fdi > 2.0:
                ch = ch_names[idx] if ch_names and idx<len(ch_names) else f"ch{idx}"
                res["alerts"].append({"type":"FDI","channel":ch,"value":fdi})
        # extreme asymmetry for pairs
        pairs = [("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        names_map = {}
        if ch_names:
            for i,n in enumerate(ch_names):
                names_map[n.upper()] = i
        for L,R in pairs:
            if L in names_map and R in names_map:
                lidx = names_map[L]; ridx = names_map[R]
                dl = delta[lidx]; dr = delta[ridx]
                ratio = float(dr / (dl+1e-9)) if dl>0 else float("inf") if dr>0 else 1.0
                res["asymmetry"][f"{L}/{R}"] = ratio
                if (isinstance(ratio,float) and (ratio>3.0 or ratio<0.33)) or (ratio==float("inf")):
                    res["alerts"].append({"type":"asymmetry","pair":f"{L}/{R}","ratio":ratio})
        # max
        max_idx = int(np.argmax(list(res["fdi"].values()))) if res["fdi"] else None
        max_val = res["fdi"].get(max_idx, None) if max_idx is not None else None
        res["max_idx"] = max_idx; res["max_val"] = max_val
    except Exception as e:
        print("compute_focal_delta_index err:", e)
    return res

# ---------- Connectivity ----------
def compute_connectivity(data: np.ndarray, sf: float, ch_names: List[str]=None, band=(8.0,13.0)):
    """Return matrix and narrative and image bytes (if matplotlib available)"""
    try:
        nchan = data.shape[0]
        if HAS_MNE:
            try:
                info = mne.create_info(ch_names, sf, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity([raw], method='coh', sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
                # con might be array-like
                conn = np.squeeze(con) if hasattr(con, "__len__") else np.array(con)
                # ensure shape NxN
                if conn.ndim==2 and conn.shape[0]==nchan:
                    mat = conn
                else:
                    mat = np.zeros((nchan,nchan))
                    # fallback fill
                narr = f"Coherence {band[0]}-{band[1]} Hz (MNE)"
            except Exception as e:
                print("mne connectivity failed:", e)
                mat = np.zeros((nchan,nchan))
                narr = "(connectivity failed)"
        else:
            # fallback: pairwise coherence with scipy
            mat = np.zeros((nchan,nchan))
            for i in range(nchan):
                for j in range(i, nchan):
                    try:
                        f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                        mask = (f>=band[0]) & (f<=band[1])
                        val = float(np.nanmean(Cxy[mask])) if mask.sum()>0 else 0.0
                    except Exception:
                        val = 0.0
                    mat[i,j] = val; mat[j,i] = val
            narr = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
        # create image
        conn_img = None
        if HAS_MATPLOTLIB:
            try:
                fig,ax = plt.subplots(figsize=(5,3))
                im = ax.imshow(mat, cmap='viridis', aspect='auto')
                ax.set_title("Connectivity (alpha)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                conn_img = safe_fig_to_png_bytes(fig)
            except Exception as e:
                print("conn img err:", e)
        mean_conn = float(np.nanmean(mat)) if mat.size>0 else 0.0
        return mat, narr, conn_img, mean_conn
    except Exception as e:
        print("compute_connectivity err:", e)
        return None, "(error)", None, None

# ---------- Topography (approx) ----------
def generate_topomap(vals: np.ndarray, ch_names: List[str], band_name: str):
    """Return PNG bytes - simple scatter on approximate 10-20 positions or fallback barplot"""
    if not HAS_MATPLOTLIB:
        return None
    # approximate coordinate map for common channels
    pos = {
        'Fp1':(-0.5,1),'Fp2':(0.5,1),'F3':(-0.8,0.3),'F4':(0.8,0.3),
        'C3':(-0.8,-0.3),'C4':(0.8,-0.3),'P3':(-0.5,-0.8),'P4':(0.5,-0.8),
        'O1':(-0.2,-1),'O2':(0.2,-1),'F7':(-1,0.6),'F8':(1,0.6),'T7':(-1,-0.3),'T8':(1,-0.3)
    }
    xs, ys, vals_plot = [], [], []
    for i,ch in enumerate(ch_names):
        up = ch.upper()
        if up in pos:
            x,y = pos[up]
            xs.append(x); ys.append(y)
            vals_plot.append(float(vals[i]))
    if len(xs) >= 3:
        fig,ax = plt.subplots(figsize=(4,3))
        sc = ax.scatter(xs, ys, c=vals_plot, s=300, cmap='RdBu_r')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{band_name} Topography")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        return safe_fig_to_png_bytes(fig)
    else:
        # fallback bar chart
        fig,ax = plt.subplots(figsize=(4,3))
        n = min(len(ch_names), len(vals))
        names = ch_names[:n]
        ax.bar(range(n), vals[:n])
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_title(f"{band_name} (bar fallback)")
        return safe_fig_to_png_bytes(fig)

# ---------- SHAP visualization helpers ----------
def load_shap_summary(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return None
    return None

def shap_bar_image(shap_dict: Dict[str,float], top_n=10):
    if not HAS_MATPLOTLIB or not shap_dict:
        return None
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]
    fig,ax = plt.subplots(figsize=(6,2.6))
    ax.barh(labels, values, color=BLUE_HEX)
    ax.set_xlabel("SHAP (impact)")
    fig.tight_layout()
    return safe_fig_to_png_bytes(fig)

# ---------- PDF generation (ReportLab) ----------
def generate_pdf(summary: Dict[str,Any], lang="en", amiri_path: Optional[str]=None):
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception as e:
            print("Amiri register failed:", e)
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE_HEX), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE_HEX), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
    story = []
    # header with logo on right
    header_cells = []
    left = Paragraph("NeuroEarly Pro — Clinical QEEG Report", styles["TitleBlue"])
    if LOGO_PATH.exists():
        try:
            img = RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch)
            header_cells = [[left, img]]
            header_table = Table(header_cells, colWidths=[4.7*inch, 1.4*inch])
            header_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(header_table)
        except Exception:
            story.append(left)
    else:
        story.append(left)
    story.append(Spacer(1,6))
    # exec summary
    ml = summary.get("final_ml_risk_display", "N/A")
    story.append(Paragraph(f"<b>Executive Summary</b>", styles["H2"]))
    story.append(Paragraph(f"Final ML Risk Score: <b>{ml}</b>", styles["Body"]))
    story.append(Spacer(1,6))
    # patient metadata
    pat = summary.get("patient_info", {})
    ptab = [["Field","Value"],["ID", pat.get("id","—")],["DOB", pat.get("dob","—")],["Sex", pat.get("sex","—")]]
    t = Table(ptab, colWidths=[1.4*inch, 4.7*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff")), ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey)]))
    story.append(t)
    story.append(Spacer(1,8))
    # QEEG metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
    metrics = summary.get("metrics", {})
    mt = [["Metric","Value","Note"]]
    # desired order
    desired = [("theta_alpha_ratio","Theta/Alpha Ratio","Slowing indicator"),
               ("theta_beta_ratio","Theta/Beta Ratio","Stress/inattention"),
               ("alpha_asym_F3_F4","Alpha Asymmetry (F3-F4)","Left-right asymmetry"),
               ("gamma_rel_mean","Gamma Relative Mean","Cognition"),
               ("mean_connectivity","Mean Connectivity (alpha)","Functional coherence")]
    for k,label,note in desired:
        v = metrics.get(k, summary.get(k, "N/A"))
        try:
            vv = f"{float(v):.4f}"
        except Exception:
            vv = str(v)
        mt.append([label, vv, note])
    table = Table(mt, colWidths=[2.6*inch, 1.2*inch, 2.3*inch])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),9)]))
    story.append(table)
    story.append(Spacer(1,8))
    # bar image
    if summary.get("bar_img"):
        try:
            story.append(Paragraph("<b>Power Ratio Comparison</b>", styles["H2"]))
            story.append(RLImage(io.BytesIO(summary["bar_img"]), width=5.6*inch, height=1.6*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass
    # topomaps
    if summary.get("topo_images"):
        story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
        imgs = []
        for band,buf in summary["topo_images"].items():
            if buf:
                try:
                    imgs.append(RLImage(io.BytesIO(buf), width=2.4*inch, height=1.5*inch))
                except Exception:
                    pass
        if imgs:
            # put in table row(s) with up to 2 per row
            rows = []
            row = []
            for i,img in enumerate(imgs):
                row.append(img)
                if len(row)==2:
                    rows.append(row); row=[]
            if row:
                rows.append(row)
            for r in rows:
                story.append(Table([r], colWidths=[3*inch]*len(r)))
                story.append(Spacer(1,4))
    # connectivity
    if summary.get("conn_image"):
        story.append(Paragraph("<b>Connectivity (Alpha)</b>", styles["H2"]))
        try:
            story.append(RLImage(io.BytesIO(summary["conn_image"]), width=5.6*inch, height=2.4*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass
    # SHAP
    if summary.get("shap_img"):
        story.append(Paragraph("<b>Explainable AI — SHAP top contributors</b>", styles["H2"]))
        try:
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.6*inch, height=1.8*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass
    elif summary.get("shap_table"):
        story.append(Paragraph("<b>Explainable AI — Top contributors</b>", styles["H2"]))
        stbl = [["Feature","Importance"]]
        for k,v in summary["shap_table"].items():
            stbl.append([k, f"{v:.4f}"])
        t2 = Table(stbl, colWidths=[3.5*inch, 2*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(t2)
        story.append(Spacer(1,6))
    # tumor / focal
    if summary.get("tumor"):
        story.append(Paragraph("<b>Focal Delta / Tumor Indicators</b>", styles["H2"]))
        ttxt = summary["tumor"].get("narrative", "")
        story.append(Paragraph(ttxt, styles["Body"]))
        if summary["tumor"].get("alerts"):
            for a in summary["tumor"]["alerts"]:
                story.append(Paragraph(f"- {a}", styles["Body"]))
        story.append(Spacer(1,6))
    # clinical recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(r, styles["Body"]))
    story.append(Spacer(1,12))
    # footer mention + doctor signature line (no patient signature)
    story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
    story.append(Spacer(1,18))
    story.append(Paragraph("Doctor signature: ___________________________", styles["Body"]))
    # build
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide", initial_sidebar_state="expanded")

# Top header
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; background:linear-gradient(90deg,{BLUE_HEX}, #2b8cff); padding:12px; border-radius:8px; color:white;">
  <div style="font-size:20px; font-weight:600;">🧠 NeuroEarly Pro — Clinical AI</div>
  <div style="display:flex; align-items:center;">
    <div style="font-size:12px; margin-right:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:40px;">' if LOGO_PATH.exists() else ''}
  </div>
</div>
""", unsafe_allow_html=True)

# Layout columns
col_main, col_side = st.columns([3,1])

with col_side:
    st.markdown("### Settings")
    # Language selection
    lang_choice = st.selectbox("Language / اللغة", options=["English","Arabic"], index=0)
    is_ar = (lang_choice == "Arabic")
    st.markdown("---")
    st.subheader("Patient")
    # Basic patient info (DOB constrained per user earlier)
    patient_id = st.text_input("ID", key="pid")
    patient_dob = st.date_input("DOB", min_value=date(1900,1,1), max_value=date.today(), value=date(1980,1,1))
    patient_sex = st.selectbox("Sex", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.subheader("Clinical / Labs")
    labs = st.multiselect("Relevant labs (select)", options=["B12","TSH","Vitamin D","Folate","Homocysteine","HbA1C","Cholesterol"])
    meds = st.text_area("Current medications (one per line)")
    comorbid = st.text_area("Comorbid conditions (one per line)")
    st.markdown("---")
    st.write("Backends status:")
    st.write(f"mne={HAS_MNE} pyedflib={HAS_PYEDF} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

with col_main:
    st.markdown("## 1) Upload EDF files")
    uploads = st.file_uploader("Upload EDF (.edf) — multiple allowed", type=["edf"], accept_multiple_files=True)
    st.markdown("## 2) Questionnaires")
    st.markdown("### PHQ-9 (Depression)")
    # PHQ-9 questions with specified changes
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
    for i,q in enumerate(PHQ_QS, start=1):
        if i==3:
            opts = ["0 — Not at all","1 — Insomnia (difficulty falling/staying asleep)","2 — Sleeping less","3 — Sleeping more"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        elif i==5:
            opts = ["0 — Not at all","1 — Eating less","2 — Eating more","3 — Both/variable"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        elif i==8:
            opts = ["0 — Not at all","1 — Moving/speaking slowly","2 — Fidgety/restless","3 — Both/variable"]
            sel = st.radio(f"Q{i}. {q}", options=opts, horizontal=True, key=f"phq{i}")
            try:
                phq_vals[f"Q{i}"] = int(sel.split("—")[0].strip())
            except Exception:
                phq_vals[f"Q{i}"] = 0
        else:
            sel = st.radio(f"Q{i}. {q}", options=["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"], horizontal=True, key=f"phq{i}")
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
    for i,q in enumerate(AD8_QS, start=1):
        col = cols[(i-1)%2]
        sel = col.radio(f"A{i}. {q}", options=[0,1], key=f"ad8_{i}", horizontal=True)
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
                    # save temp
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                    tmp.write(up.getbuffer()); tmp.flush(); tmp.close()
                    data, sf, ch_names = read_edf_path(tmp.name)  # data: (nchan, n_samples)
                    # basic checks
                    if data is None:
                        processing.error(f"Could not read {up.name}")
                        continue
                    # preprocess
                    cleaned = preprocess_data(data, sf, do_notch=apply_notch)
                    # bandpowers DataFrame
                    dfbands = compute_bandpowers(cleaned, sf)
                    # attach ch names as index if lengths match
                    try:
                        if len(ch_names) == dfbands.shape[0]:
                            dfbands.index = ch_names
                        else:
                            # create placeholder names
                            dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    except Exception:
                        dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    # aggregate features
                    agg = aggregate_features(dfbands, ch_names=ch_names)
                    # focal delta index
                    focal = compute_focal_delta_index(dfbands, ch_names=ch_names)
                    # topomaps
                    topo_imgs = {}
                    if generate_topos:
                        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                            try:
                                vals = dfbands[f"{band}_rel"].values if f"{band}_rel" in dfbands.columns else np.zeros(dfbands.shape[0])
                                topo_imgs[band] = generate_topomap(vals, ch_names=list(dfbands.index), band_name=band)
                            except Exception:
                                topo_imgs[band] = None
                    # connectivity
                    conn_mat, conn_narr, conn_img, mean_conn = (None, None, None, None)
                    if compute_conn:
                        try:
                            conn_mat, conn_narr, conn_img, mean_conn = compute_connectivity(cleaned, sf, ch_names=list(dfbands.index), band=(8.0,13.0))
                        except Exception:
                            conn_mat, conn_narr, conn_img, mean_conn = (None, "(connectivity failed)", None, None)
                    # simple ML heuristic (transparent)
                    ta = agg.get("theta_alpha_ratio", 0.0)
                    phq_norm = phq_total / 27.0
                    ad8_norm = ad8_total / 8.0
                    ta_norm = min(1.0, ta / 1.6)
                    ml_risk = min(1.0, (ta_norm * 0.55 + phq_norm * 0.3 + ad8_norm * 0.15))
                    # collect
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
                    st.error(traceback.format_exc())
            # show results summary for first file
            if results:
                st.markdown("## Results (first file)")
                r0 = results[0]
                st.metric("Final ML Risk Score", f"{r0['ml_risk']*100:.1f}%")
                st.markdown("### QEEG Key Metrics")
                try:
                    st.table(pd.DataFrame([{
                        "Theta/Alpha Ratio": r0["agg"].get("theta_alpha_ratio",0),
                        "Theta/Beta Ratio": r0["agg"].get("theta_beta_ratio",0),
                        "Alpha mean (rel)": r0["agg"].get("alpha_rel_mean",0),
                        "Theta mean (rel)": r0["agg"].get("theta_rel_mean",0),
                        "Alpha Asymmetry (F3-F4)": r0["agg"].get("alpha_asym_F3_F4",0)
                    }]).T.rename(columns={0:"Value"}))
                except Exception:
                    st.write(r0["agg"])
                st.markdown("### Normative Comparison")
                # create bar images
                try:
                    bar_img = None
                    # Theta/Alpha comparison
                    ta_val = r0["agg"].get("theta_alpha_ratio", 0.0)
                    fig,ax = plt.subplots(figsize=(4.5,1.8))
                    rng = NORM["theta_alpha"]
                    ax.barh([0], [rng[1]-rng[0]], left=rng[0], height=0.6, color='white', edgecolor='gray')
                    ax.barh([0], [min(ta_val,rng[1]*2)-0], left=0, height=0.4, color=BLUE_HEX)
                    ax.set_yticks([])
                    ax.set_title("Theta/Alpha comparison")
                    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0); bar_img = buf.getvalue()
                    st.image(bar_img, use_column_width=False)
                except Exception:
                    pass
                st.markdown("### Topography (Alpha/Gamma/…)")
                cols = st.columns(5)
                bands_show = ["Delta","Theta","Alpha","Beta","Gamma"]
                for i,band in enumerate(bands_show):
                    img = r0["topo_imgs"].get(band)
                    if img:
                        cols[i].image(img, caption=band, use_container_width=True)
                st.markdown("### Connectivity")
                if r0.get("connectivity_img"):
                    st.image(r0["connectivity_img"], caption="Connectivity (Alpha)", use_container_width=True)
                else:
                    st.info("Connectivity image not available.")
                st.markdown("### Focal Delta / Tumor Indicators")
                st.json(r0["focal"])
                # XAI UI
                if use_shap:
                    shap_data = load_shap_summary(SHAP_JSON) if SHAP_JSON.exists() else None
                    if shap_data:
                        model_key = "depression_global" if r0["agg"].get("theta_alpha_ratio",0) <= 1.3 else "alzheimers_global"
                        features = shap_data.get(model_key, {})
                        if features:
                            st.markdown("### SHAP Top contributors")
                            s = pd.Series(features).abs().sort_values(ascending=False)
                            # matplotlib rendering for nice look
                            fig,ax = plt.subplots(figsize=(6,2.2))
                            s.head(10).sort_values().plot.barh(ax=ax, color=BLUE_HEX); ax.set_xlabel("abs SHAP")
                            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                            st.image(buf.getvalue(), caption="SHAP Top contributors", use_column_width=True)
                        else:
                            st.info("SHAP file present but no model key matched.")
                    else:
                        st.info("No shap_summary.json found. Upload shap file in sidebar to enable XAI.")
            # persist results in session state to generate PDF
            st.session_state["ne_results"] = results

    # Generate PDF
    st.markdown("---")
    st.markdown("## Generate Clinical PDF Report")
    report_lang = st.selectbox("Report language for PDF", options=["English","Arabic"], index=0)
    if st.button("Generate & Download PDF"):
        try:
            if "ne_results" not in st.session_state or not st.session_state["ne_results"]:
                st.error("No processed results available — process EDF(s) first.")
            else:
                r0 = st.session_state["ne_results"][0]
                summary = {}
                summary["patient_info"] = {"id": patient_id or "—", "dob": str(patient_dob), "sex": patient_sex}
                summary["final_ml_risk_display"] = f"{r0['ml_risk']*100:.1f}%"
                # fill metrics
                summary["metrics"] = {
                    "theta_alpha_ratio": r0["agg"].get("theta_alpha_ratio", 0.0),
                    "theta_beta_ratio": r0["agg"].get("theta_beta_ratio", 0.0),
                    "alpha_asym_F3_F4": r0["agg"].get("alpha_asym_F3_F4", 0.0),
                    "gamma_rel_mean": r0["agg"].get("gamma_rel_mean", 0.0) if "gamma_rel_mean" in r0["agg"] else 0.0,
                    "mean_connectivity": r0.get("mean_connectivity", None)
                }
                # topo images bytes
                summary["topo_images"] = {k:v for k,v in (r0.get("topo_imgs") or {}).items()}
                summary["conn_image"] = r0.get("connectivity_img")
                # SHAP
                shap_table = {}
                shap_img = None
                if SHAP_JSON.exists():
                    try:
                        sd = json.load(open(SHAP_JSON,"r",encoding="utf-8"))
                        model_key = "depression_global" if summary["metrics"]["theta_alpha_ratio"] <= 1.3 else "alzheimers_global"
                        feats = sd.get(model_key, {})
                        if feats:
                            shap_table = feats
                            shap_img = shap_bar_image(feats, top_n=10)
                    except Exception:
                        shap_table = {}
                summary["shap_table"] = shap_table
                summary["shap_img"] = shap_img
                # tumor
                summary["tumor"] = {"narrative": r0["focal"].get("narrative",""), "alerts":[str(a) for a in r0["focal"].get("alerts",[])]}
                # recommendations
                recs = []
                recs.append("Correlate QEEG findings with clinical exam and PHQ-9/AD8 scores.")
                if summary["metrics"]["theta_alpha_ratio"] > 1.4 or r0["focal"].get("max_val",0) and r0["focal"].get("max_val",0)>2.0:
                    recs.append("Recommend neuroimaging (MRI) and neurology referral for further evaluation.")
                else:
                    recs.append("Clinical follow-up and repeat EEG in 3-6 months.")
                recs.append("Check reversible causes: B12, TSH, metabolic panel.")
                summary["recommendations"] = recs
                # bar image (Theta/Alpha & Alpha asymmetry comparison)
                try:
                    fig,ax = plt.subplots(figsize=(5.6,1.6))
                    ta = summary["metrics"]["theta_alpha_ratio"] or 0.0
                    aa = summary["metrics"]["alpha_asym_F3_F4"] or 0.0
                    ax.bar([0,1],[ta, aa], width=0.6, color=BLUE_HEX)
                    ax.set_xticks([0,1]); ax.set_xticklabels(["Theta/Alpha","Alpha Asym (F3-F4)"])
                    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                    summary["bar_img"] = buf.getvalue()
                except Exception:
                    summary["bar_img"] = None
                # Final ML risk numeric for logic
                summary["final_ml_risk_num"] = r0["ml_risk"]
                # generate PDF bytes
                amiri = str(AMIRI_PATH) if AMIRI_PATH.exists() else None
                pdf_bytes = generate_pdf(summary, lang=("ar" if report_lang=="Arabic" else "en"), amiri_path=amiri)
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

# ---------- END ----------
