# app.py
"""
NeuroEarly Pro — Clinical Edition (final)
- Default language: English
- Arabic support via arabic_reshaper + python-bidi (if installed)
- Multi-EDF upload + comparison
- PHQ-9 (corrected Q3/Q5/Q8), AD8
- Clinical context (labs, meds, comorbidities)
- Topography maps (matplotlib if available) or SVG placeholders
- Connectivity (coherence or mne_connectivity if available)
- Explainable AI: shap_summary.json support, fallback to model.feature_importances_
- PDF generation with reportlab (Amiri font if provided) and branded footer/logo
"""
import os
import io
import json
import tempfile
import traceback
import datetime
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="NeuroEarly Pro – AI EEG Assistant", layout="wide")

# Optional heavy imports
HAS_MNE = False
HAS_MNE_CONN = False
HAS_PYEDF = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_ARABIC_TOOLS = False
HAS_SHAP = False

try:
    import mne
    HAS_MNE = True
    try:
        import mne_connectivity
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

from scipy.signal import welch, butter, filtfilt, iirnotch, coherence
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None

try:
    import joblib
except Exception:
    joblib = None

# Assets and files
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)
LOGO_PATH = ASSETS_DIR / "goldenbird_logo.png"
TOPO_PLACEHOLDER = ASSETS_DIR / "topo_placeholder.svg"
CONN_PLACEHOLDER = ASSETS_DIR / "conn_placeholder.svg"
SHAP_JSON = Path("shap_summary.json")
MODEL_DEP = Path("model_depression.pkl")
MODEL_AD = Path("model_alzheimer.pkl")
AMIRI_TTF = Path("Amiri-Regular.ttf")

# Bands
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
DEFAULT_SF = 256.0

# Normative ranges (example - tune later)
NORM_RANGES = {
    "theta_alpha_ratio": {"healthy_low": 0.0, "healthy_high": 1.1, "at_risk_low": 1.1, "at_risk_high": 1.4},
    "alpha_asym_F3_F4": {"healthy_low": -0.05, "healthy_high": 0.05, "at_risk_low": -0.2, "at_risk_high": -0.05}
}

# ---------------- Helpers ----------------
def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e):
    tb = traceback.format_exc()
    st.error("Internal error — see logs")
    st.code(tb)
    print(tb)

def reshape_ar(text: str) -> str:
    if not text:
        return ""
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def format_for_pdf_value(v):
    try:
        if v is None:
            return "N/A"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        if isinstance(v, dict):
            kvs = []
            for kk, vv in v.items():
                if isinstance(vv, (int, float, np.floating, np.integer)):
                    kvs.append(f"{kk}={vv:.2f}")
                else:
                    kvs.append(f"{kk}={str(vv)}")
            return ", ".join(kvs)
        if isinstance(v, (list, tuple)):
            return ", ".join([format_for_pdf_value(x) for x in v])
        return str(v)
    except Exception:
        return str(v)

# ---------------- EDF IO ----------------
def save_tmp_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path):
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return {"backend": "mne", "raw": raw, "data": raw.get_data(), "ch_names": raw.ch_names, "sfreq": raw.info.get("sfreq", None)}
    elif HAS_PYEDF:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        chs = f.getSignalLabels()
        try:
            sf = f.getSampleFrequency(0)
        except Exception:
            sf = None
        sigs = [f.readSignal(i).astype(np.float64) for i in range(n)]
        f._close()
        data = np.vstack(sigs)
        return {"backend": "pyedflib", "raw": None, "data": data, "ch_names": chs, "sfreq": sf}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ---------------- Signal processing ----------------
def notch_filter(sig, sf, freq=50.0, Q=30.0):
    if sf is None or sf <= 0:
        return sig
    b, a = iirnotch(freq, Q, sf)
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig, sf, low=0.5, high=45.0, order=4):
    if sf is None or sf <= 0:
        return sig
    ny = 0.5 * sf
    low_n = max(low / ny, 1e-6)
    high_n = min(high / ny, 0.999)
    b, a = butter(order, [low_n, high_n], btype='band')
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def preprocess_data(raw_data, sf, do_notch=True):
    cleaned = np.zeros_like(raw_data)
    for i in range(raw_data.shape[0]):
        s = raw_data[i].astype(np.float64)
        if do_notch:
            s = notch_filter(s, sf)
        s = bandpass_filter(s, sf)
        cleaned[i] = s
    return cleaned

# ---------------- PSD & features ----------------
def compute_psd_bands(data, sf, nperseg=1024):
    rows = []
    for ch in range(data.shape[0]):
        sig = data[ch]
        try:
            freqs, pxx = welch(sig, fs=sf, nperseg=min(nperseg, max(256, len(sig))))
        except Exception:
            freqs = np.array([]); pxx = np.array([])
        total = float(np.trapz(pxx, freqs)) if freqs.size > 0 else 0.0
        row = {"channel_idx": ch}
        for band, (lo, hi) in BANDS.items():
            if freqs.size == 0:
                abs_p = 0.0
            else:
                mask = (freqs >= lo) & (freqs <= hi)
                abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum() > 0 else 0.0
            rel = float(abs_p / total) if total > 0 else 0.0
            row[f"{band}_abs"] = abs_p
            row[f"{band}_rel"] = rel
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_bands(df_bands, ch_names=None):
    if df_bands.empty:
        return {}
    out = {}
    for band in BANDS.keys():
        out[f"{band.lower()}_abs_mean"] = float(df_bands[f"{band}_abs"].mean())
        out[f"{band.lower()}_rel_mean"] = float(df_bands[f"{band}_rel"].mean())
    out["theta_alpha_ratio"] = out.get("theta_rel_mean", 0) / (out.get("alpha_rel_mean", 1e-9))
    out["theta_beta_ratio"] = out.get("theta_rel_mean", 0) / (out.get("beta_rel_mean", 1e-9))
    out["beta_alpha_ratio"] = out.get("beta_rel_mean", 0) / (out.get("alpha_rel_mean", 1e-9))
    # alpha asymmetry F3-F4 best-effort
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def find(token):
                for i, n in enumerate(names):
                    if token in n:
                        return i
                return None
            i3 = find("F3"); i4 = find("F4")
            if i3 is not None and i4 is not None:
                a3 = df_bands.loc[df_bands['channel_idx'] == i3, 'Alpha_rel'].values
                a4 = df_bands.loc[df_bands['channel_idx'] == i4, 'Alpha_rel'].values
                if a3.size > 0 and a4.size > 0:
                    out["alpha_asym_F3_F4"] = float(a3[0] - a4[0])
        except Exception:
            out["alpha_asym_F3_F4"] = 0.0
    return out

# ---------------- Topomap generation ----------------
def generate_topomap_image(band_vals, ch_names=None, band_name="Alpha"):
    if band_vals is None:
        return None
    if not HAS_MATPLOTLIB or griddata is None:
        return str(TOPO_PLACEHOLDER) if TOPO_PLACEHOLDER.exists() else None
    try:
        coords = []
        labels = []
        if ch_names:
            names = [n.upper() for n in ch_names]
            approx = {
                "FP1":(-0.3,0.9),"FP2":(0.3,0.9),"F3":(-0.5,0.5),"F4":(0.5,0.5),
                "F7":(-0.8,0.2),"F8":(0.8,0.2),"C3":(-0.5,0.0),"C4":(0.5,0.0),
                "P3":(-0.5,-0.5),"P4":(0.5,-0.5),"O1":(-0.3,-0.9),"O2":(0.3,-0.9)
            }
            for v in names:
                placed = False
                for k, p in approx.items():
                    if k in v:
                        coords.append(p); labels.append(v); placed = True; break
                if not placed:
                    coords.append((np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9)))
                    labels.append(v)
        else:
            nch = len(band_vals)
            thetas = np.linspace(0, 2*np.pi, nch, endpoint=False)
            coords = [(0.8*np.sin(t), 0.8*np.cos(t)) for t in thetas]
            labels = [f"ch{i}" for i in range(len(coords))]
        xs = np.array([c[0] for c in coords]); ys = np.array([c[1] for c in coords])
        vals = np.array(band_vals[:len(coords)])
        xi = np.linspace(-1.0, 1.0, 160); yi = np.linspace(-1.0, 1.0, 160)
        XI, YI = np.meshgrid(xi, yi)
        Z = griddata((xs, ys), vals, (XI, YI), method='cubic', fill_value=np.nan)
        fig = plt.figure(figsize=(4,4), dpi=120)
        ax = fig.add_subplot(111)
        cmap = cm.get_cmap('RdBu_r')
        im = ax.imshow(Z, origin='lower', extent=[-1,1,-1,1], cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{band_name} topography", fontsize=9)
        circle = plt.Circle((0,0), 0.95, color='k', fill=False, linewidth=1)
        ax.add_artist(circle)
        ax.scatter(xs, ys, s=20, c='k')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return str(TOPO_PLACEHOLDER) if TOPO_PLACEHOLDER.exists() else None

# ---------------- Connectivity (coherence / mne-connectivity) ----------------
def compute_connectivity(data, sf, ch_names=None, band=(8,13)):
    try:
        nchan = data.shape[0]
        lo, hi = band
        if HAS_MNE_CONN and HAS_MNE:
            # create raw and compute spectral connectivity using mne_connectivity or mne.connectivity
            info = mne.create_info(ch_names=ch_names if ch_names else [f"ch{i}" for i in range(nchan)], sfreq=sf, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            try:
                # use mne.connectivity.spectral_connectivity (if available)
                con_methods = ["wpli", "pli", "coh"]
                try:
                    from mne_connectivity import spectral_connectivity
                    con = spectral_connectivity(raw, method=con_methods, sfreq=sf, fmin=lo, fmax=hi, mt_adaptive=False)
                    # simplify: pick 'coh' if exists otherwise mean across methods
                    if "coh" in con:
                        mat = con["coh"].mean(axis=0)
                    else:
                        mats = [con[m].mean(axis=0) for m in con_methods if m in con]
                        mat = np.mean(mats, axis=0) if mats else np.eye(nchan)
                except Exception:
                    # fallback to mne.connectivity.spectral_connectivity in newer API
                    from mne.connectivity import spectral_connectivity as sc
                    con = sc(raw, method='coh', sfreq=sf, fmin=lo, fmax=hi)
                    mat = con.get_data(output='dense') if hasattr(con, 'get_data') else np.eye(nchan)
                narrative = f"Connectivity computed via mne_connectivity in {lo}-{hi} Hz."
                return mat, narrative
            except Exception:
                pass
        # fallback: pairwise coherence via scipy
        n = data.shape[0]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                try:
                    f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, len(data[i]))))
                    mask = (f >= lo) & (f <= hi)
                    val = float(np.mean(Cxy[mask])) if mask.sum() > 0 else 0.0
                except Exception:
                    val = 0.0
                mat[i, j] = val; mat[j, i] = val
        narrative = f"Connectivity (mean coherence) in {lo}-{hi} Hz computed via scipy.signal.coherence."
        return mat, narrative
    except Exception as e:
        return None, f"Connectivity computation failed: {str(e)}"

# ---------------- Normative comparison plot ----------------
def plot_norm_comparison(metric_key, patient_value, title=None):
    rng = NORM_RANGES.get(metric_key)
    if not HAS_MATPLOTLIB:
        return None
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=120)
    # background: healthy (light), at-risk (red transparent)
    if rng:
        healthy_low, healthy_high = rng["healthy_low"], rng["healthy_high"]
        at_low, at_high = rng["at_risk_low"], rng["at_risk_high"]
        # draw ranges as horizontal bars (vertical orientation)
        ax.add_patch(patches.Rectangle((0, healthy_low), 0.5, healthy_high - healthy_low, facecolor='white', edgecolor='gray', alpha=0.7))
        ax.add_patch(patches.Rectangle((0, at_low), 0.5, at_high - at_low, facecolor='red', edgecolor='red', alpha=0.2))
    ax.bar(0.25, patient_value, width=0.2, color='#0b3d91')
    ax.set_xlim(0, 1)
    if rng:
        ymin = min(healthy_low, at_low, patient_value) - 0.2 * abs(patient_value if patient_value != 0 else 1)
        ymax = max(healthy_high, at_high, patient_value) + 0.2 * abs(patient_value if patient_value != 0 else 1)
        ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_ylabel(metric_key)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ---------------- SHAP load ----------------
def load_shap_summary():
    if SHAP_JSON.exists():
        try:
            with open(SHAP_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------------- PDF helpers ----------------
def register_amiri(ttf_path=None):
    if not HAS_REPORTLAB:
        return "Helvetica"
    try:
        if ttf_path and Path(ttf_path).exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(ttf_path)))
            return "Amiri"
        if AMIRI_TTF.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
            return "Amiri"
    except Exception:
        pass
    return "Helvetica"

def generate_pdf_report(summary, lang='en', amiri_path=None):
    if not HAS_REPORTLAB:
        return json.dumps(summary, indent=2, ensure_ascii=False).encode('utf-8')
    font = register_amiri(amiri_path)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    m = 36
    x = m
    y = H - m

    # Title
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font, 16)
    if lang == 'en':
        c.drawCentredString(W / 2, y, title_en)
    else:
        c.drawCentredString(W / 2, y, reshape_ar(title_ar))
    y -= 28

    # Final ML Risk prominently
    ml = summary.get("ml_risk", None)
    cat = summary.get("risk_category", "-")
    c.setFont(font, 12)
    if ml is not None:
        if lang == 'en':
            c.drawString(x, y, f"Final ML Risk Score: {format_for_pdf_value(ml)}%    Category: {cat}")
        else:
            c.drawRightString(W - m, y, reshape_ar(f"معدل الخطر ML النهائي: {format_for_pdf_value(ml)}%    الفئة: {cat}"))
        y -= 18

    # Patient meta
    c.setFont(font, 10)
    p = summary.get("patient", {})
    if lang == 'en':
        c.drawString(x, y, f"Patient: {p.get('name','-')}    ID: {p.get('id','-')}    DOB: {p.get('dob','-')}")
    else:
        c.drawRightString(W - m, y, reshape_ar(f"المريض: {p.get('name','-')}    المعرف: {p.get('id','-')}    الميلاد: {p.get('dob','-')}"))
    y -= 14

    # QEEG interpretation
    q = summary.get("qeegh", "-")
    if lang == 'en':
        c.drawString(x, y, f"QEEG Interpretation: {q}")
    else:
        c.drawRightString(W - m, y, reshape_ar("تفسير QEEG: " + str(q)))
    y -= 16

    # Clinical context
    clinical = summary.get("clinical", {})
    if lang == 'en':
        c.drawString(x, y, "Clinical context:"); y -= 12
        c.drawString(x + 6, y, f"Labs: {', '.join(clinical.get('labs', [])) if clinical.get('labs') else 'None'}"); y -= 10
        c.drawString(x + 6, y, f"Medications: {clinical.get('meds', 'None')}"); y -= 10
        c.drawString(x + 6, y, f"Comorbidities: {clinical.get('conditions', 'None')}"); y -= 12
    else:
        c.drawRightString(W - m, y, reshape_ar("السياق السريري:")); y -= 12
        c.drawRightString(W - m, y, reshape_ar(f"التحاليل: {', '.join(clinical.get('labs', [])) if clinical.get('labs') else 'لا'}")); y -= 10
        c.drawRightString(W - m, y, reshape_ar(f"الأدوية: {clinical.get('meds', 'لا')}")); y -= 10
        c.drawRightString(W - m, y, reshape_ar(f"الأمراض المصاحبة: {clinical.get('conditions', 'لا')}")); y -= 12

    # For each file: band table, feature table, topomaps, connectivity
    files = summary.get("files", [])
    for idx, fdata in enumerate(files):
        if y < 220:
            c.showPage()
            y = H - m
        fname = fdata.get("filename", f"File_{idx+1}")
        agg = fdata.get("agg_features", {})
        if lang == 'en':
            c.drawString(x, y, f"[{idx+1}] File: {fname}"); y -= 12
        else:
            c.drawRightString(W - m, y, reshape_ar(f"[{idx+1}] الملف: {fname}")); y -= 12

        # band table columns
        col1 = x + 6; col2 = x + 160; col3 = x + 300
        if lang == 'en':
            c.drawString(col1, y, "Band"); c.drawString(col2, y, "Abs_mean"); c.drawString(col3, y, "Rel_mean")
        else:
            c.drawRightString(W - m, y, reshape_ar("التردد")); c.drawRightString(W - m - 120, y, reshape_ar("المطلق")); c.drawRightString(W - m - 30, y, reshape_ar("النسبي"))
        y -= 12
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            a = agg.get(f"{band.lower()}_abs_mean", 0.0)
            r = agg.get(f"{band.lower()}_rel_mean", 0.0)
            if lang == 'en':
                c.drawString(col1, y, f"{band}"); c.drawString(col2, y, f"{a:>12.4f}"); c.drawString(col3, y, f"{r:>12.4f}")
            else:
                c.drawRightString(W - m, y, reshape_ar(f"{band}")); c.drawRightString(W - m - 120, y, format_for_pdf_value(a)); c.drawRightString(W - m - 30, y, format_for_pdf_value(r))
            y -= 10
        y -= 8

        # feature table aligned
        c.setFont(font, 9)
        if lang == 'en':
            c.drawString(col1, y, "Feature"); c.drawString(col2, y, "Value"); y -= 10
            feat_list = ["theta_alpha_ratio", "alpha_asym_F3_F4", "theta_beta_ratio", "beta_alpha_ratio", "gamma_rel_mean"]
            for fn in feat_list:
                val = agg.get(fn, None)
                c.drawString(col1, y, f"{fn}") ; c.drawString(col2, y, format_for_pdf_value(val)); y -= 9
        else:
            c.drawRightString(W - m, y, reshape_ar("الميزة")); c.drawRightString(W - m - 120, y, reshape_ar("القيمة")); y -= 10
            feat_list = ["theta_alpha_ratio", "alpha_asym_F3_F4", "theta_beta_ratio", "beta_alpha_ratio", "gamma_rel_mean"]
            for fn in feat_list:
                val = agg.get(fn, None)
                c.drawRightString(W - m, y, reshape_ar(f"{fn}")); c.drawRightString(W - m - 120, y, reshape_ar(format_for_pdf_value(val))); y -= 9
        y -= 12

        # topomaps
        topo_imgs = fdata.get("topo_images", {})
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
                    c.setFont(font, 8); c.drawString(xi, yi - 10, band)
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

        # connectivity
        conn_img = fdata.get("connectivity_image", None)
        if conn_img:
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(io.BytesIO(conn_img)) if isinstance(conn_img, (bytes, bytearray)) else ImageReader(str(conn_img))
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
        c.showPage(); y = H - m
    if lang == 'en':
        c.drawString(x, y, "Explainable AI — Top contributors:"); y -= 12
    else:
        c.drawRightString(W - m, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:")); y -= 12
    xai = summary.get("xai", None)
    if xai:
        if isinstance(xai, dict):
            for group_key, group_val in xai.items():
                if isinstance(group_val, dict):
                    if lang == 'en':
                        c.drawString(x + 6, y, f"{group_key}:"); y -= 10
                    else:
                        c.drawRightString(W - m, y, reshape_ar(f"{group_key}:")); y -= 10
                    for kk, vv in sorted(group_val.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int, float)) else 0)[:20]:
                        if lang == 'en':
                            c.drawString(x + 10, y, f"{kk}: {format_for_pdf_value(vv)}"); y -= 9
                        else:
                            c.drawRightString(W - m, y, reshape_ar(f"{kk}: {format_for_pdf_value(vv)}")); y -= 9
                else:
                    if lang == 'en':
                        c.drawString(x + 6, y, f"{group_key}: {format_for_pdf_value(group_val)}"); y -= 10
                    else:
                        c.drawRightString(W - m, y, reshape_ar(f"{group_key}: {format_for_pdf_value(group_val)}")); y -= 10
    else:
        if lang == 'en':
            c.drawString(x + 6, y, "XAI not available (no shap_summary.json or model)."); y -= 10
        else:
            c.drawRightString(W - m, y, reshape_ar("لا توجد بيانات XAI (لا يوجد shap_summary.json أو نموذج).")); y -= 10

    y -= 8
    # Recommendations
    if lang == 'en':
        c.drawString(x, y, "Structured Clinical Recommendations:"); y -= 12
    else:
        c.drawRightString(W - m, y, reshape_ar("التوصيات السريرية المنظمة:")); y -= 12
    for r in summary.get("recommendations", []):
        if lang == 'en':
            c.drawString(x + 6, y, "- " + r); y -= 10
        else:
            c.drawRightString(W - m, y, reshape_ar("- " + r)); y -= 10

    # Footer: logo + branding + disclaimer
    c.setFont(font, 8)
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
    if lang == 'en':
        c.drawCentredString(W / 2, 30, footer_en); c.drawCentredString(W / 2, 18, disc_en)
    else:
        c.drawCentredString(W / 2, 30, footer_ar); c.drawCentredString(W / 2, 18, disc_ar)

    c.save()
    buf.seek(0)
    return buf.read()

# ---------------- UI layout ----------------
st.markdown("""
<style>
.block-container { max-width:1200px; }
.header { background: linear-gradient(90deg,#0b3d91,#2451a6); color:white; padding:14px; border-radius:8px; }
.small { color:#cbd5e1; font-size:13px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro – AI EEG Assistant</h2><div class='small'>EEG / QEEG + Explainable AI — Clinical support</div></div>", unsafe_allow_html=True)
with col2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140, use_container_width=False)
    else:
        st.markdown("<div style='text-align:right;font-weight:600'>Golden Bird LLC</div>", unsafe_allow_html=True)

# Sidebar: settings & patient
with st.sidebar:
    st.header("Settings & Patient")
    lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)
    st.markdown("---")
    st.subheader("Patient information")
    patient_name = st.text_input("Name / الاسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1940,1,1), max_value=date.today())
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

# Main: upload EDF(s)
st.markdown("### 1) Upload EDF file(s) (.edf) — you can upload multiple files to compare")
uploads = st.file_uploader("Drag & drop EDF files here", type=["edf"], accept_multiple_files=True)

# PHQ-9 (corrected Q3,Q5,Q8)
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
    label_en = f"Q{i}. {PHQ_EN[i-1]}"
    if lang == 'ar':
        display_label = reshape_ar(PHQ_AR[i-1])
    else:
        display_label = label_en

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

    if lang == 'en':
        ans = st.radio(label_en, options=opts_display, index=0, key=f"phq_{i}")
    else:
        st.markdown(f"**{display_label}**")
        ans = st.radio("", options=opts_display, index=0, key=f"phq_{i}_ar")

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
        val = st.radio(label, options=[0,1], index=0, key=f"ad8_{i}")
    else:
        label = reshape_ar(AD8_AR[i-1]); st.markdown(f"**{label}**"); val = st.radio("", options=[0,1], index=0, key=f"ad8_{i}_ar", horizontal=True)
    ad8_answers[f"A{i}"] = int(val)

ad8_total = sum(ad8_answers.values())
st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

# ---------------- Processing options ----------------
st.markdown("---")
st.header("Processing & Visualization Options")
use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
do_topomap = st.checkbox("Generate topography maps (if matplotlib available)", value=True)
do_connectivity = st.checkbox("Compute connectivity (Alpha band coherence)", value=True)
run_models = st.checkbox("Run ML models if provided (model_depression.pkl / model_alzheimer.pkl)", value=False)

# ---------------- Process uploaded EDF files ----------------
results = []
if uploads:
    for up in uploads:
        st.write(f"Processing {up.name} ...")
        try:
            tmp = save_tmp_upload(up)
            edf = read_edf(tmp)
            data = edf["data"]; sf = edf.get("sfreq") or DEFAULT_SF
            st.success(f"Loaded: backend={edf['backend']}   channels={data.shape[0]}   sfreq={sf}")
            cleaned = preprocess_data(data, sf, do_notch=use_notch)
            dfbands = compute_psd_bands(cleaned, sf)
            st.dataframe(dfbands.head(10))
            agg = aggregate_bands(dfbands, ch_names=edf.get("ch_names"))
            st.write("Aggregated features:", agg)
            # add gamma_rel_mean to agg to support listing
            if "gamma_rel_mean" not in agg:
                agg["gamma_rel_mean"] = float(dfbands["Gamma_rel"].mean()) if not dfbands.empty else 0.0

            topo_images = {}
            if do_topomap:
                for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                    vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(data.shape[0])
                    img = generate_topomap_image(vals, ch_names=edf.get("ch_names"), band_name=band)
                    topo_images[band] = img
                    if isinstance(img, (bytes, bytearray)):
                        st.image(img, caption=f"{band} topomap", use_container_width=False)
                    elif isinstance(img, str) and Path(img).exists():
                        st.image(str(img), caption=f"{band} topomap", use_container_width=False)
                    else:
                        if TOPO_PLACEHOLDER.exists():
                            st.image(str(TOPO_PLACEHOLDER), caption=f"{band} topomap (placeholder)", use_container_width=False)

            conn_img = None; conn_narr = None; conn_mat = None
            if do_connectivity:
                conn_mat, conn_narr = compute_connectivity(cleaned, sf, ch_names=edf.get("ch_names"), band=BANDS["Alpha"])
                if conn_mat is not None and HAS_MATPLOTLIB:
                    fig = plt.figure(figsize=(4,3)); plt.imshow(conn_mat, cmap='viridis'); plt.colorbar(); plt.title("Connectivity (Alpha)"); buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    conn_img = buf.getvalue()
                    st.image(conn_img, caption="Functional disconnection (Alpha coherence)", use_container_width=False)
                else:
                    if CONN_PLACEHOLDER.exists():
                        st.image(str(CONN_PLACEHOLDER), caption="Connectivity placeholder", use_container_width=False)

            results.append({
                "filename": up.name,
                "agg_features": agg,
                "df_bands": dfbands,
                "topo_images": topo_images,
                "connectivity_image": conn_img,
                "connectivity_matrix": conn_mat,
                "connectivity_narrative": conn_narr
            })
        except Exception as e:
            st.error(f"Failed processing {up.name}: {e}")
            _trace(e)

# ---------------- Build summary ----------------
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

# Heuristic QEEG interpretation
if results:
    agg0 = results[0].get("agg_features", {})
    ta = agg0.get("theta_alpha_ratio", None)
    if ta is not None:
        if ta > 1.4:
            summary["qeegh"] = "Elevated Theta/Alpha ratio consistent with early cognitive slowing."
        elif ta > 1.1:
            summary["qeegh"] = "Mild elevation of Theta/Alpha; correlate clinically."
        else:
            summary["qeegh"] = "Theta/Alpha within expected range."
    if results[0].get("connectivity_narrative"):
        summary["connectivity"] = results[0].get("connectivity_narrative")

# ---------------- XAI loading ----------------
shap_json = load_shap_summary()
if shap_json:
    summary["xai"] = shap_json
else:
    # fallback: try to read feature_importances_ from pickled models
    fe = {}
    try:
        if joblib and MODEL_DEP.exists():
            md = joblib.load(str(MODEL_DEP))
            if hasattr(md, "feature_importances_"):
                names = list(md.feature_names_in_) if hasattr(md, "feature_names_in_") else [f"f{i}" for i in range(len(md.feature_importances_))]
                fe["depression_global"] = dict(zip(names, md.feature_importances_.tolist()))
        if joblib and MODEL_AD.exists():
            ma = joblib.load(str(MODEL_AD))
            if hasattr(ma, "feature_importances_"):
                names = list(ma.feature_names_in_) if hasattr(ma, "feature_names_in_") else [f"f{i}" for i in range(len(ma.feature_importances_))]
                fe["alzheimers_global"] = dict(zip(names, ma.feature_importances_.tolist()))
    except Exception:
        pass
    if fe:
        summary["xai"] = fe

# ---------------- ML predictions & Final ML Risk ----------------
mlrisk = None
if run_models and joblib and (MODEL_DEP.exists() or MODEL_AD.exists()) and results:
    try:
        Xdf = pd.DataFrame([r.get("agg_features", {}) for r in results]).fillna(0)
        preds = []
        if MODEL_DEP.exists():
            mdep = joblib.load(str(MODEL_DEP))
            p = mdep.predict_proba(Xdf)[:,1] if hasattr(mdep, "predict_proba") else mdep.predict(Xdf)
            summary.setdefault("predictions", {})["depression_prob"] = [float(x) for x in p]
            preds.append(np.mean(p))
        if MODEL_AD.exists():
            mad = joblib.load(str(MODEL_AD))
            p2 = mad.predict_proba(Xdf)[:,1] if hasattr(mad, "predict_proba") else mad.predict(Xdf)
            summary.setdefault("predictions", {})["alzheimers_prob"] = [float(x) for x in p2]
            preds.append(np.mean(p2))
        if preds:
            mlrisk = float(np.mean(preds)) * 100.0
    except Exception as e:
        st.warning("Model prediction failed: " + str(e))

# heuristic fallback if models missing
if mlrisk is None:
    ta_val = results[0]["agg_features"].get("theta_alpha_ratio", 1.0) if results else 1.0
    phq_norm = phq_total / 27.0
    ad8_norm = ad8_total / 8.0
    ta_norm = min(1.0, ta_val / 1.4)
    mlrisk = min(100.0, (ta_norm * 0.5 + phq_norm * 0.3 + ad8_norm * 0.2) * 100.0)

summary["ml_risk"] = mlrisk
if mlrisk >= 50:
    summary["risk_category"] = "High"
elif mlrisk >= 25:
    summary["risk_category"] = "Moderate"
else:
    summary["risk_category"] = "Low"

# ---------------- Recommendations ----------------
if summary["phq9"]["total"] >= 10:
    summary["recommendations"].append("PHQ-9 suggests moderate/severe depression — consider psychiatric referral and treatment planning.")
if summary["ad8"]["total"] >= 2 or (results and results[0]["agg_features"].get("theta_alpha_ratio", 0) > 1.4):
    summary["recommendations"].append("AD8 elevated or Theta/Alpha increased — consider neurocognitive testing and neuroimaging (MRI/FDG-PET).")
summary["recommendations"].append("Correlate QEEG/connectivity findings with PHQ-9, AD8 and clinical interview.")
summary["recommendations"].append("Review medications that may affect EEG.")
if not summary["recommendations"]:
    summary["recommendations"].append("Clinical follow-up and re-evaluation in 3–6 months.")

# ---------------- Display Executive Summary & Normative charts ----------------
st.markdown("---")
st.subheader("Executive Summary")
st.write("Final ML Risk Score (first item):")
st.metric(label="Final ML Risk Score", value=f"{summary['ml_risk']:.1f}%", delta=summary.get("risk_category", "-"))
st.write("Brief QEEG interpretation:"); st.write(summary.get("qeegh", "-"))

if results:
    agg0 = results[0]["agg_features"]
    tar = agg0.get("theta_alpha_ratio", None)
    asym = agg0.get("alpha_asym_F3_F4", None)
    if tar is not None:
        img = plot_norm_comparison("theta_alpha_ratio", tar, title="Theta/Alpha vs Norm")
        if img:
            st.image(img, caption="Theta/Alpha comparison", use_container_width=False)
    if asym is not None:
        img2 = plot_norm_comparison("alpha_asym_F3_F4", asym, title="Alpha Asymmetry (F3-F4)")
        if img2:
            st.image(img2, caption="Alpha Asymmetry comparison", use_container_width=False)

# XAI display
st.markdown("---")
st.header("Explainable AI (XAI)")
if summary.get("xai"):
    st.write("Top contributors (from shap_summary.json or model importances):")
    st.json(summary["xai"])
else:
    st.info("XAI data not available. Upload shap_summary.json or model pickle to enable.")

# ---------------- Generate PDF ----------------
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
