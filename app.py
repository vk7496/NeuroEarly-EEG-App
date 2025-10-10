# app.py
"""
NeuroEarly Pro — Streamlit Clinical XAI (final)
- Default language: English
- Arabic support via arabic_reshaper + python-bidi (if installed)
- Multi-EDF upload + comparison
- PHQ-9 (with corrected Q3/Q5/Q8), AD8
- Clinical context (labs, meds, comorbidities)
- Topography maps (matplotlib if available) or SVG placeholders
- Functional disconnection (placeholder or computed)
- XAI: shap_summary.json support, fallback to model.feature_importances_
- PDF generation with ReportLab (Amiri font if provided)
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro — Clinical XAI", layout="wide")

import os, io, json, tempfile, traceback, datetime
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

# Optional heavy imports (safe)
HAS_MNE = False
HAS_PYEDF = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_ARABIC_TOOLS = False
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
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
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

from scipy.signal import welch, butter, filtfilt, iirnotch
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None

# joblib for loading models
try:
    import joblib
except Exception:
    joblib = None

# Constants and asset paths
ASSETS_DIR = Path("./assets")
ASSETS_DIR.mkdir(exist_ok=True)
LOGO_SVG = ASSETS_DIR / "GoldenBird_logo.svg"
TOPO_PLACEHOLDER = ASSETS_DIR / "topo_placeholder.svg"
CONN_PLACEHOLDER = ASSETS_DIR / "conn_placeholder.svg"
SHAP_JSON = Path("shap_summary.json")
MODEL_DEP = Path("model_depression.pkl")
MODEL_AD = Path("model_alzheimer.pkl")
AMIRI_TTF = Path("Amiri-Regular.ttf")

BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
DEFAULT_SF = 256.0

# ----------------- Helpers -----------------
def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e):
    tb = traceback.format_exc()
    st.error("Internal error — see details")
    st.code(tb)
    print(tb)

def reshape_ar(text: str) -> str:
    """Apply arabic_reshaper + bidi if available; otherwise return text."""
    if not text:
        return ""
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def format_for_pdf_value(v):
    """Return a printable string for different types (float, int, dict, list, None)."""
    try:
        if v is None:
            return "N/A"
        if isinstance(v, (int, float, np.floating, np.integer)):
            return f"{float(v):.4f}" if isinstance(v, float) or isinstance(v, np.floating) else f"{int(v)}"
        if isinstance(v, dict):
            # join key=val pairs (numbers formatted)
            kvs = []
            for kk, vv in v.items():
                if isinstance(vv, (int, float, np.floating, np.integer)):
                    kvs.append(f"{kk}={float(vv):.2f}")
                else:
                    kvs.append(f"{kk}={str(vv)}")
            return ", ".join(kvs)
        if isinstance(v, (list, tuple)):
            return ", ".join([format_for_pdf_value(x) for x in v])
        return str(v)
    except Exception:
        return str(v)

# ----------------- EDF IO -----------------
def save_tmp_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path):
    """Return dict with backend, data (channels x samples), ch_names, sfreq"""
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return {"backend":"mne","raw":raw,"data":raw.get_data(),"ch_names":raw.ch_names,"sfreq":raw.info.get("sfreq",None)}
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
        return {"backend":"pyedflib","raw":None,"data":data,"ch_names":chs,"sfreq":sf}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ----------------- Signal processing -----------------
def notch_filter(sig, sf, freq=50.0, Q=30.0):
    if sf is None or sf <= 0:
        return sig
    b,a = iirnotch(freq, Q, sf)
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def bandpass_filter(sig, sf, low=0.5, high=45.0, order=4):
    if sf is None or sf <= 0:
        return sig
    ny = 0.5*sf
    low_n = max(low/ny, 1e-6)
    high_n = min(high/ny, 0.999)
    b,a = butter(order, [low_n, high_n], btype='band')
    try:
        return filtfilt(b,a,sig)
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

# ----------------- PSD / features -----------------
def compute_psd_bands(data, sf, nperseg=1024):
    rows = []
    for ch in range(data.shape[0]):
        sig = data[ch]
        try:
            freqs, pxx = welch(sig, fs=sf, nperseg=min(nperseg, max(256, len(sig))))
        except Exception:
            freqs = np.array([]); pxx = np.array([])
        total = float(np.trapz(pxx, freqs)) if freqs.size>0 else 0.0
        row = {"channel_idx": ch}
        for band,(lo,hi) in BANDS.items():
            if freqs.size==0:
                abs_p = 0.0
            else:
                mask = (freqs>=lo)&(freqs<=hi)
                abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum()>0 else 0.0
            rel = float(abs_p/total) if total>0 else 0.0
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
    out["theta_alpha_ratio"] = out.get("theta_rel_mean",0) / (out.get("alpha_rel_mean",1e-9))
    out["theta_beta_ratio"] = out.get("theta_rel_mean",0) / (out.get("beta_rel_mean",1e-9))
    out["beta_alpha_ratio"] = out.get("beta_rel_mean",0) / (out.get("alpha_rel_mean",1e-9))
    # alpha asymmetry F3-F4 best effort
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def find(token):
                for i,n in enumerate(names):
                    if token in n:
                        return i
                return None
            i3 = find("F3"); i4 = find("F4")
            if i3 is not None and i4 is not None:
                a3 = df_bands.loc[df_bands['channel_idx']==i3,'Alpha_rel'].values
                a4 = df_bands.loc[df_bands['channel_idx']==i4,'Alpha_rel'].values
                if a3.size>0 and a4.size>0:
                    out["alpha_asym_F3_F4"] = float(a3[0]-a4[0])
        except Exception:
            out["alpha_asym_F3_F4"] = 0.0
    return out

# ----------------- Topomap image (cloud-safe) -----------------
def generate_topomap_image(band_vals, ch_names=None, band_name="Alpha"):
    """
    band_vals: 1D array-like of length = n_channels
    Returns PNG bytes if matplotlib available else path to placeholder or None
    """
    if band_vals is None:
        return None
    # If matplotlib not available or no griddata, fallback to placeholder SVG if present
    if not HAS_MATPLOTLIB or griddata is None:
        if TOPO_PLACEHOLDER.exists():
            return str(TOPO_PLACEHOLDER)
        return None
    try:
        # determine coords for channels (approximate mapping for common labels)
        coords = []
        labels = []
        if ch_names:
            names = [n.upper() for n in ch_names]
            approx = {
                "FP1":(-0.3,0.9),"FP2":(0.3,0.9),"F3":(-0.5,0.5),"F4":(0.5,0.5),
                "F7":(-0.8,0.2),"F8":(0.8,0.2),"C3":(-0.5,0.0),"C4":(0.5,0.0),
                "P3":(-0.5,-0.5),"P4":(0.5,-0.5),"O1":(-0.3,-0.9),"O2":(0.3,-0.9)
            }
            for n,v in enumerate(names):
                placed=False
                for k,p in approx.items():
                    if k in v:
                        coords.append(p); labels.append(v); placed=True; break
                if not placed:
                    coords.append((np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9)))
                    labels.append(v)
        else:
            nch = len(band_vals)
            thetas = np.linspace(0,2*np.pi,nch,endpoint=False)
            coords = [(0.8*np.sin(t), 0.8*np.cos(t)) for t in thetas]
            labels = [f"ch{i}" for i in range(len(coords))]
        xs = np.array([c[0] for c in coords]); ys = np.array([c[1] for c in coords])
        vals = np.array(band_vals[:len(coords)])
        # grid interpolation
        xi = np.linspace(-1.0,1.0,160); yi = np.linspace(-1.0,1.0,160)
        XI, YI = np.meshgrid(xi, yi)
        Z = griddata((xs,ys), vals, (XI, YI), method='cubic', fill_value=np.nan)
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
        if TOPO_PLACEHOLDER.exists():
            return str(TOPO_PLACEHOLDER)
        return None

# ----------------- Functional disconnection placeholder -----------------
def compute_connectivity_placeholder():
    """Return synthetic matrix and narrative"""
    mat = np.random.uniform(0.4,0.95,(10,10))
    for i in range(10):
        mat[i,i]=1.0
    # simulate reduced frontal-parietal coherence
    mat[1,7] *= 0.6; mat[7,1] *= 0.6
    narrative = "Functional disconnection: reduction in Alpha coherence between frontal and parietal regions (~15%)."
    return mat, narrative

# ----------------- SHAP loading -----------------
def load_shap_summary():
    if SHAP_JSON.exists():
        try:
            with open(SHAP_JSON,'r',encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ----------------- PDF generation -----------------
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

def generate_pdf_report(summary, lang='en', amiri_path=None, topo_images=None, conn_image=None):
    """
    topo_images: dict band->(PNG bytes or filepath or None)
    conn_image: PNG bytes or filepath or None
    """
    if not HAS_REPORTLAB:
        return json.dumps(summary, indent=2, ensure_ascii=False).encode('utf-8')
    font = register_amiri(amiri_path)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W,H = A4
    m = 36
    x = m
    y = H - m

    # Title
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font, 16)
    if lang == 'en':
        c.drawCentredString(W/2, y, title_en)
    else:
        c.drawCentredString(W/2, y, reshape_ar(title_ar))
    y -= 26

    # Patient meta & exec summary
    c.setFont(font, 10)
    p = summary.get("patient",{})
    if lang == 'en':
        c.drawString(x, y, f"Patient: {p.get('name','-')}   ID: {p.get('id','-')}   DOB: {p.get('dob','-')}")
    else:
        c.drawRightString(W-m, y, reshape_ar(f"المريض: {p.get('name','-')}   المعرف: {p.get('id','-')}   الميلاد: {p.get('dob','-')}"))
    y -= 14

    if summary.get("ml_risk") is not None:
        if lang == 'en':
            c.drawString(x, y, f"ML Risk Score: {format_for_pdf_value(summary.get('ml_risk'))}%    Risk: {summary.get('risk_category','-')}")
        else:
            c.drawRightString(W-m, y, reshape_ar(f"معدل الخطر ML: {format_for_pdf_value(summary.get('ml_risk'))}%    الفئة: {summary.get('risk_category','-')}"))
        y -= 14

    qeegh = summary.get("qeegh", "-")
    if lang == 'en':
        c.drawString(x, y, f"QEEG Interpretation: {qeegh}")
    else:
        c.drawRightString(W-m, y, reshape_ar("تفسير QEEG: " + str(qeegh)))
    y -= 18

    # Clinical context (labs/meds/conditions)
    clinical = summary.get("clinical", {})
    if lang == 'en':
        c.drawString(x, y, "Clinical context:")
    else:
        c.drawRightString(W-m, y, reshape_ar("السياق السريري:"))
    y -= 12
    if clinical:
        labs = clinical.get("labs", [])
        meds = clinical.get("meds", "")
        conds = clinical.get("conditions", "")
        if lang == 'en':
            c.drawString(x+6, y, f"Labs available: {', '.join(labs) if labs else 'None'}"); y -= 10
            c.drawString(x+6, y, f"Medications: {meds if meds else 'None'}"); y -= 10
            c.drawString(x+6, y, f"Comorbidities: {conds if conds else 'None'}"); y -= 12
        else:
            c.drawRightString(W-m, y, reshape_ar(f"التحاليل المتاحة: {', '.join(labs) if labs else 'لا'}")); y -= 10
            c.drawRightString(W-m, y, reshape_ar(f"الأدوية: {meds if meds else 'لا'}")); y -= 10
            c.drawRightString(W-m, y, reshape_ar(f"الأمراض المصاحبة: {conds if conds else 'لا'}")); y -= 12
    else:
        y -= 6

    # For each uploaded EDF: band table + feature table + topomap
    files = summary.get("files", [])
    if files:
        for idx, fdata in enumerate(files):
            if y < 200:
                c.showPage()
                y = H - m
            fname = fdata.get("filename", f"File_{idx+1}")
            agg = fdata.get("agg_features", {})
            if lang == 'en':
                c.setFont(font, 11)
                c.drawString(x, y, f"[{idx+1}] File: {fname}"); y -= 14
                c.setFont(font, 10)
                c.drawString(x+6, y, "Band    Absolute_mean    Relative_mean"); y -= 12
                for band in ["Delta","Theta","Alpha","Beta"]:
                    a = agg.get(f"{band.lower()}_abs_mean", 0.0)
                    r = agg.get(f"{band.lower()}_rel_mean", 0.0)
                    c.drawString(x+8, y, f"{band:<6} {a:>12.4f}    {r:>10.4f}"); y -= 10
            else:
                c.setFont(font, 11)
                c.drawRightString(W-m, y, reshape_ar(f"[{idx+1}] الملف: {fname}")); y -= 14
                c.setFont(font, 10)
                for band in ["Delta","Theta","Alpha","Beta"]:
                    a = agg.get(f"{band.lower()}_abs_mean", 0.0)
                    r = agg.get(f"{band.lower()}_rel_mean", 0.0)
                    c.drawRightString(W-m, y, reshape_ar(f"{band}    {a:.4f}    {r:.4f}")); y -= 10
            y -= 6

            # feature table (compact)
            feat_names = [
                "delta_abs_mean","theta_abs_mean","alpha_abs_mean","beta_abs_mean",
                "delta_rel_mean","theta_rel_mean","alpha_rel_mean","beta_rel_mean",
                "theta_beta_ratio","theta_alpha_ratio","beta_alpha_ratio","alpha_asym_F3_F4"
            ]
            if lang == 'en':
                c.drawString(x+6, y, "Features:"); y -= 10
                for fn in feat_names:
                    val = agg.get(fn, None)
                    c.drawString(x+10, y, f"{fn:<28} {format_for_pdf_value(val)}"); y -= 9
            else:
                c.drawRightString(W-m, y, reshape_ar("الميزات:")); y -= 10
                for fn in feat_names:
                    val = agg.get(fn, None)
                    c.drawRightString(W-m, y, reshape_ar(f"{fn}    {format_for_pdf_value(val)}")); y -= 9
            y -= 6

            # Topomap images for this file
            topo_imgs = fdata.get("topo_images", {})
            if topo_imgs:
                # show up to 4 topomaps row
                img_w = 110
                x0 = x
                colcount = 0
                for band, img in topo_imgs.items():
                    if img is None: continue
                    try:
                        from reportlab.lib.utils import ImageReader
                        ir = ImageReader(io.BytesIO(img)) if isinstance(img, (bytes,bytearray)) else ImageReader(str(img))
                        xi = x + (colcount % 4) * (img_w + 8)
                        yi = y - img_w
                        c.drawImage(ir, xi, yi, width=img_w, height=img_w, preserveAspectRatio=True, mask='auto')
                        c.setFont(font, 8)
                        c.drawString(xi, yi - 10, band)
                        colcount += 1
                        if colcount % 4 == 0:
                            y -= (img_w + 24)
                    except Exception:
                        pass
                y -= (img_w + 12)
            else:
                # placeholder
                if TOPO_PLACEHOLDER.exists():
                    try:
                        from reportlab.lib.utils import ImageReader
                        ir = ImageReader(str(TOPO_PLACEHOLDER))
                        c.drawImage(ir, x, y-140, width=240, height=140, mask='auto')
                        y -= 160
                    except Exception:
                        y -= 6
            # connectivity image or placeholder for this file
            conn_img = fdata.get("connectivity_image", None)
            if conn_img:
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(io.BytesIO(conn_img)) if isinstance(conn_img, (bytes,bytearray)) else ImageReader(str(conn_img))
                    c.drawImage(ir, x, y-160, width=320, height=160, mask='auto')
                    y -= 170
                except Exception:
                    y -= 6
            else:
                if CONN_PLACEHOLDER.exists():
                    try:
                        from reportlab.lib.utils import ImageReader
                        ir = ImageReader(str(CONN_PLACEHOLDER))
                        c.drawImage(ir, x, y-140, width=240, height=140, mask='auto')
                        y -= 160
                    except Exception:
                        y -= 6

    else:
        if lang == 'en':
            c.drawString(x, y, "No EDF files processed."); y -= 14
        else:
            c.drawRightString(W-m, y, reshape_ar("لا توجد ملفات EDF معالجة.")); y -= 14

    # XAI section (from summary['xai'])
    if y < 150:
        c.showPage()
        y = H - m
    if lang == 'en':
        c.drawString(x, y, "Explainable AI — Top contributors:"); y -= 12
    else:
        c.drawRightString(W-m, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:")); y -= 12
    xai = summary.get("xai", None)
    if xai:
        # xai might be a dict with groups or direct feature->importance
        if isinstance(xai, dict):
            # If it contains group keys (e.g., depression_global), iterate groups
            for group_key, group_val in xai.items():
                if isinstance(group_val, dict):
                    if lang == 'en':
                        c.drawString(x+6, y, f"{group_key}:"); y -= 10
                        for kk, vv in sorted(group_val.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int,float)) else 0)[:12]:
                            c.drawString(x+10, y, f"{kk}: {format_for_pdf_value(vv)}"); y -= 9
                    else:
                        c.drawRightString(W-m, y, reshape_ar(f"{group_key}:")); y -= 10
                        for kk, vv in sorted(group_val.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int,float)) else 0)[:12]:
                            c.drawRightString(W-m, y, reshape_ar(f"{kk}: {format_for_pdf_value(vv)}")); y -= 9
                else:
                    if lang == 'en':
                        c.drawString(x+6, y, f"{group_key}: {format_for_pdf_value(group_val)}"); y -= 10
                    else:
                        c.drawRightString(W-m, y, reshape_ar(f"{group_key}: {format_for_pdf_value(group_val)}")); y -= 10
        else:
            # fallback if xai is list/string
            if lang == 'en':
                c.drawString(x+6, y, str(xai)); y -= 10
            else:
                c.drawRightString(W-m, y, reshape_ar(str(xai))); y -= 10
    else:
        if lang == 'en':
            c.drawString(x+6, y, "XAI not available (no shap_summary.json or model)."); y -= 10
        else:
            c.drawRightString(W-m, y, reshape_ar("لا توجد بيانات XAI (لا يوجد shap_summary.json أو نموذج).")); y -= 10

    y -= 8
    # Recommendations
    if lang == 'en':
        c.drawString(x, y, "Structured Clinical Recommendations:"); y -= 12
    else:
        c.drawRightString(W-m, y, reshape_ar("التوصيات السريرية المنظمة:")); y -= 12
    for r in summary.get("recommendations", []):
        if lang == 'en':
            c.drawString(x+6, y, "- " + r); y -= 10
        else:
            c.drawRightString(W-m, y, reshape_ar("- " + r)); y -= 10

    # Footer with branding & disclaimer
    footer_en = "Designed and developed by Golden Bird LLC — Vista Kaviani"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني")
    disc_en = "Research/demo only — Not a clinical diagnosis."
    disc_ar = reshape_ar("هذه لأغراض البحث/العرض فقط — ليست تشخيصًا طبيًا نهائياً.")
    c.setFont(font, 8)
    if lang == 'en':
        c.drawCentredString(W/2, 30, footer_en)
        c.drawCentredString(W/2, 18, disc_en)
    else:
        c.drawCentredString(W/2, 30, footer_ar)
        c.drawCentredString(W/2, 18, disc_ar)

    c.save()
    buf.seek(0)
    return buf.read()

# ----------------- UI -----------------
st.markdown("""
<style>
.block-container { max-width:1200px; }
.header { background: linear-gradient(90deg,#0b3d91,#2451a6); color:white; padding:14px; border-radius:8px; }
.card { background:#fff; padding:12px; border-radius:8px; box-shadow:0 1px 6px rgba(0,0,0,0.06); }
.small { color:#cbd5e1; font-size:13px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical XAI</h2><div class='small'>EEG / QEEG + Explainable AI — Clinical support</div></div>", unsafe_allow_html=True)
with col2:
    if LOGO_SVG.exists():
        st.image(str(LOGO_SVG), width=120, use_container_width=False)
    else:
        st.markdown("<div style='text-align:right;font-weight:600'>Golden Bird LLC</div>", unsafe_allow_html=True)

# Sidebar: patient & settings (English default)
with st.sidebar:
    st.header("Settings & Patient")
    lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)  # default English
    st.markdown("---")
    st.subheader("Patient information")
    patient_name = st.text_input("Name / الاسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1940,1,1), max_value=date.today())
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))
    st.markdown("---")
    st.subheader("Clinical & Labs")
    lab_options = [
        "Vitamin B12", "Thyroid (TSH)", "Vitamin D", "Folate", "Homocysteine", "HbA1C", "Cholesterol / Lipids"
    ]
    selected_labs = st.multiselect("Available lab results", options=lab_options)
    lab_notes = st.text_area("Notes / lab values (optional)", help="Enter numeric values or comments for relevant labs")
    meds = st.text_area("Current medications (name + dose)")
    conditions = st.text_area("Comorbid conditions (e.g., diabetes, hypertension, anxiety)")
    st.markdown("---")
    st.write(f"Backends: mne={HAS_MNE} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

# Main: EDF upload
st.markdown("### 1) Upload EDF file(s) (.edf) — you can upload multiple files to compare")
uploads = st.file_uploader("Drag & drop EDF files here", type=["edf"], accept_multiple_files=True)

# PHQ-9 (with corrected Q3/Q5/Q8) — show in selected language
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
for i in range(1,10):
    label_en = f"Q{i}. {PHQ_EN[i-1]}"
    label_ar = f"س{i}. {PHQ_AR[i-1]}"
    if lang == 'ar':
        display_label = reshape_ar(PHQ_AR[i-1])
    else:
        display_label = label_en

    # build options with leading ascii digits to keep mapping simple; arabic text reshaped only on phrase
    if i == 3:
        if lang == 'ar':
            opts_display = [f"0 — {reshape_ar('لا')}", f"1 — {reshape_ar('أرق (صعوبة في النوم)')}", f"2 — {reshape_ar('قلة النوم')}", f"3 — {reshape_ar('زيادة النوم')}"]
        else:
            opts_display = ["0 — Not at all", "1 — Insomnia (difficulty falling/staying asleep)", "2 — Sleeping less", "3 — Sleeping more"]
    elif i == 5:
        if lang == 'ar':
            opts_display = [f"0 — {reshape_ar('لا')}", f"1 — {reshape_ar('قلة الأكل')}", f"2 — {reshape_ar('زيادة الأكل')}", f"3 — {reshape_ar('متغير / كلاهما')}"]
        else:
            opts_display = ["0 — Not at all", "1 — Eating less", "2 — Eating more", "3 — Both / variable"]
    elif i == 8:
        if lang == 'ar':
            opts_display = [f"0 — {reshape_ar('لا')}", f"1 — {reshape_ar('تباطؤ في الحركة/التكلم')}", f"2 — {reshape_ar('تململ/قلق')}", f"3 — {reshape_ar('متغير / كلاهما')}"]
        else:
            opts_display = ["0 — Not at all", "1 — Moving/speaking slowly", "2 — Fidgety / restless", "3 — Both / variable"]
    else:
        if lang == 'ar':
            opts_display = [f"0 — {reshape_ar('لا شيء')}", f"1 — {reshape_ar('عدة أيام')}", f"2 — {reshape_ar('أكثر من نصف الأيام')}", f"3 — {reshape_ar('تقريباً كل يوم')}"]
        else:
            opts_display = ["0 — Not at all", "1 — Several days", "2 — More than half the days", "3 — Nearly every day"]

    # label the radio with either english or reshaped arabic phrase
    if lang == 'en':
        ans = st.radio(label_en, options=opts_display, index=0, key=f"phq_{i}")
    else:
        # show reshaped label text above choices
        st.markdown(f"**{display_label}**")
        ans = st.radio("", options=opts_display, index=0, key=f"phq_{i}_ar")

    # parse value from selected option (starts with ascii digit)
    try:
        val = int(str(ans).split("—")[0].strip())
    except Exception:
        # fallback: first character numeric?
        s = str(ans).strip()
        val = int(s[0]) if s and s[0].isdigit() else 0
    phq_answers[f"Q{i}"] = val

phq_total = sum(phq_answers.values())
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal,5–9 mild,10–14 moderate,15–19 mod-severe,20–27 severe)")

# AD8 (8 yes/no)
st.markdown("### 3) AD8 (Cognitive screening — 8 items)")
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
        label = reshape_ar(AD8_AR[i-1])
        st.markdown(f"**{label}**")
        val = st.radio("", options=[0,1], index=0, key=f"ad8_{i}_ar", horizontal=True)
    ad8_answers[f"A{i}"] = int(val)

ad8_total = sum(ad8_answers.values())
st.info(f"AD8 total: {ad8_total} (score ≥2 suggests cognitive impairment)")

# Processing options
st.markdown("---")
st.header("Processing & Visualization Options")
use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
do_topomap = st.checkbox("Generate topography maps (if matplotlib available)", value=True)
do_connectivity = st.checkbox("Compute functional disconnection (placeholder if not available)", value=True)
run_models = st.checkbox("Run ML models if provided (model_depression.pkl, model_alzheimer.pkl)", value=False)

# Process EDF files (multi)
results = []
if uploads:
    for up in uploads:
        st.write(f"Processing {up.name} ...")
        try:
            tmp = save_tmp_upload(up)
            edf = read_edf(tmp)
            data = edf["data"]
            sf = edf.get("sfreq") or DEFAULT_SF
            st.success(f"Loaded: backend={edf['backend']}   channels={data.shape[0]}   sfreq={sf}")
            cleaned = preprocess_data(data, sf, do_notch=use_notch)
            dfbands = compute_psd_bands(cleaned, sf)
            st.dataframe(dfbands.head(10))
            agg = aggregate_bands(dfbands, ch_names=edf.get("ch_names"))
            st.write("Aggregated features:", agg)
            topo_images = {}
            if do_topomap:
                for band in ["Delta","Theta","Alpha","Beta"]:
                    vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(data.shape[0])
                    img = generate_topomap_image(vals, ch_names=edf.get("ch_names"), band_name=band)
                    topo_images[band] = img
                    # show inline in streamlit
                    if isinstance(img, (bytes,bytearray)):
                        st.image(img, caption=f"{band} topomap", use_container_width=False)
                    elif isinstance(img, str) and Path(img).exists():
                        st.image(str(img), caption=f"{band} topomap", use_container_width=False)
                    else:
                        if TOPO_PLACEHOLDER.exists():
                            st.image(str(TOPO_PLACEHOLDER), caption=f"{band} topomap (placeholder)", use_container_width=False)
            conn_img = None
            conn_narr = None
            if do_connectivity:
                mat, narr = compute_connectivity_placeholder()
                conn_narr = narr
                if HAS_MATPLOTLIB:
                    fig = plt.figure(figsize=(4,3))
                    plt.imshow(mat, cmap='viridis')
                    plt.colorbar()
                    plt.title("Connectivity (placeholder)")
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    conn_img = buf.getvalue()
                    st.image(conn_img, caption="Functional disconnection (placeholder)", use_container_width=False)
                else:
                    if CONN_PLACEHOLDER.exists():
                        st.image(str(CONN_PLACEHOLDER), caption="Connectivity placeholder", use_container_width=False)
            results.append({
                "filename": up.name,
                "agg_features": agg,
                "df_bands": dfbands,
                "topo_images": topo_images,
                "connectivity_image": conn_img,
                "connectivity_narrative": conn_narr
            })
        except Exception as e:
            st.error(f"Failed processing {up.name}: {e}")
            _trace(e)

# Build summary
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

# Load XAI summary if present
shap_json = load_shap_summary()
if shap_json:
    summary["xai"] = shap_json
else:
    # try to compute fallback from model feature_importances_
    if joblib and (MODEL_DEP.exists() or MODEL_AD.exists()):
        try:
            fe = {}
            if MODEL_DEP.exists():
                md = joblib.load(str(MODEL_DEP))
                if hasattr(md, "feature_importances_"):
                    fe["depression_global"] = dict(zip(getattr(md, "feature_names_in_", []) if hasattr(md,"feature_names_in_") else [f"f{i}" for i in range(len(md.feature_importances_))], md.feature_importances_.tolist()))
            if MODEL_AD.exists():
                ma = joblib.load(str(MODEL_AD))
                if hasattr(ma, "feature_importances_"):
                    fe["alzheimers_global"] = dict(zip(getattr(ma, "feature_names_in_", []) if hasattr(ma,"feature_names_in_") else [f"f{i}" for i in range(len(ma.feature_importances_))], ma.feature_importances_.tolist()))
            if fe:
                summary["xai"] = fe
        except Exception:
            pass

# Model predictions (optional)
if run_models and joblib and results:
    try:
        Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
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
            mlrisk = float(np.mean(preds))*100.0
            summary["ml_risk"] = mlrisk
            if mlrisk >= 50:
                summary["risk_category"] = "High"
            elif mlrisk >= 25:
                summary["risk_category"] = "Moderate"
            else:
                summary["risk_category"] = "Low"
    except Exception as e:
        st.warning("Model prediction failed: " + str(e))

# Recommendations (rule-based)
if summary["phq9"]["total"] >= 10:
    summary["recommendations"].append("PHQ-9 suggests moderate/severe depression — consider psychiatric referral and treatment planning.")
if summary["ad8"]["total"] >= 2 or (results and results[0]["agg_features"].get("theta_alpha_ratio",0) > 1.4):
    summary["recommendations"].append("AD8 elevated or Theta/Alpha increased — consider neurocognitive testing and neuroimaging (MRI/FDG-PET).")
summary["recommendations"].append("Correlate QEEG/connectivity findings with PHQ-9 and AD8 and clinical interview.")
summary["recommendations"].append("Review medications that may affect EEG.")
if not summary["recommendations"]:
    summary["recommendations"].append("Clinical follow-up and re-evaluation in 3-6 months.")

# ----------------- Report generation UI -----------------
st.markdown("---")
st.header("Generate report")
st.write("Choose one language for the report (English or Arabic). Only one language at a time will be used in the PDF.")
colA, colB = st.columns([3,1])
with colA:
    report_lang = st.selectbox("Report language / لغة التقرير", options=["en","ar"], index=0)
    amiri_path = st.text_input("Amiri TTF path (optional) - leave blank if Amiri-Regular.ttf in root", value="")
with colB:
    if st.button("Generate PDF report"):
        try:
            topo_imgs = {}
            if results and results[0].get("topo_images"):
                topo_imgs = results[0]["topo_images"]
            conn_img = results[0].get("connectivity_image") if results else None
            pdfb = generate_pdf_report(summary, lang=report_lang, amiri_path=(amiri_path or None), topo_images=topo_imgs, conn_image=conn_img)
            st.download_button("Download PDF", data=pdfb, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("Report generated. Footer includes Golden Bird LLC.")
        except Exception as e:
            st.error("PDF generation failed.")
            _trace(e)

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis.")
