# app_v6.py — NeuroEarly Pro v6 (Final Clinical Edition)
# Features: bilingual (EN/AR), logo, healthy baseline, topomaps (Δ,θ,α,β,γ),
# FDI, connectivity (coherence/correlation fallback), PHQ+Alzheimer questionnaires,
# SHAP visualization (from shap_summary.json), modern PDF report (ReportLab + Amiri),
# graceful degradation when optional libs are missing.

import os, io, sys, tempfile, traceback, json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image as PILImage

# optional heavy libs (try import; if missing, app still works with warnings)
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

# SciPy is required for PSD/coherence; degrade gracefully if missing
HAS_SCIPY = True
try:
    from scipy.signal import welch, butter, sosfilt, coherence
except Exception:
    HAS_SCIPY = False

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"
HEALTHY_EDF = ASSETS / "healthy_baseline.edf"  # we'll create if missing (if possible)

APP_TITLE = "NeuroEarly Pro — Clinical"
BLUE = "#0b63d6"
LIGHT_BG = "#eaf4ff"

# frequency bands to compute
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

def now_str(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.utcnow().strftime(fmt)

def clamp01(x):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.0

# ----------------------------
# Synthetic healthy baseline generator
# ----------------------------
def generate_synthetic_healthy_edf(path: Path, n_channels=19, sf=250, seconds=60):
    """
    Try to create a synthetic 'healthy' EEG and save as EDF using pyedflib if available.
    If pyedflib not available, save as numpy .npy and we will load that as fallback.
    """
    n_samples = sf * seconds
    # simple band-limited noise: alpha peak ~10Hz, theta ~6Hz, lower power in delta/beta/gamma
    t = np.arange(n_samples) / sf
    signals = []
    rng = np.random.RandomState(42)
    for ch in range(n_channels):
        # base noise
        sig = 0.5 * rng.normal(size=n_samples)
        # add small rhythmic alpha component with random phase
        alpha = 5.0 * np.sin(2 * np.pi * (8+2*rng.rand()) * t + rng.rand()*2*np.pi) * (0.6 + 0.4*rng.rand())
        theta = 2.0 * np.sin(2 * np.pi * (4+1*rng.rand()) * t + rng.rand()*2*np.pi) * (0.4 + 0.3*rng.rand())
        # combine and low amplitude gamma
        gamma = 0.5 * np.sin(2*np.pi*(30+10*rng.rand())*t)*0.2*rng.rand()
        s = sig + alpha + theta + gamma
        # slight channel-specific scaling
        s *= (1 + 0.05 * rng.randn())
        signals.append(s.astype(np.float32))
    arr = np.vstack(signals)
    try:
        if HAS_PYEDF:
            import pyedflib
            ch_labels = [f"Ch{c+1}" for c in range(n_channels)]
            f = pyedflib.EdfWriter(str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
            channel_info = []
            for ch in range(n_channels):
                ch_dict = {'label': ch_labels[ch],
                           'dimension': 'uV',
                           'sample_rate': sf,
                           'physical_min': float(np.min(arr[ch])),
                           'physical_max': float(np.max(arr[ch])),
                           'digital_min': -32768,
                           'digital_max': 32767,
                           'transducer': '',
                           'prefilter': ''}
                channel_info.append(ch_dict)
            f.setSignalHeaders(channel_info)
            f.writeSamples(arr.tolist())
            f.close()
            return True
    except Exception as e:
        print("pyedflib write failed:", e)
    # fallback: save numpy
    try:
        np.savez_compressed(str(path.with_suffix(".npz")), data=arr, sf=sf)
        return True
    except Exception as e:
        print("saving npz failed:", e)
        return False

# ----------------------------
# EDF reader wrapper (robust)
# ----------------------------
def read_edf_uploaded(uploaded) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
    """
    Returns (data (n_ch x n_s), meta dict {'sfreq':..., 'ch_names':[...]}, err_msg)
    Accepts streamlit UploadedFile or Path string.
    """
    if uploaded is None:
        return None, None, "No file"
    # if path-like given:
    if isinstance(uploaded, (str, Path)):
        p = str(uploaded)
        try:
            if HAS_MNE:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                data = raw.get_data()
                meta = {"sfreq": float(raw.info.get("sfreq", 256.0)), "ch_names": raw.info.get("ch_names", [])}
                return data, meta, None
            elif HAS_PYEDF:
                edf = pyedflib.EdfReader(p)
                n = edf.signals_in_file
                chs = edf.getSignalLabels()
                sf = edf.getSampleFrequency(0)
                arrs = [edf.readSignal(i) for i in range(n)]
                edf.close()
                data = np.vstack(arrs)
                meta = {"sfreq": float(sf), "ch_names": chs}
                return data, meta, None
            else:
                return None, None, "No EDF reader (install mne or pyedflib)."
        except Exception as e:
            return None, None, f"Read file error: {e}"
    # else UploadedFile
    try:
        raw_bytes = uploaded.getvalue()
    except Exception as e:
        return None, None, f"uploaded access error: {e}"
    # write to temp file and read
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
            tf.write(raw_bytes)
            tmp = tf.name
        return read_edf_uploaded(tmp)
    finally:
        # keep file for mne if it uses mmap; we won't delete here
        pass

# ----------------------------
# band power computation
# ----------------------------
def compute_band_powers(data: np.ndarray, sf: float, bands=BANDS) -> pd.DataFrame:
    """
    data: n_ch x n_samples
    returns DataFrame indexed by channel with columns like 'Theta_abs','Theta_rel'
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for spectral computations")
    n_ch = data.shape[0]
    rows = []
    for i in range(n_ch):
        f, Pxx = welch(data[i,:], fs=sf, nperseg=min(2048, data.shape[1]))
        total = float(np.trapz(Pxx[(f>=1)&(f<=45)], f[(f>=1)&(f<=45)])) if np.any((f>=1)&(f<=45)) else 0.0
        row = {}
        for name,(lo,hi) in bands.items():
            mask = (f>=lo)&(f<hi)
            val = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            row[f"{name}_abs"] = val
            row[f"{name}_rel"] = (val/total) if total>0 else 0.0
        row["total_power"] = total
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"ch_{i}" for i in range(n_ch)])
    return df

# ----------------------------
# topomap rendering (grid fallback) -> returns PNG bytes
# ----------------------------
def topomap_png_from_vals(vals: np.ndarray, band_name:str="Band"):
    try:
        arr = np.asarray(vals).astype(float)
        n = len(arr)
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig,ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest", origin='upper')
        ax.set_title(f"{band_name} Topomap")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png',dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# ----------------------------
# connectivity: try coherence (scipy) then fallback to Pearson corr
# ----------------------------
def compute_connectivity(data: np.ndarray, sf: float, method="coherence") -> Optional[np.ndarray]:
    """
    data: n_ch x n_samples
    returns connectivity matrix n_ch x n_ch (values 0..1 or -1..1 for correlation)
    """
    try:
        n_ch = data.shape[0]
        if HAS_SCIPY and method=="coherence":
            # compute mean coherence in alpha band between pairs (can be slow for many channels)
            lo,hi = BANDS["Alpha"]
            conn = np.zeros((n_ch,n_ch))
            for i in range(n_ch):
                for j in range(i,n_ch):
                    try:
                        f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(2048, data.shape[1]))
                        mask = (f>=lo)&(f<=hi)
                        mean_coh = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                    except Exception:
                        mean_coh = 0.0
                    conn[i,j] = mean_coh
                    conn[j,i] = mean_coh
            return conn
        else:
            # Pearson correlation fallback
            x = data.copy()
            x = (x - x.mean(axis=1,keepdims=True)) / (x.std(axis=1,keepdims=True)+1e-12)
            conn = np.corrcoef(x)
            conn = np.nan_to_num(conn)
            return conn
    except Exception as e:
        print("connectivity compute failed:", e)
        return None

# ----------------------------
# Focal Delta Index (FDI)
# ----------------------------
def compute_fdi_from_df(df: pd.DataFrame) -> Dict[str,Any]:
    try:
        if "Delta_rel" not in df.columns:
            return {}
        vals = df["Delta_rel"].values
        global_mean = float(np.nanmean(vals))
        idx = int(np.nanargmax(vals))
        top_val = float(vals[idx])
        fdi = float(top_val / (global_mean + 1e-12)) if global_mean>0 else None
        return {"global_mean": global_mean, "top_idx": idx, "top_name": df.index[idx] if idx < len(df.index) else "", "top_value": top_val, "FDI": fdi}
    except Exception as e:
        print("FDI compute error:", e)
        return {}

# ----------------------------
# SHAP render from shap_summary.json
# ----------------------------
def render_shap_png(shap_path: Path, model_hint="depression_global"):
    if not shap_path.exists():
        return None
    try:
        with open(shap_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        key = model_hint if model_hint in sj else next(iter(sj.keys()))
        feats = sj.get(key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig,ax = plt.subplots(figsize=(6,3))
        s.plot.bar(ax=ax)
        ax.set_title("SHAP - top contributors")
        fig.tight_layout()
        buf=io.BytesIO(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("shap render failed:", e)
        return None

# ----------------------------
# PDF generator (ReportLab) bilingual support
# ----------------------------
def reshape_ar(text:str) -> str:
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf(summary: dict, lang="en") -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if AMIRI_TTF.exists() and HAS_ARABIC:
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri register failed:", e)
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story=[]
        # Header
        title = "NeuroEarly Pro — Clinical Report" if lang=="en" else reshape_ar("تقرير NeuroEarly Pro السريري")
        story.append(Paragraph(title, styles["TitleBlue"]))
        story.append(Spacer(1,6))
        if LOGO_PATH.exists():
            try:
                story.append(RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch))
            except Exception:
                pass
        story.append(Spacer(1,6))
        # Patient info
        pi = summary.get("patient_info", {})
        rows=[["Field","Value"], ["Patient ID", pi.get("id","-")], ["DOB", pi.get("dob","-")], ["Report date", summary.get("created", now_str())]]
        t=Table(rows,colWidths=[2.6*inch,3.2*inch])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t); story.append(Spacer(1,8))
        # Final ML risk
        if summary.get("final_ml_risk") is not None:
            story.append(Paragraph(f"<b>Final ML Risk Score: {summary['final_ml_risk']*100:.1f}%</b>", styles["H2"]))
            story.append(Spacer(1,6))
        # Metrics table
        story.append(Paragraph("QEEG Key Metrics", styles["H2"]))
        metrics = summary.get("metrics", {})
        if metrics:
            rows = [[k,str(v)] for k,v in metrics.items()]
            t2 = Table(rows,colWidths=[3.2*inch,2.6*inch])
            t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(t2); story.append(Spacer(1,6))
        # normative bar
        if summary.get("normative_bar"):
            try:
                story.append(Paragraph("Normative Comparison", styles["H2"]))
                story.append(RLImage(io.BytesIO(summary["normative_bar"]), width=5.5*inch, height=2.0*inch))
                story.append(Spacer(1,6))
            except Exception:
                pass
        # topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["H2"]))
            imgs=[]
            for band, b in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch))
                except Exception:
                    pass
            # arrange 2-per-row
            row=[]
            for i,im in enumerate(imgs):
                row.append(im)
                if (i%2)==1:
                    story.append(Table([row], colWidths=[2.6*inch,2.6*inch])); row=[]
            if row: story.append(Table([row], colWidths=[2.6*inch,2.6*inch]))
            story.append(Spacer(1,6))
        # connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("Functional Connectivity", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,6))
        # FDI
        if summary.get("fdi"):
            story.append(Paragraph("Focal Delta Index (FDI)", styles["H2"]))
            fdi = summary["fdi"]
            story.append(Paragraph(f"Top channel: {fdi.get('top_name','-')} — FDI: {fdi.get('FDI', '-')}", styles["Body"]))
            story.append(Spacer(1,6))
        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,6))
        # Clinical scores
        if summary.get("clinical"):
            story.append(Paragraph("Clinical Questionnaires", styles["H2"]))
            cli = summary["clinical"]
            story.append(Paragraph(f"PHQ-9 Score: {cli.get('phq_score','-')}", styles["Body"]))
            story.append(Paragraph(f"Cognitive Score: {cli.get('ad_score','-')}", styles["Body"]))
            story.append(Spacer(1,6))
        # Recommendations
        if summary.get("recommendations"):
            story.append(Paragraph("Recommendations", styles["H2"]))
            for r in summary["recommendations"]:
                story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,6))
        # footer
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro System", styles["Note"]))
        story.append(Spacer(1,6))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF build error:", e)
        traceback.print_exc()
        return None

# ----------------------------
# Questionnaires definitions (PHQ variant & Alzheimer's short)
# ----------------------------
PHQ9 = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Sleep problems (select: insomnia/hypersomnia)"
    "4. Feeling tired or having little energy",
    "5. Appetite changes (select: increased/decreased)",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "7. Trouble concentrating on things",
    "8. Moving or speaking so slowly that others notice OR being more fidgety/restless",
    "9. Thoughts that you would be better off dead or of hurting yourself"
]
PHQ_STD = [("0 - Not at all",0),("1 - Several days",1),("2 - More than half the days",2),("3 - Nearly every day",3)]
PHQ_Q3 = [("0 - No change",0),("1 - Insomnia - Several days",1),("2 - Insomnia - More than half the days",2),("3 - Hypersomnia - Nearly every day",3)]
PHQ_Q5 = [("0 - No change",0),("1 - Decreased appetite - Several days",1),("2 - Increased or decreased appetite - More than half the days",2),("3 - Marked change - Nearly every day",3)]
PHQ_Q8 = [("0 - No change",0),("1 - Slight change - Several days",1),("2 - Noticeable change - More than half the days",2),("3 - Marked change - Nearly every day",3)]

ALZ_QUESTIONS = [
    "1. Recurrent memory loss (forgetting recent events)",
    "2. Orientation problems (time/place)",
    "3. Naming difficulties",
    "4. Getting lost in familiar places",
    "5. Personality / behavior changes",
    "6. Difficulty with daily tasks",
    "7. Impaired judgement",
    "8. Social withdrawal / apathy"
]
ALZ_OPTIONS = [("0 - No",0),("1 - Occasionally",1),("2 - Often",2),("3 - Always / Severe",3)]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:12px;border-radius:8px;background:{LIGHT_BG};">
  <div style="font-weight:700;color:{BLUE};font-size:18px;">🧠 {APP_TITLE}</div>
  <div style="font-size:12px;color:#333;opacity:0.9">Prepared by Golden Bird LLC</div>
</div>""", unsafe_allow_html=True)

# Sidebar left
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English","Arabic"])
    lang = "ar" if lang_choice.startswith("Ar") else "en"
    st.markdown("---")
    st.subheader("Patient info")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    st.selectbox("Sex", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.subheader("Medications")
    meds = st.text_area("Medications (one per line)", height=80)
    st.subheader("Blood tests")
    labs = st.text_area("Labs (one per line)", height=80)
    st.markdown("---")
    st.subheader("Upload EDF")
    uploads = st.file_uploader("Upload .edf files (multiple allowed)", type=["edf","EDF"], accept_multiple_files=True)
    st.markdown("")
    analyze = st.button("Process & Analyze")

# show logo centrally on main page
if LOGO_PATH.exists():
    try:
        logo_img = PILImage.open(str(LOGO_PATH))
        st.image(logo_img, width=180)
    except Exception:
        pass

st.markdown("## Clinical Questionnaires")
# render PHQ-9
phq_vals = {}
for i,q in enumerate(PHQ9, start=1):
    key = f"phq_{i}"
    if i==3:
        sel = st.selectbox(q, [o[0] for o in PHQ_Q3], key=key)
        phq_vals[key] = dict(PHQ_Q3)[sel] if sel in dict(PHQ_Q3) else PHQ_Q3[[o[0] for o in PHQ_Q3].index(sel)][1]
    elif i==5:
        sel = st.selectbox(q, [o[0] for o in PHQ_Q5], key=key)
        phq_vals[key] = dict(PHQ_Q5)[sel] if sel in dict(PHQ_Q5) else PHQ_Q5[[o[0] for o in PHQ_Q5].index(sel)][1]
    elif i==8:
        sel = st.selectbox(q, [o[0] for o in PHQ_Q8], key=key)
        phq_vals[key] = dict(PHQ_Q8)[sel] if sel in dict(PHQ_Q8) else PHQ_Q8[[o[0] for o in PHQ_Q8].index(sel)][1]
    else:
        sel = st.selectbox(q, [o[0] for o in PHQ_STD], key=key)
        phq_vals[key] = dict(PHQ_STD)[sel] if sel in dict(PHQ_STD) else PHQ_STD[[o[0] for o in PHQ_STD].index(sel)][1]

# Alzheimer's short form
st.markdown("## Cognitive Screening (Alzheimer short form)")
alz_vals = {}
for i,q in enumerate(ALZ_QUESTIONS, start=1):
    key = f"alz_{i}"
    sel = st.selectbox(q, [o[0] for o in ALZ_OPTIONS], key=key)
    alz_vals[key] = dict(ALZ_OPTIONS)[sel] if sel in dict(ALZ_OPTIONS) else ALZ_OPTIONS[[o[0] for o in ALZ_OPTIONS].index(sel)][1]

# session results store
if "results" not in st.session_state:
    st.session_state["results"] = []

# If user pressed analyze:
if analyze:
    # ensure healthy baseline exists (create synthetic if missing)
    if not HEALTHY_EDF.exists():
        created = generate_synthetic_healthy_edf(HEALTHY_EDF)
        if created:
            st.success("Healthy baseline created (assets/healthy_baseline.edf or npz).")
        else:
            st.warning("Could not create healthy EDF baseline; continuing without baseline.")

    if not uploads:
        st.info("No EDF uploaded — running analysis on healthy baseline only (demo mode).")
        # use default baseline if present
        if HEALTHY_EDF.exists():
            data_b, meta_b, errb = read_edf_uploaded(str(HEALTHY_EDF))
            if data_b is None:
                st.error(f"Baseline load error: {errb}")
    else:
        st.session_state["results"].clear()
        for up in uploads:
            st.write(f"Processing {up.name} ...")
            data, meta, err = read_edf_uploaded(up)
            if err or data is None:
                st.error(f"{up.name} read error: {err}")
                continue
            sf = float(meta.get("sfreq", 250.0))
            ch_names = meta.get("ch_names", [f"ch{i}" for i in range(data.shape[0])])
            # band powers
            try:
                df_bands = compute_band_powers(data, sf)
            except Exception as e:
                st.error(f"Band power error: {e}")
                continue
            # FDI
            fdi = compute_fdi_from_df(df_bands)
            # topomaps
            topo_imgs = {}
            for b in BANDS.keys():
                col = f"{b}_rel"
                if col in df_bands.columns:
                    vals = df_bands[col].values
                else:
                    vals = df_bands.get(f"{b}_abs", pd.Series(np.zeros(data.shape[0]))).values
                topo_imgs[b] = topomap_png_from_vals(vals, band_name=b)
            # connectivity
            conn_img = None
            conn_matrix = None
            try:
                conn_matrix = compute_connectivity(data, sf, method="coherence" if HAS_SCIPY else "corr")
                if conn_matrix is not None:
                    fig,ax = plt.subplots(figsize=(4,3))
                    im = ax.imshow(conn_matrix, cmap="viridis")
                    ax.set_title("Connectivity")
                    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                    conn_img = buf.getvalue()
            except Exception as e:
                st.warning(f"Connectivity not available: {e}")

            # normative comparison using baseline if exists
            normative_bar = None
            if HEALTHY_EDF.exists():
                # try load baseline (npz or edf)
                base_data, base_meta, berr = None, None, None
                if HEALTHY_EDF.exists():
                    base_data, base_meta, berr = read_edf_uploaded(str(HEALTHY_EDF))
                if base_data is not None:
                    try:
                        df_base = compute_band_powers(base_data, base_meta.get("sfreq", sf))
                        # compare Theta/Alpha and Alpha asymmetry (F3-F4)
                        theta_alpha_patient = (df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean()+1e-12)) if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns else 0.0
                        theta_alpha_base = (df_base["Theta_rel"].mean() / (df_base["Alpha_rel"].mean()+1e-12)) if "Theta_rel" in df_base.columns and "Alpha_rel" in df_base.columns else 0.0
                        # alpha asymmetry F3-F4
                        def find_idx(names, key):
                            for i,nm in enumerate(names):
                                if key in nm:
                                    return i
                            return None
                        p_f3 = find_idx(ch_names, "F3"); p_f4 = find_idx(ch_names, "F4")
                        b_f3 = find_idx(base_meta.get("ch_names",[]), "F3"); b_f4 = find_idx(base_meta.get("ch_names",[]), "F4")
                        alpha_asym_p = None
                        alpha_asym_b = None
                        if p_f3 is not None and p_f4 is not None and "Alpha_rel" in df_bands.columns:
                            alpha_asym_p = float(df_bands.iloc[p_f3]["Alpha_rel"] - df_bands.iloc[p_f4]["Alpha_rel"])
                        if b_f3 is not None and b_f4 is not None and "Alpha_rel" in df_base.columns:
                            alpha_asym_b = float(df_base.iloc[b_f3]["Alpha_rel"] - df_base.iloc[b_f4]["Alpha_rel"])
                        # draw normative bar
                        fig,ax = plt.subplots(figsize=(5.5,2.2))
                        labels = ["Theta/Alpha (patient)","Theta/Alpha (baseline)"]
                        vals = [theta_alpha_patient, theta_alpha_base]
                        ax.bar([0,1], vals, color=['#1f77b4','#2ca02c'])
                        ax.set_xticks([0,1]); ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.set_title("Theta/Alpha: patient vs baseline")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                        normative_bar = buf.getvalue()
                    except Exception as e:
                        print("normative compare failed:", e)

            # SHAP render
            shap_png = None
            if SHAP_JSON.exists():
                try:
                    shap_png = render_shap_png(SHAP_JSON, model_hint="depression_global")
                except Exception:
                    shap_png = None

            # questionnaire scores
            phq_score = sum([int(v) for v in phq_vals.values() if isinstance(v,(int,float,str))])
            alz_score = sum([int(v) for v in alz_vals.values() if isinstance(v,(int,float,str))])

            # Final ML risk heuristic
            theta_alpha_val = (df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean()+1e-12)) if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns else 0.0
            ta_norm = clamp01(theta_alpha_val/2.0)
            phq_norm = clamp01(phq_score/27.0)
            ad_norm = clamp01(alz_score/24.0)
            final_risk = 0.45*ta_norm + 0.35*ad_norm + 0.2*phq_norm

            metrics = {
                "theta_alpha_ratio": float(theta_alpha_val),
                "alpha_asym_F3_F4": None,
                "mean_connectivity_alpha": float(np.nanmean(conn_matrix)) if 'conn_matrix' in locals() and conn_matrix is not None else None
            }
            # alpha asym F3-F4
            try:
                idx_f3 = next((i for i,n in enumerate(ch_names) if "F3" in n), None)
                idx_f4 = next((i for i,n in enumerate(ch_names) if "F4" in n), None)
                if idx_f3 is not None and idx_f4 is not None and "Alpha_rel" in df_bands.columns:
                    metrics["alpha_asym_F3_F4"] = float(df_bands.iloc[idx_f3]["Alpha_rel"] - df_bands.iloc[idx_f4]["Alpha_rel"])
            except Exception:
                pass

            result = {
                "filename": up.name,
                "sfreq": sf,
                "ch_names": ch_names,
                "df_bands": df_bands,
                "topo_images": topo_imgs,
                "connectivity_image": conn_img,
                "conn_matrix": conn_matrix,
                "fdi": fdi,
                "normative_bar": normative_bar,
                "shap": shap_png,
                "metrics": metrics,
                "phq_score": phq_score,
                "alz_score": alz_score,
                "final_risk": final_risk
            }
            st.session_state["results"].append(result)
            st.success(f"{up.name} processed.")

# Show results if any
if st.session_state.get("results"):
    for r in st.session_state["results"]:
        st.markdown(f"## {r.get('filename')}")
        st.write("Final ML Risk:", f"{r.get('final_risk')*100:.1f}%")
        st.write("PHQ-9 score:", r.get("phq_score"))
        st.write("Cognitive score:", r.get("alz_score"))
        st.write("Metrics:", r.get("metrics"))
        # topomaps display
        cols = st.columns(3)
        i=0
        for band, img in r.get("topo_images", {}).items():
            if img:
                try:
                    cols[i%3].image(img, caption=band, use_column_width=False, width=300)
                except Exception:
                    pass
            i+=1
        if r.get("connectivity_image"):
            st.markdown("Connectivity matrix")
            st.image(r.get("connectivity_image"), width=500)
        if r.get("normative_bar"):
            st.markdown("Normative comparison")
            st.image(r.get("normative_bar"), width=520)
        if r.get("shap"):
            st.markdown("SHAP XAI")
            st.image(r.get("shap"), width=520)
        if r.get("fdi"):
            st.markdown("FDI")
            st.write(r.get("fdi"))
        st.markdown("---")
    # export
    try:
        rows=[]
        for r in st.session_state["results"]:
            rows.append({"filename": r["filename"], "phq":r["phq_score"], "alz":r["alz_score"], "risk": r["final_risk"]})
        dfexp = pd.DataFrame(rows)
        st.download_button("Download CSV", dfexp.to_csv(index=False).encode("utf-8"), file_name=f"NeuroEarly_metrics_{now_str('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    except Exception:
        pass

    # PDF
    if st.button("Generate full PDF (first result)"):
        try:
            s = st.session_state["results"][0]
            summary = {
                "patient_info": {"id": patient_id, "dob": dob.isoformat(), "meds": meds, "labs": labs},
                "metrics": s.get("metrics"),
                "topo_images": s.get("topo_images"),
                "connectivity_image": s.get("connectivity_image"),
                "fdi": s.get("fdi"),
                "shap_img": s.get("shap"),
                "normative_bar": s.get("normative_bar"),
                "clinical": {"phq_score": s.get("phq_score"), "ad_score": s.get("alz_score")},
                "final_ml_risk": s.get("final_risk"),
                "recommendations": [
                    "Automated screening only — clinical correlation required.",
                    "Consider MRI if FDI > 2 or extreme asymmetry present.",
                    "Follow-up in 3-6 months for moderate risk cases."
                ],
                "created": now_str()
            }
            pdf_bytes = generate_pdf(summary, lang=lang)
            if pdf_bytes:
                st.download_button("Download PDF report", pdf_bytes, file_name=f"NeuroEarly_Report_{now_str('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                st.success("PDF ready.")
            else:
                st.error("PDF generation failed — ensure reportlab & fonts are installed.")
        except Exception as e:
            st.error(f"PDF exception: {e}\n{traceback.format_exc()}")

else:
    st.info("No results yet. Upload EDF files and click Process & Analyze.")

st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro System (v6)")

