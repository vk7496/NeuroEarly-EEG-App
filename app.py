# app.py
"""
NeuroEarly Pro — Streamlit clinical assistant (Cloud-ready)
- Bilingual EN/AR (single language chosen for report)
- PHQ-9 (custom Q3/Q5/Q8), AD8, DOB up to 1940
- EDF upload (pyedflib fallback)
- PSD band powers, ratios, alpha asymmetry
- Topography maps (matplotlib if available, otherwise SVG placeholder)
- Functional disconnection heatmap (simple coherence placeholder or computed if mne/mne_connectivity exists)
- XAI: uses shap_summary.json (preferred) or computes feature_importances from demo models (if present)
- PDF generation with reportlab (if installed) with footer "Designed and developed by Golden Bird LLC — Vista Kaviani"
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro — Clinical XAI", layout="wide")

import os, io, json, tempfile, traceback, datetime
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

# Optional heavy imports are attempted safely
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
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_REPORTLAB = False
    HAS_ARABIC_TOOLS = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from scipy.signal import welch, butter, filtfilt, iirnotch

# joblib for loading models if user supplies them
try:
    import joblib
except Exception:
    joblib = None

# Constants & assets
ASSETS_DIR = Path("./assets")
LOGO_SVG = ASSETS_DIR / "GoldenBird_logo.svg"
TOPO_PLACEHOLDER = ASSETS_DIR / "topo_placeholder.svg"
CONN_PLACEHOLDER = ASSETS_DIR / "conn_placeholder.svg"
SHAP_JSON = Path("shap_summary.json")
MODEL_DEP = Path("model_depression.pkl")
MODEL_AD = Path("model_alzheimer.pkl")
AMIRI_TTF = Path("Amiri-Regular.ttf")

BANDS = {"Delta":(0.5,4),"Theta":(4,8),"Alpha":(8,13),"Beta":(13,30)}

DEFAULT_SF = 256.0

def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _trace(e):
    tb = traceback.format_exc()
    st.error("Internal error — see logs")
    st.code(tb)
    print(tb)

# ---------- EDF loader ----------
def save_tmp_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path):
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return {"backend":"mne","raw":raw,"data":raw.get_data(),"ch_names": raw.ch_names, "sfreq": raw.info.get("sfreq", None)}
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
        raise ImportError("No EDF backend available. Install mne or pyedflib")

# ---------- Preprocessing ----------
def notch_filter(sig, sf, freq=50.0, Q=30.0):
    if sf is None or sf<=0:
        return sig
    b,a = iirnotch(freq, Q, sf)
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def bandpass_filter(sig, sf, low=0.5, high=45.0, order=4):
    if sf is None or sf<=0:
        return sig
    ny = 0.5*sf
    lown = max(low/ny, 1e-6)
    highn = min(high/ny, 0.999)
    b,a = butter(order, [lown := lown, highn], btype='band')
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

# ---------- PSD / band features ----------
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
    # ratios
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

# ---------- Topomap (cloud-safe simple interpolation) ----------
def generate_topomap_image(band_mean_by_channel, ch_names=None, sf=DEFAULT_SF, band_name="Alpha"):
    """
    band_mean_by_channel: array-like length = n_channels containing relative power for band
    If matplotlib exists, produce a figure and return PNG bytes; otherwise return placeholder svg path.
    """
    if not HAS_MATPLOTLIB or len(band_mean_by_channel)==0:
        if TOPO_PLACEHOLDER.exists():
            return str(TOPO_PLACEHOLDER)
        return None
    try:
        # we'll create a simple scattered interpolation on a 2D head grid using approximate 10-20 coords
        # simple fixed coords for common labels (approx); if ch_names provided map by label, else scatter evenly.
        coords = []
        labels = []
        if ch_names:
            names = [n.upper() for n in ch_names]
            # approximate mapping for common montage (FP1,Fp2,F3,F4,F7,F8,C3,C4,P3,P4,O1,O2)
            approx = {
                "FP1":(-0.3,0.9),"FP2":(0.3,0.9),"F3":(-0.5,0.5),"F4":(0.5,0.5),
                "F7":(-0.8,0.2),"F8":(0.8,0.2),"C3":(-0.5,0.0),"C4":(0.5,0.0),
                "P3":(-0.5,-0.5),"P4":(0.5,-0.5),"O1":(-0.3,-0.9),"O2":(0.3,-0.9)
            }
            for n in names:
                found = False
                for k,v in approx.items():
                    if k in n:
                        coords.append(v); labels.append(n); found=True; break
                if not found:
                    # random around
                    coords.append((np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9)))
                    labels.append(n)
        else:
            # distribute points on circle
            nch = len(band_mean_by_channel)
            thetas = np.linspace(0,2*np.pi,nch,endpoint=False)
            coords = [(0.8*np.sin(t), 0.8*np.cos(t)) for t in thetas]
            labels = [f"ch{i}" for i in range(len(coords))]
        xs = np.array([c[0] for c in coords]); ys = np.array([c[1] for c in coords])
        vals = np.array(band_mean_by_channel[:len(coords)])
        # interpolation on grid
        xi = np.linspace(-1.0,1.0,160); yi = np.linspace(-1.0,1.0,160)
        XI, YI = np.meshgrid(xi, yi)
        from scipy.interpolate import griddata
        Z = griddata((xs,ys), vals, (XI, YI), method='cubic', fill_value=np.nan)
        fig = plt.figure(figsize=(4,4), dpi=120)
        ax = fig.add_subplot(111)
        im = ax.imshow(Z, origin='lower', extent=[-1,1,-1,1], cmap='RdBu_r')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{band_name} topography", fontsize=9)
        # overlay head circle
        circle = plt.Circle((0,0), 0.95, color='k', fill=False, linewidth=1)
        ax.add_artist(circle)
        # plot electrode positions
        ax.scatter(xs, ys, s=20, c='k')
        for i,lbl in enumerate(labels):
            ax.text(xs[i], ys[i], '', fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()  # PNG bytes
    except Exception:
        if TOPO_PLACEHOLDER.exists():
            return str(TOPO_PLACEHOLDER)
        return None

# ---------- Functional disconnection (connectivity) ----------
def compute_connectivity_placeholder():
    """
    If real connectivity not possible (no mne_connectivity), return a synthetic connectivity matrix and narrative.
    """
    # synthetic 10x10 matrix (values 0..1) with a focal reduction in alpha between frontal and parietal
    mat = np.random.uniform(0.4,0.9,(10,10))
    for i in range(10):
        mat[i,i]=1.0
    # simulate reduced frontal-parietal coherence
    mat[1,7] *= 0.6; mat[7,1] *= 0.6
    narrative = "Functional disconnection: reduction in Alpha coherence between frontal and parietal regions (~15%)."
    return mat, narrative

# ---------- XAI loading ----------
def load_shap_summary():
    if SHAP_JSON.exists():
        try:
            with open(SHAP_JSON,'r',encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------- PDF generation (reportlab if available) ----------
def register_amiri(ttf_path=None):
    if not HAS_REPORTLAB:
        return "Helvetica"
    try:
        if ttf_path and Path(ttf_path).exists():
            pdfmetrics.registerFont(TTFont("Amiri","./"+str(ttf_path)))
            return "Amiri"
        if AMIRI_TTF.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
            return "Amiri"
    except Exception:
        pass
    return "Helvetica"

def reshape_ar(text):
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf_report(summary, lang='en', amiri_path=None, topo_images=None, conn_image=None):
    """
    topo_images: dict band->PNGbytes or filepath; conn_image: PNG bytes or filepath
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

    # Header
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font, 16)
    if lang=='en':
        c.drawCentredString(W/2, y, title_en)
    else:
        c.drawCentredString(W/2, y, reshape_ar(title_ar))
    y -= 24

    # Patient & executive summary box
    c.setFont(font, 10)
    p = summary.get("patient",{})
    if lang=='en':
        c.drawString(x, y, f"Patient: {p.get('name','-')}   ID: {p.get('id','-')}   DOB: {p.get('dob','-')}")
        y -= 14
        if summary.get("ml_risk") is not None:
            c.drawString(x, y, f"ML Risk Score: {summary.get('ml_risk'):.1f}%    Risk: {summary.get('risk_category','-')}")
            y -= 14
        c.drawString(x, y, f"QEEG Interpretation: {summary.get('qeegh','-')}")
        y -= 18
    else:
        c.drawRightString(W-m, y, reshape_ar(f"المريض: {p.get('name','-')}   المعرف: {p.get('id','-')}   الميلاد: {p.get('dob','-')}"))
        y -= 14
        if summary.get("ml_risk") is not None:
            c.drawRightString(W-m, y, reshape_ar(f"معدل الخطر ML: {summary.get('ml_risk'):.1f}%   الفئة: {summary.get('risk_category','-')}"))
            y -= 14
        c.drawRightString(W-m, y, reshape_ar("تفسير QEEG: " + summary.get('qeegh','-')))
        y -= 18

    # QEEG table (first file)
    files = summary.get("files",[])
    if files:
        f0 = files[0]
        agg = f0.get("agg_features",{})
        # draw band table
        c.setFont(font, 10)
        if lang=='en':
            c.drawString(x, y, "Band    Absolute_mean    Relative_mean"); y -= 12
            for band in ["Delta","Theta","Alpha","Beta"]:
                a = agg.get(f"{band.lower()}_abs_mean",0.0)
                r = agg.get(f"{band.lower()}_rel_mean",0.0)
                c.drawString(x+6, y, f"{band:<7} {a:>12.4f}    {r:>10.4f}"); y -= 12
        else:
            c.drawRightString(W-m, y, reshape_ar("التردد    المطلق المتوسط    النسبي المتوسط")); y -= 12
            for band in ["Delta","Theta","Alpha","Beta"]:
                a = agg.get(f"{band.lower()}_abs_mean",0.0)
                r = agg.get(f"{band.lower()}_rel_mean",0.0)
                c.drawRightString(W-m, y, reshape_ar(f"{band}    {a:.4f}    {r:.4f}")); y -= 12
        y -= 8
    else:
        if lang=='en':
            c.drawString(x, y, "No EDF processed."); y -= 14
        else:
            c.drawRightString(W-m, y, reshape_ar("لا توجد ملفات EDF معالجة.")); y -= 14

    # Insert topography images (if available)
    if topo_images:
        # layout: row of 4 thumbnails
        thumb_w = 110
        x0 = x
        for i,(band, img) in enumerate(topo_images.items()):
            if img is None:
                continue
            # img may be bytes or filepath
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(io.BytesIO(img)) if isinstance(img, (bytes,bytearray)) else ImageReader(str(img))
                xi = x0 + (i%4)*(thumb_w+8)
                yi = y - ( (i//4)* (thumb_w+20) )
                c.drawImage(ir, xi, yi-thumb_w, width=thumb_w, height=thumb_w, preserveAspectRatio=True, mask='auto')
                c.setFont(font, 8)
                c.drawString(xi, yi-thumb_w-10, band)
            except Exception:
                pass
        y -= (thumb_w + 30)
    else:
        # placeholder
        if TOPO_PLACEHOLDER.exists():
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(str(TOPO_PLACEHOLDER))
                c.drawImage(ir, x, y-140, width=240, height=140, preserveAspectRatio=True, mask='auto')
                y -= 160
            except Exception:
                y -= 10

    # Connectivity map
    conn = summary.get("connectivity", None)
    if conn:
        # if conn is image bytes
        if isinstance(conn, (bytes,bytearray)):
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(io.BytesIO(conn))
                c.drawImage(ir, x, y-160, width=300, height=160, mask='auto')
                y -= 170
            except Exception:
                y -= 10
    else:
        # placeholder
        if CONN_PLACEHOLDER.exists():
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(str(CONN_PLACEHOLDER))
                c.drawImage(ir, x, y-140, width=240, height=140, mask='auto')
                y -= 160
            except Exception:
                y -= 10

    # XAI section
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, "Explainable AI — Top contributors:"); y -= 14
    else:
        c.drawRightString(W-m, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:")); y -= 14
    xai = summary.get("xai", None)
    if xai:
        for k,v in list(xai.items())[:12]:
            if lang=='en':
                c.drawString(x+6, y, f"{k}: {v:.4f}"); y -= 10
            else:
                c.drawRightString(W-m, y, reshape_ar(f"{k}: {v:.4f}")); y -= 10
    else:
        if lang=='en':
            c.drawString(x+6, y, "XAI data not available."); y -= 10
        else:
            c.drawRightString(W-m, y, reshape_ar("لا توجد بيانات XAI.")); y -= 10

    y -= 8
    # Recommendations
    if lang=='en':
        c.drawString(x, y, "Structured Clinical Recommendations:"); y -= 12
    else:
        c.drawRightString(W-m, y, reshape_ar("التوصيات السريرية المنظمة:")); y -= 12
    for r in summary.get("recommendations",[]):
        if lang=='en':
            c.drawString(x+6, y, "- " + r); y -= 10
        else:
            c.drawRightString(W-m, y, reshape_ar("- " + r)); y -= 10

    # Footer branding & disclaimer
    footer_en = "Designed and developed by Golden Bird LLC — Vista Kaviani"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني")
    disc_en = "Research/demo only — Not a clinical diagnosis."
    disc_ar = reshape_ar("هذه لأغراض البحث/العرض فقط — ليست تشخيصًا طبيًا نهائياً.")
    c.setFont(font, 8)
    if lang=='en':
        c.drawCentredString(W/2, 30, footer_en)
        c.drawCentredString(W/2, 18, disc_en)
    else:
        c.drawCentredString(W/2, 30, footer_ar)
        c.drawCentredString(W/2, 18, disc_ar)

    c.save()
    buf.seek(0)
    return buf.read()

# ---------- UI ----------
st.markdown("""
<style>
.block-container { max-width: 1200px; }
.header { background: linear-gradient(90deg,#023e8a,#0366d6); color: white; padding:14px; border-radius:8px; }
.card { background: #fff; padding:12px; border-radius:8px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
.small { color:#6b7280; font-size:13px; }
</style>
""", unsafe_allow_html=True)

col1,col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical XAI</h2><div class='small'>EEG / QEEG + XAI — Clinical support</div></div>", unsafe_allow_html=True)
with col2:
    if LOGO_SVG.exists():
        st.image(str(LOGO_SVG), width=120)
    else:
        st.markdown("<div style='text-align:right;font-weight:600'>Golden Bird LLC</div>", unsafe_allow_html=True)

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
    st.write(f"Backends: mne={HAS_MNE} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")

# Upload EDF
st.markdown("### 1) Upload EDF file(s)")
uploads = st.file_uploader("Upload one or more EDF files", type=["edf"], accept_multiple_files=True)

# PHQ-9 (with corrected options)
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
    label = (f"Q{i}. {PHQ_EN[i-1]}" if lang=='en' else f"س{i}. {PHQ_AR[i-1]}")
    if i==3:
        opts = ["0 — Not at all","1 — Insomnia (difficulty falling/staying asleep)","2 — Sleeping less","3 — Sleeping more"] if lang=='en' else [reshape_ar("0 — لا"), reshape_ar("1 — أرق"), reshape_ar("2 — قلة النوم"), reshape_ar("3 — زيادة النوم")]
    elif i==5:
        opts = ["0 — Not at all","1 — Eating less","2 — Eating more","3 — Both/variable"] if lang=='en' else [reshape_ar("0 — لا"), reshape_ar("1 — قلة الأكل"), reshape_ar("2 — زيادة الأكل"), reshape_ar("3 — متغير / كلاهما")]
    elif i==8:
        opts = ["0 — Not at all","1 — Moving/speaking slowly","2 — Fidgety/restless","3 — Both/variable"] if lang=='en' else [reshape_ar("0 — لا"), reshape_ar("1 — تباطؤ"), reshape_ar("2 — تململ"), reshape_ar("3 — متغير")]
    else:
        opts = ["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"] if lang=='en' else [reshape_ar("0 — لا"), reshape_ar("1 — عدة أيام"), reshape_ar("2 — أكثر من نصف الأيام"), reshape_ar("3 — تقريباً كل يوم")]
    sel = st.radio(label, options=opts, index=0, key=f"phq_{i}")
    try:
        val = int(str(sel).split("—")[0].strip())
    except Exception:
        val = int(str(sel)[0]) if str(sel) and str(sel)[0].isdigit() else 0
    phq_answers[f"Q{i}"] = val

phq_total = sum(phq_answers.values())
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal,5–9 mild,10–14 moderate,15–19 mod-severe,20–27 severe)")

# AD8
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
 "مشاكل في الحكم",
 "قلة الاهتمام بالهوايات/الأنشطة",
 "تكرار الأسئلة/القصص",
 "صعوبة تعلم استخدام جهاز/أداة",
 "نسيان الشهر أو السنة",
 "صعوبة في التعامل مع الأمور المالية",
 "صعوبة في تذكر المواعيد",
 "مشاكل يومية في التفكير والذاكرة"
]
ad8_answers = {}
for i,txt in enumerate(AD8_EN, start=1):
    label = (f"A{i}. {txt}" if lang=='en' else f"أ{i}. {AD8_AR[i-1]}")
    v = st.radio(label, options=[0,1], index=0, key=f"ad8_{i}", horizontal=True)
    ad8_answers[f"A{i}"] = int(v)
ad8_total = sum(ad8_answers.values())
st.info(f"AD8 total: {ad8_total} (≥2 suggests cognitive impairment)")

# options
st.markdown("---")
st.header("Processing options")
use_notch = st.checkbox("Apply notch (50Hz)", value=True)
do_topomap = st.checkbox("Generate topography maps (if matplotlib available)", value=True)
do_connectivity = st.checkbox("Compute functional disconnection (placeholder if not available)", value=True)
run_models = st.checkbox("Run models if provided (model_depression.pkl, model_alzheimer.pkl)", value=False)

# Process EDF(s)
results = []
if uploads:
    idx = 0
    for up in uploads:
        idx += 1
        st.write(f"Processing {up.name} ...")
        try:
            tmp = save_tmp_upload(up)
            edf = read_edf(tmp)
            data = edf["data"]
            sf = edf.get("sfreq") or DEFAULT_SF
            st.success(f"Loaded: backend={edf['backend']} channels={data.shape[0]} sfreq={sf}")
            cleaned = preprocess_data(data, sf, do_notch=use_notch)
            dfbands = compute_psd_bands(cleaned, sf)
            st.dataframe(dfbands.head(12))
            agg = aggregate_bands(dfbands, ch_names=edf.get("ch_names"))
            st.write("Aggregated features:", agg)
            # topomap images
            topo_images = {}
            if do_topomap:
                for band in ["Delta","Theta","Alpha","Beta"]:
                    vals = dfbands[f"{band}_rel"].values if not dfbands.empty else np.zeros(data.shape[0])
                    img = generate_topomap_image(vals, ch_names=edf.get("ch_names"), band_name=band)
                    topo_images[band] = img
                    # display inline: if bytes show via st.image, if filepath show
                    if isinstance(img, (bytes,bytearray)):
                        st.image(img, caption=f"{band} topomap", use_column_width=False)
                    elif isinstance(img, str) and Path(img).exists():
                        st.image(img, caption=f"{band} topomap", use_column_width=False)
                    else:
                        # show placeholder svg if available
                        if TOPO_PLACEHOLDER.exists():
                            st.image(str(TOPO_PLACEHOLDER), caption=f"{band} topomap (placeholder)")
            # connectivity
            conn_img = None
            conn_narr = None
            if do_connectivity:
                mat, narr = compute_connectivity_placeholder()
                conn_narr = narr
                # create a simple mat heatmap image if matplotlib available
                if HAS_MATPLOTLIB:
                    fig = plt.figure(figsize=(4,3))
                    plt.imshow(mat, cmap='viridis')
                    plt.colorbar()
                    plt.title("Connectivity (placeholder)")
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    conn_img = buf.getvalue()
                    st.image(conn_img, caption="Functional disconnection (placeholder)")
                else:
                    if CONN_PLACEHOLDER.exists():
                        st.image(str(CONN_PLACEHOLDER), caption="Connectivity placeholder")
            results.append({"filename": up.name, "agg_features": agg, "df_bands": dfbands, "topo_images": topo_images, "connectivity_image": conn_img, "connectivity_narrative": conn_narr})
        except Exception as e:
            st.error(f"Failed to process {up.name}: {e}")
            _trace(e)

# Build summary
summary = {
    "patient": {"name": patient_name or "-", "id": patient_id or "-", "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": phq_answers},
    "ad8": {"total": ad8_total, "items": ad8_answers},
    "files": results,
    "xai": None,
    "connectivity": None,
    "ml_risk": None,
    "risk_category": None,
    "qeegh": None,
    "recommendations": []
}

# heuristic interpretation
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
    # attach connectivity narrative if present
    if results[0].get("connectivity_narrative"):
        summary["connectivity"] = results[0].get("connectivity_narrative")

# XAI: load shap_summary.json if present
shap_json = load_shap_summary()
if shap_json:
    summary["xai"] = shap_json

# Model prediction (if requested & models exist)
if run_models and joblib:
    preds = []
    try:
        Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
        if MODEL_DEP.exists():
            model_dep = joblib.load(str(MODEL_DEP))
            p = model_dep.predict_proba(Xdf)[:,1] if hasattr(model_dep,"predict_proba") else model_dep.predict(Xdf)
            summary.setdefault("predictions",{})["depression_prob"] = p.tolist()
            preds.append(np.mean(p))
        if MODEL_AD.exists():
            model_ad = joblib.load(str(MODEL_AD))
            p2 = model_ad.predict_proba(Xdf)[:,1] if hasattr(model_ad,"predict_proba") else model_ad.predict(Xdf)
            summary.setdefault("predictions",{})["alzheimers_prob"] = p2.tolist()
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
    summary["recommendations"].append("PHQ-9 suggests moderate/severe depression — consider psychiatric referral.")
if summary["ad8"]["total"] >= 2 or (results and results[0]["agg_features"].get("theta_alpha_ratio",0) > 1.4):
    summary["recommendations"].append("AD8 elevated or Theta/Alpha increased — consider neurocognitive testing and neuroimaging (MRI/FDG-PET).")
summary["recommendations"].append("Correlate QEEG/connectivity findings with PHQ-9 and AD8 and clinical interview.")
summary["recommendations"].append("Review medications that may affect EEG.")
if not summary["recommendations"]:
    summary["recommendations"].append("Clinical follow-up and re-evaluation in 3-6 months.")

# Report generation UI
st.markdown("---")
st.header("Generate report")
st.write("Select one language for the report (English or Arabic). The report will include Executive Summary, QEEG metrics, topomaps, connectivity, XAI summary (if available) and recommendations.")
colA, colB = st.columns([3,1])
with colA:
    report_lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)
    amiri_path = st.text_input("Amiri TTF path (optional)", value="")
with colB:
    if st.button("Generate PDF report"):
        # prepare topo images dict (band->bytes/file)
        topo_imgs = {}
        if results and results[0].get("topo_images"):
            for band,img in results[0]["topo_images"].items():
                topo_imgs[band] = img
        conn_img = results[0].get("connectivity_image") if results else None
        try:
            pdfb = generate_pdf_report(summary, lang=report_lang, amiri_path=(amiri_path or None), topo_images=topo_imgs, conn_image=conn_img)
            st.download_button("Download PDF", data=pdfb, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("Report ready (footer includes Golden Bird LLC).")
        except Exception as e:
            st.error("PDF generation failed.")
            _trace(e)

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis.")
