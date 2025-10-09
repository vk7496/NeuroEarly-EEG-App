# app_neuroearly_pro_clinical.py
"""
NeuroEarly Pro — Clinical (Pro edition)
- Polished UI
- Bilingual PDF (EN / AR with Amiri if available)
- EDF reading (mne or pyedflib fallback)
- Preprocess: notch + bandpass + ICA(if mne)
- QEEG features, ratios, alpha asymmetry
- Optional connectivity & microstate
- XAI via SHAP if available (fallback: precomputed or placeholder)
- Model hooks: model_depression.pkl, model_alzheimer.pkl (joblib)
- Branding: Golden Bird LLC in footer
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro — Clinical (Pro)", layout="wide", initial_sidebar_state="expanded")

# core
import os, io, json, tempfile, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Safe optional imports (no ModuleNotFound crash)
HAS_MNE = False
HAS_PYEDF = False
HAS_CONN = False
HAS_SHAP = False
HAS_SKLEARN = False
HAS_REPORTLAB = False
HAS_ARABIC_TOOLS = False
HAS_XGBOOST = False

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
    import mne_connectivity as mne_conn
    HAS_CONN = True
except Exception:
    HAS_CONN = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

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
    import xgboost
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# plotting fallback: use streamlit built-ins
from scipy.signal import welch, butter, filtfilt, iirnotch

# joblib for model load
try:
    import joblib
except Exception:
    joblib = None

# ---------- Constants ----------
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
DEFAULT_NOTCH = 50.0
AMIRI_TTF = "Amiri-Regular.ttf"

# ---------- Utility helpers ----------
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def show_traceback(e):
    tb = traceback.format_exc()
    st.error("Internal error — see details.")
    st.code(tb)
    print(tb)

def save_tmp_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

# ---------- EDF loader ----------
def read_edf_generic(path):
    """
    Returns dict:
      backend: 'mne' or 'pyedflib'
      data: np.ndarray (n_channels, n_samples)
      ch_names: list
      sfreq: float
      raw: mne.Raw if mne else None
    """
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        return {"backend":"mne", "data":data, "ch_names": raw.ch_names, "sfreq": raw.info.get("sfreq", None), "raw": raw}
    elif HAS_PYEDF:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        chs = f.getSignalLabels()
        sf = f.getSampleFrequency(0) if f.getSampleFrequency(0) else None
        sigs = [f.readSignal(i).astype(np.float64) for i in range(n)]
        f._close()
        data = np.vstack(sigs)
        return {"backend":"pyedflib", "data":data, "ch_names": chs, "sfreq": sf, "raw": None}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ---------- Preprocessing ----------
def notch_filter(sig, sfreq, freq=DEFAULT_NOTCH, Q=30.0):
    if sfreq is None or sfreq <= 0:
        return sig
    b,a = iirnotch(freq, Q, sfreq)
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def bandpass_filter(sig, sfreq, l=0.5, h=45.0, order=4):
    if sfreq is None or sfreq <= 0:
        return sig
    nyq = 0.5 * sfreq
    low = max(l/nyq, 1e-6)
    high = min(h/nyq, 0.999)
    b,a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def preprocess_full(edfobj, do_notch=True, notch_freq=DEFAULT_NOTCH, l=0.5, h=45.0):
    data = edfobj["data"].astype(np.float64)
    sf = edfobj.get("sfreq") or 256.0
    cleaned = np.zeros_like(data)
    for i in range(data.shape[0]):
        x = data[i]
        if do_notch:
            x = notch_filter(x, sf, freq=notch_freq)
        x = bandpass_filter(x, sf, l=l, h=h)
        cleaned[i] = x
    return cleaned, sf

# ---------- ICA artifact attempt ----------
def run_ica(edfobj, n_components=15):
    if not HAS_MNE or edfobj.get("backend") != "mne":
        return edfobj["data"], {"ica_status":"not_available"}
    try:
        raw = edfobj["raw"]
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, verbose=False)
        ica.fit(raw, picks=picks, verbose=False)
        # conservative: do not auto-remove components; return fitted info
        return raw.get_data(), {"ica_status":"fitted", "n_components": ica.n_components_}
    except Exception as e:
        return edfobj["data"], {"ica_status":f"failed:{str(e)}"}

# ---------- PSD & band features ----------
def compute_band_powers_matrix(data, sfreq, picks=None, nperseg=1024):
    idxs = picks if picks is not None else list(range(data.shape[0]))
    rows = []
    for i in idxs:
        sig = data[i]
        try:
            freqs, pxx = welch(sig, fs=sfreq, nperseg=min(nperseg, max(256,len(sig))))
        except Exception:
            freqs = np.array([])
            pxx = np.array([])
        total = float(np.trapz(pxx, freqs)) if freqs.size>0 else 0.0
        row = {"channel_idx": i}
        for b,(lo,hi) in BANDS.items():
            if freqs.size==0:
                abs_p = 0.0
            else:
                mask = (freqs>=lo) & (freqs<=hi)
                abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum()>0 else 0.0
            rel = float(abs_p/total) if total>0 else 0.0
            row[f"{b}_abs"] = abs_p
            row[f"{b}_rel"] = rel
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_band_features(df_bands, ch_names=None):
    if df_bands.empty:
        return {}
    agg = {
        "alpha_rel_mean": float(df_bands['alpha_rel'].mean()),
        "beta_rel_mean": float(df_bands['beta_rel'].mean()),
        "theta_rel_mean": float(df_bands['theta_rel'].mean()),
        "delta_rel_mean": float(df_bands['delta_rel'].mean()),
        "theta_alpha_ratio": float((df_bands['theta_rel'].mean()) / (df_bands['alpha_rel'].mean()+1e-9)),
        "theta_beta_ratio": float((df_bands['theta_rel'].mean()) / (df_bands['beta_rel'].mean()+1e-9))
    }
    # alpha asymmetry best-effort: if ch_names contain F3/F4 or Fp1/Fp2
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def find_idx(token_list):
                for i,n in enumerate(names):
                    for t in token_list:
                        if t in n:
                            return i
                return None
            i_f3 = find_idx(["F3"]); i_f4 = find_idx(["F4"])
            if i_f3 is not None and i_f4 is not None:
                v3 = df_bands.loc[df_bands['channel_idx']==i_f3,'alpha_rel'].values
                v4 = df_bands.loc[df_bands['channel_idx']==i_f4,'alpha_rel'].values
                if v3.size>0 and v4.size>0:
                    agg['alpha_asym_F3_F4'] = float(v3[0]-v4[0])
        except Exception:
            pass
    return agg

# ---------- Connectivity (optional) ----------
def compute_connectivity_summary(raw, sfreq, method="pli", fmin=8, fmax=13):
    if not (HAS_CONN and HAS_MNE):
        raise ImportError("Connectivity backend not available")
    from mne_connectivity import spectral_connectivity
    try:
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events=events, tmin=0.0, tmax=1.0-1.0/raw.info['sfreq'], baseline=None, preload=True, verbose=False)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method=method, sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        return {"shape": con.shape, "freqs": list(freqs)}
    except Exception as e:
        raise

# ---------- Microstate (optional) ----------
def microstate_summary(data, sfreq, n_states=4):
    if not HAS_SKLEARN:
        return {"status":"sklearn_not_available"}
    try:
        gfp = data.std(axis=0)
        thr = np.percentile(gfp, 95)
        peaks = np.where(gfp >= thr)[0]
        if peaks.size == 0:
            return {"status":"no_peaks"}
        maps = data[:, peaks].T
        scaler = StandardScaler()
        maps_s = scaler.fit_transform(maps)
        kmeans = KMeans(n_clusters=n_states, random_state=42).fit(maps_s)
        centers = kmeans.cluster_centers_
        return {"status":"ok", "n_peaks": int(peaks.size), "centers": centers.tolist()}
    except Exception as e:
        return {"status": f"failed:{str(e)}"}

# ---------- Model load & XAI ----------
def load_model_safe(path):
    try:
        if joblib and os.path.exists(path):
            return joblib.load(path)
    except Exception:
        pass
    return None

def compute_shap_summary(model, Xdf):
    if not HAS_SHAP:
        return None
    try:
        if hasattr(model, "booster") or model.__class__.__name__.lower().startswith("xgb"):
            expl = shap.TreeExplainer(model)
            vals = expl.shap_values(Xdf)
            # aggregate mean abs
            if isinstance(vals, np.ndarray):
                mean_abs = np.abs(vals).mean(axis=0)
                return dict(zip(Xdf.columns, mean_abs.tolist()))
            else:
                try:
                    arr = np.abs(vals.values).mean(axis=0)
                    return dict(zip(Xdf.columns, arr.tolist()))
                except Exception:
                    return None
        else:
            expl = shap.Explainer(model.predict, Xdf)
            sv = expl(Xdf)
            try:
                arr = np.abs(sv.values).mean(axis=0)
                return dict(zip(Xdf.columns, arr.tolist()))
            except Exception:
                return None
    except Exception as e:
        print("SHAP error:", e)
        return None

# ---------- PDF generation ----------
def register_font_amiri(ttf_path=None):
    """
    Try to register Amiri TTF if available; return font name to use.
    """
    try:
        if not HAS_REPORTLAB:
            return "Helvetica"
        if ttf_path and Path(ttf_path).exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(ttf_path)))
            return "Amiri"
        p = Path("./Amiri-Regular.ttf")
        if p.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(p)))
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

def generate_professional_pdf(summary, lang='en', shap_local=None, amiri_path=None):
    """
    Professional PDF with Executive Summary, QEEG table, XAI, Structured Recommendations.
    Returns bytes.
    """
    if not HAS_REPORTLAB:
        # fallback: return json bytes
        return json.dumps(summary, indent=2, ensure_ascii=False).encode('utf-8')

    font_name = register_font_amiri(amiri_path)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 40
    x = margin
    y = H - margin

    # Header block
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font_name, 18)
    if lang=='en':
        c.drawCentredString(W/2, y, title_en)
    else:
        c.drawCentredString(W/2, y, reshape_ar(title_ar))
    y -= 30

    # Executive Summary (box)
    def draw_boxed_heading(title):
        nonlocal y
        c.setFont(font_name, 12)
        if lang=='en':
            c.drawString(x, y, title)
        else:
            c.drawRightString(W-margin, y, reshape_ar(title))
        y -= 14

    # Executive summary content
    draw_boxed_heading("Executive Summary / الملخص التنفيذي")
    c.setFont(font_name, 10)
    p = summary.get("patient",{})
    if lang=='en':
        c.drawString(x, y, f"Patient: {p.get('name','-')}  | ID: {p.get('id','-')}  | DOB: {p.get('dob','-')}")
        y -= 12
        # ML risk
        risk = summary.get("ml_risk", None)
        if risk is not None:
            c.drawString(x, y, f"ML Risk Score: {risk:.2f}%")
            y -= 12
        rc = summary.get("risk_category", None)
        if rc:
            c.drawString(x, y, f"Risk Category: {rc}")
            y -= 12
        ps = summary.get("primary_suggestion", "")
        if ps:
            c.drawString(x, y, f"Primary Suggestion: {ps}")
            y -= 12
        # QEEG heuristic
        qh = summary.get("qeegh", "")
        if qh:
            c.drawString(x, y, f"QEEG Interpretation: {qh}")
            y -= 12
    else:
        # Arabic right-aligned
        c.drawRightString(W-margin, y, reshape_ar(f"المريض: {p.get('name','-')}  المعرف: {p.get('id','-')}  الميلاد: {p.get('dob','-')}"))
        y -= 12
        risk = summary.get("ml_risk", None)
        if risk is not None:
            c.drawRightString(W-margin, y, reshape_ar(f"معدل الخطر ML: {risk:.2f}%"))
            y -= 12
        rc = summary.get("risk_category", None)
        if rc:
            c.drawRightString(W-margin, y, reshape_ar(f"فئة الخطر: {rc}"))
            y -= 12
        ps = summary.get("primary_suggestion", "")
        if ps:
            c.drawRightString(W-margin, y, reshape_ar(f"التوصية الأساسية: {ps}"))
            y -= 12
        qh = summary.get("qeegh", "")
        if qh:
            c.drawRightString(W-margin, y, reshape_ar(qh))
            y -= 12
    y -= 6

    # QEEG Metrics table
    draw_boxed_heading("QEEG Key Metrics / مؤشرات QEEG الأساسية")
    c.setFont(font_name, 10)
    # If files present, list first file metrics
    if summary.get("files"):
        f0 = summary["files"][0]
        # Table-like layout
        metrics = f0.get("agg_features", {})
        table_lines = [
            ("Theta/Alpha Ratio", f"{metrics.get('theta_alpha_ratio', 'N/A'):.3f}" if metrics.get('theta_alpha_ratio') else "N/A"),
            ("Theta/Beta Ratio", f"{metrics.get('theta_beta_ratio', 'N/A'):.3f}" if metrics.get('theta_beta_ratio') else "N/A"),
            ("Alpha mean (rel)", f"{metrics.get('alpha_rel_mean', 0):.4f}"),
            ("Theta mean (rel)", f"{metrics.get('theta_rel_mean', 0):.4f}"),
            ("Alpha Asymmetry (F3-F4)", f"{metrics.get('alpha_asym_F3_F4','N/A')}")
        ]
        for label, val in table_lines:
            if lang=='en':
                c.drawString(x+6, y, f"{label}: {val}")
            else:
                c.drawRightString(W-margin, y, reshape_ar(f"{label}: {val}"))
            y -= 12
    else:
        if lang=='en':
            c.drawString(x+6, y, "No EDF processed yet.")
        else:
            c.drawRightString(W-margin, y, reshape_ar("لا توجد ملفات EDF معالجة بعد."))
        y -= 14
    y -= 6

    # XAI summary top features
    draw_boxed_heading("Explainable AI (XAI) — Top contributors / الذكاء الاصطناعي القابل للتفسير")
    c.setFont(font_name, 10)
    xai_block = summary.get("xai_summary", {})
    if xai_block:
        # show top contributors
        for feat,imp in list(xai_block.items())[:10]:
            if lang=='en':
                c.drawString(x+6, y, f"{feat}: {imp:.4f}")
            else:
                c.drawRightString(W-margin, y, reshape_ar(f"{feat}: {imp:.4f}"))
            y -= 10
            if y < 100:
                c.showPage(); y = H - margin
    else:
        if lang=='en':
            c.drawString(x+6, y, "XAI not available (SHAP not installed or no model).")
        else:
            c.drawRightString(W-margin, y, reshape_ar("XAI غير متاح (SHAP غير مثبت أو لا يوجد نموذج)."))
        y -= 14
    y -= 6

    # Structured recommendations
    draw_boxed_heading("Structured Clinical Recommendations / التوصيات السريرية المنظمة")
    c.setFont(font_name, 10)
    recs = summary.get("recommendations", [])
    if recs:
        for r in recs:
            if lang=='en':
                c.drawString(x+6, y, f"- {r}")
            else:
                c.drawRightString(W-margin, y, reshape_ar(f"- {r}"))
            y -= 10
            if y < 100:
                c.showPage(); y = H - margin
    else:
        if lang=='en':
            c.drawString(x+6, y, "No recommendations generated.")
        else:
            c.drawRightString(W-margin, y, reshape_ar("لا توجد توصيات."))
        y -= 12

    # Footer branding
    footer_en = "Designed and developed by Golden Bird LLC — Vista Kaviani | Muscat, Sultanate of Oman"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني | مسقط، سلطنة عمان")
    c.setFont(font_name, 9)
    if lang=='en':
        c.drawCentredString(W/2, 30, footer_en)
    else:
        c.drawCentredString(W/2, 30, footer_ar)
    c.setFont(font_name, 8)
    disc = "Research/demo only — Not a clinical diagnosis."
    c.drawCentredString(W/2, 18, reshape_ar(disc) if lang!='en' else disc)

    c.save()
    buf.seek(0)
    return buf.read()

# ---------- UI: polished layout ----------
# Top header bar
st.markdown("""
<style>
body { background-color: #f7fafc; }
.header-card {background: linear-gradient(90deg, #0f172a, #1e293b); color: white; padding: 18px; border-radius: 10px;}
.small-muted { color: #6b7280; font-size:12px; }
.card { background: white; padding: 14px; border-radius:10px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.metric-green { color: #16a34a; font-weight:600; }
.metric-amber { color: #f59e0b; font-weight:600; }
.metric-red { color: #ef4444; font-weight:600; }
</style>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([4,1])
with col_a:
    st.markdown("<div class='header-card'><h1 style='margin:0'>🧠 NeuroEarly Pro — Clinical (Pro)</h1><div class='small-muted'>EEG/QEEG + XAI — Professional clinical reporting</div></div>", unsafe_allow_html=True)
with col_b:
    st.markdown("<div style='text-align:right'><img src='https://raw.githubusercontent.com/vk7496/placeholder/main/logo.png' width='84' /></div>", unsafe_allow_html=True)

st.markdown("")  # spacing

# Sidebar controls
with st.sidebar:
    st.header("Settings & Patient")
    lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)
    st.markdown("---")
    st.subheader("Patient")
    patient_name = st.text_input("Name / اسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB")
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))
    st.markdown("---")
    st.write("Model files (optional): put model_depression.pkl & model_alzheimer.pkl in app root")
    st.write(f"mne installed: {HAS_MNE} | pyedflib: {HAS_PYEDF} | shap: {HAS_SHAP}")

# Main content
st.markdown("### 1) Upload EDF files")
uploads = st.file_uploader("Drag & drop EDF files here (.edf)", type=["edf"], accept_multiple_files=True)

# PHQ-9 UI (corrected)
st.markdown("### 2) PHQ-9 (Depression screening)")
PHQ9_ITEMS = [
 "Little interest or pleasure in doing things",
 "Feeling down, depressed, or hopeless",
 "Trouble falling/staying asleep, or sleeping too much",
 "Feeling tired or having little energy",
 "Poor appetite or overeating",
 "Feeling bad about yourself — or that you are a failure",
 "Trouble concentrating on things, such as reading or watching TV",
 "Moving or speaking slowly OR being fidgety/restless",
 "Thoughts that you would be better off dead or of harming yourself"
]
phq = {}
cols = st.columns(3)
for i,item in enumerate(PHQ9_ITEMS, start=1):
    with cols[(i-1)%3]:
        phq[f"q{i}"] = st.radio(f"Q{i}", [0,1,2,3], index=0, key=f"phq{i}", horizontal=True)
phq_total = sum(phq.values())
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 mod-severe, 20–27 severe)")

# AD8 UI
st.markdown("### 3) AD8 (Cognitive screening)")
AD8_ITEMS = [
 "Problems with judgment (e.g., problems making decisions)",
 "Less interest in hobbies/activities",
 "Repeats questions/stories",
 "Trouble learning to use a tool, appliance, or gadget",
 "Forgetting the correct month or year",
 "Difficulty handling complicated financial affairs",
 "Trouble remembering appointments",
 "Daily problems with thinking and memory"
]
ad8 = {}
for i,item in enumerate(AD8_ITEMS, start=1):
    ad8[f"a{i}"] = st.radio(f"A{i}", [0,1], index=0, key=f"ad8_{i}", horizontal=True)
ad8_total = sum(ad8.values())
st.info(f"AD8 total: {ad8_total} (score ≥2 suggests cognitive impairment)")

# Processing options
st.markdown("---")
st.header("Processing options & actions")
do_ica = st.checkbox("Attempt ICA (if mne available)", value=False)
do_conn = st.checkbox("Compute connectivity (if available)", value=False)
do_micro = st.checkbox("Microstate analysis (if sklearn available)", value=False)
run_models = st.checkbox("Run models (if model files present)", value=True)
amiri_path_input = st.text_input("Amiri TTF path (optional)", value="")

# Process
results = []
if uploads:
    st.markdown("### Processing EDF(s)...")
    prog_bar = st.progress(0)
    for idx, up in enumerate(uploads, start=1):
        try:
            st.markdown(f"**File:** {up.name} ({up.size/1024/1024:.2f} MB)")
            tmp = save_tmp_upload(up)
            edf = read_edf_generic(tmp)
            st.success(f"Loaded (backend: {edf.get('backend')}) channels: {len(edf.get('ch_names',[]))}  sfreq: {edf.get('sfreq','-')}")
            data_pre, sf = preprocess_full(edf)
            ica_info = None
            if do_ica:
                data_pre, ica_info = run_ica(edf)
                st.write("ICA status:", ica_info)
            df_bands = compute_band_powers_matrix(data_pre, sf)
            st.dataframe(df_bands.head(10))
            agg = aggregate_band_features(df_bands, ch_names=edf.get('ch_names'))
            st.write("Aggregated features:", agg)
            conn_summary = None
            if do_conn and HAS_CONN and HAS_MNE and edf.get("backend")=="mne":
                try:
                    conn_summary = compute_connectivity_summary(edf["raw"], sf)
                    st.write("Connectivity:", conn_summary)
                except Exception as e:
                    st.warning("Connectivity failed: " + str(e))
            micro = None
            if do_micro and HAS_SKLEARN:
                micro = microstate_summary(data_pre, sf)
                st.write("Microstate:", micro)
            results.append({
                "filename": up.name,
                "raw_summary": {"n_channels": int(data_pre.shape[0]), "sfreq": float(sf)},
                "df_bands": df_bands,
                "agg_features": agg,
                "connectivity": conn_summary,
                "microstate": micro
            })
        except Exception as e:
            st.error(f"Failed processing {up.name}: {e}")
            show_traceback(e)
        prog_bar.progress(int(100*idx/len(uploads)))
    prog_bar.empty()

# Build full summary
full_summary = {
    "patient": {"name": patient_name, "id": patient_id, "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": phq},
    "ad8": {"total": ad8_total, "items": ad8},
    "files": results,
    "ml_risk": None,
    "risk_category": None,
    "primary_suggestion": None,
    "qeegh": None,
    "predictions": {},
    "xai_summary": {},
    "recommendations": []
}

# Simple heuristic QEEG text (for Executive Summary) — example
if results:
    # pick first file agg
    af = results[0].get("agg_features", {})
    if af:
        # example heuristic
        t_a = af.get("theta_alpha_ratio", None)
        if t_a:
            if t_a > 1.4:
                full_summary["qeegh"] = "Elevated Theta/Alpha ratio consistent with early cognitive slowing."
            elif t_a > 1.1:
                full_summary["qeegh"] = "Mild elevation of Theta/Alpha ratio; correlate clinically."
            else:
                full_summary["qeegh"] = "Theta/Alpha within expected range."

# Load models (optional) and compute predictions + XAI
if run_models and results:
    model_dep = load_model_safe("model_depression.pkl")
    model_ad = load_model_safe("model_alzheimer.pkl")
    Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
    if model_dep is not None:
        try:
            preds = model_dep.predict_proba(Xdf)[:,1] if hasattr(model_dep, "predict_proba") else model_dep.predict(Xdf)
            full_summary["predictions"]["depression_probabilities"] = preds.tolist()
            # ml risk combine (simple avg)
            full_summary["ml_risk"] = float(np.mean(preds))*100.0
        except Exception:
            full_summary["predictions"]["depression_probabilities"] = None
    if model_ad is not None:
        try:
            preds2 = model_ad.predict_proba(Xdf)[:,1] if hasattr(model_ad, "predict_proba") else model_ad.predict(Xdf)
            full_summary["predictions"]["alzheimers_probabilities"] = preds2.tolist()
            if full_summary["ml_risk"] is None:
                full_summary["ml_risk"] = float(np.mean(preds2))*100.0
            else:
                # combine by average
                full_summary["ml_risk"] = float((full_summary["ml_risk"]/100.0 + np.mean(preds2))/2.0 * 100.0)
        except Exception:
            full_summary["predictions"]["alzheimers_probabilities"] = None

    # risk categorization
    if full_summary["ml_risk"] is not None:
        r = full_summary["ml_risk"]
        if r >= 50:
            full_summary["risk_category"] = "High"
        elif r >= 25:
            full_summary["risk_category"] = "Moderate"
        else:
            full_summary["risk_category"] = "Low"
        # primary suggestion based on risk
        if full_summary["risk_category"] == "High":
            full_summary["primary_suggestion"] = "Urgent neurological referral and imaging recommended."
        elif full_summary["risk_category"] == "Moderate":
            full_summary["primary_suggestion"] = "Clinical follow-up and further cognitive testing recommended (AD8, MMSE)."
        else:
            full_summary["primary_suggestion"] = "Routine monitoring; correlate with PHQ-9 / AD8."

    # SHAP summary
    shap_sum_dep = None
    shap_sum_ad = None
    if HAS_SHAP:
        try:
            if model_dep is not None:
                sdep = compute_shap_summary(model_dep, Xdf)
                if sdep:
                    full_summary["xai_summary"]["depression_global"] = sdep
            if model_ad is not None:
                sad = compute_shap_summary(model_ad, Xdf)
                if sad:
                    full_summary["xai_summary"]["alzheimers_global"] = sad
        except Exception as e:
            print("SHAP compute error:", e)

# Simple rule-based recommendations (expandable)
if full_summary["qeegh"]:
    if "Elevated Theta/Alpha" in full_summary["qeegh"] or (full_summary.get("ml_risk") and full_summary["ml_risk"]>25):
        full_summary["recommendations"].append("Correlate QEEG with AD8 and formal cognitive testing (MMSE).")
        full_summary["recommendations"].append("Check B12 and TSH to rule out reversible causes.")
        full_summary["recommendations"].append("Consider MRI or FDG-PET if ML risk > 25% and Theta/Alpha > 1.4.")
    else:
        full_summary["recommendations"].append("Clinical follow-up and re-evaluation in 3-6 months.")

# UI: show summary and actions
st.markdown("---")
st.header("Summary & Actions")
colL, colR = st.columns([2,1])
with colL:
    st.subheader("Patient summary")
    st.write(full_summary["patient"])
    st.subheader("ML Risk & QEEG")
    if full_summary["ml_risk"] is not None:
        val = full_summary["ml_risk"]
        if val < 25:
            clr = "metric-green"
        elif val < 50:
            clr = "metric-amber"
        else:
            clr = "metric-red"
        st.markdown(f"<div class='{clr}' style='font-size:20px'>ML Risk Score: {val:.1f}% — {full_summary.get('risk_category','-')}</div>", unsafe_allow_html=True)
    else:
        st.info("No model predictions available. Place model files to enable predictions.")
    st.markdown("**QEEG narrative:**")
    st.write(full_summary.get("qeegh","-"))
    st.markdown("**Recommendations:**")
    for r in full_summary.get("recommendations",[]):
        st.write("- " + r)

with colR:
    st.subheader("Files processed")
    st.write(f"{len(results)} file(s)")
    if results:
        # show aggregated metrics table
        rows = []
        for r in results:
            af = r.get("agg_features",{})
            rows.append({
                "file": r.get("filename"),
                "alpha_rel": af.get("alpha_rel_mean","-"),
                "theta_rel": af.get("theta_rel_mean","-"),
                "theta/alpha": af.get("theta_alpha_ratio","-")
            })
        st.table(pd.DataFrame(rows))

# XAI visualization (simple)
if full_summary.get("xai_summary"):
    st.markdown("### XAI — Top global features")
    xs = full_summary["xai_summary"]
    # prefer depression global then alzheimers
    xai_display = xs.get("depression_global") or xs.get("alzheimers_global") or {}
    if xai_display:
        feat_df = pd.Series(xai_display).sort_values(ascending=False).head(10)
        st.bar_chart(feat_df)
    else:
        st.info("XAI: SHAP not computed or no model available.")

# Report generation
st.markdown("---")
st.header("Generate Professional Report (PDF)")
st.markdown("Executive summary, QEEG metrics, XAI, and structured recommendations will be included.")
if results:
    colp1, colp2 = st.columns([3,1])
    with colp1:
        lang_choice = st.selectbox("Report language", options=["en","ar"], index=0)
        amiri_path = st.text_input("Optional Amiri TTF path (leave empty if Amiri-Regular.ttf in root)", value="")
    with colp2:
        if st.button("Generate PDF"):
            try:
                # prepare xai summary (top features)
                xai_for_pdf = {}
                if full_summary.get("xai_summary"):
                    # merge depression and alzheimers if present
                    if "depression_global" in full_summary["xai_summary"]:
                        xai_for_pdf = dict(sorted(full_summary["xai_summary"]["depression_global"].items(), key=lambda x:-x[1])[:12])
                    elif "alzheimers_global" in full_summary["xai_summary"]:
                        xai_for_pdf = dict(sorted(full_summary["xai_summary"]["alzheimers_global"].items(), key=lambda x:-x[1])[:12])
                pdf_bytes = generate_professional_pdf(full_summary, lang=lang_choice, shap_local=xai_for_pdf, amiri_path=(amiri_path or None))
                st.download_button("Download PDF Report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("PDF generation failed.")
                show_traceback(e)
else:
    st.info("Process at least one EDF file to enable report generation.")

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis. Final clinical decisions must be made by a qualified physician.")
