# app_neuroearly_final.py
"""
NeuroEarly Pro — Final (Cloud-aware, bilingual EN/AR, Amiri RTL PDF)
Place models (model_depression.pkl, model_alzheimer.pkl) and optionally
Amiri-Regular.ttf in app root.
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro — Clinical XAI", layout="wide")

# core imports
import os, io, json, tempfile, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Safe optional imports ----------
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
    import joblib
except Exception:
    joblib = None

# ReportLab + Arabic shaping
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

# XGBoost optional detection
try:
    import xgboost
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from scipy.signal import welch, butter, filtfilt, iirnotch

# ---------- Helpers ----------
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def log_exc(e):
    tb = traceback.format_exc()
    st.error("Internal error — see logs.")
    st.code(tb)
    print(tb)

# ---------- EDF loader (mne or pyedflib fallback) ----------
def save_uploaded_tmp(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path):
    """
    Returns dict: backend, data (n_chan,n_samples) or raw if mne
    """
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        chs = raw.ch_names
        sf = raw.info.get("sfreq", None)
        return {"backend":"mne", "raw":raw, "data":data, "ch_names":chs, "sfreq":sf}
    elif HAS_PYEDF:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        chs = f.getSignalLabels()
        sf = f.getSampleFrequency(0) if f.getSampleFrequency(0) else None
        sigs = [f.readSignal(i).astype(np.float64) for i in range(n)]
        f._close()
        data = np.vstack(sigs)
        return {"backend":"pyedflib", "data":data, "ch_names":chs, "sfreq":sf}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ---------- Preprocessing ----------
def notch_filter(sig, sfreq, freq=50.0, Q=30.0):
    if sfreq is None or sfreq<=0:
        return sig
    b,a = iirnotch(freq, Q, sfreq)
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def bandpass_filter(sig, sfreq, l=0.5, h=45.0, order=4):
    if sfreq is None or sfreq<=0:
        return sig
    nyq = 0.5*sfreq
    low = max(l/nyq, 1e-6)
    high = min(h/nyq, 0.999)
    b,a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def preprocess_data(edf_dict, notch_freq=50.0, l=0.5, h=45.0):
    data = edf_dict["data"].astype(np.float64)
    sf = edf_dict.get("sfreq") or 256.0
    pre = np.zeros_like(data)
    for i in range(data.shape[0]):
        x = data[i]
        x = notch_filter(x, sf, freq=notch_freq)
        x = bandpass_filter(x, sf, l=l, h=h)
        pre[i] = x
    return pre, sf

# ---------- ICA artifact removal if possible ----------
def run_ica_if_possible(edf_dict, n_components=15):
    if not HAS_MNE or edf_dict.get("backend")!="mne":
        return edf_dict["data"], {"ica_status":"not_available"}
    try:
        raw = edf_dict["raw"]
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, verbose=False)
        ica.fit(raw, picks=picks, verbose=False)
        # conservative: do not auto exclude; return fitted info and data (no change)
        return raw.get_data(), {"ica_status":"fitted", "n_components": ica.n_components_}
    except Exception as e:
        return edf_dict["data"], {"ica_status":f"failed:{str(e)}"}

# ---------- PSD & feature extraction ----------
def compute_psd_band(data, sfreq, picks=None, nperseg=1024):
    bands = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
    nchan = data.shape[0]
    idxs = picks if picks is not None else list(range(nchan))
    rows = []
    for i in idxs:
        sig = data[i]
        try:
            freqs, pxx = welch(sig, fs=sfreq, nperseg=min(nperseg, max(256,len(sig))))
        except Exception:
            freqs = np.array([])
            pxx = np.array([])
        total = float(np.trapz(pxx,freqs)) if freqs.size>0 else 0.0
        row = {"channel_idx": i}
        for k,(lo,hi) in bands.items():
            if freqs.size==0:
                abs_p = 0.0
            else:
                mask = (freqs>=lo)&(freqs<=hi)
                abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum()>0 else 0.0
            rel_p = float(abs_p/total) if total>0 else 0.0
            row[f"{k}_abs"] = abs_p
            row[f"{k}_rel"] = rel_p
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_features(df_bands, ch_names=None):
    agg = {}
    if df_bands.empty:
        return agg
    agg['alpha_rel_mean'] = float(df_bands['alpha_rel'].mean())
    agg['beta_rel_mean'] = float(df_bands['beta_rel'].mean())
    agg['theta_rel_mean'] = float(df_bands['theta_rel'].mean())
    agg['delta_rel_mean'] = float(df_bands['delta_rel'].mean())
    agg['theta_beta_ratio'] = float((df_bands['theta_rel'].mean()) / (df_bands['beta_rel'].mean() + 1e-9))
    # frontal alpha asymmetry (best-effort indices)
    if ch_names and len(ch_names)>=4:
        try:
            names = [n.upper() for n in ch_names]
            def idx_of(pos):
                for i,n in enumerate(names):
                    if pos in n:
                        return i
                return None
            i_f3 = idx_of("F3"); i_f4 = idx_of("F4")
            if i_f3 is not None and i_f4 is not None:
                val_f3 = df_bands.loc[df_bands['channel_idx']==i_f3,'alpha_rel'].values
                val_f4 = df_bands.loc[df_bands['channel_idx']==i_f4,'alpha_rel'].values
                if val_f3.size>0 and val_f4.size>0:
                    agg['alpha_asym_F3_F4'] = float(val_f3[0]-val_f4[0])
        except Exception:
            pass
    return agg

# ---------- Connectivity (if available) ----------
def compute_connectivity_if_available(raw, sfreq, method="pli", fmin=8, fmax=13):
    if not (HAS_CONN and HAS_MNE):
        raise ImportError("mne_connectivity not available")
    try:
        from mne_connectivity import spectral_connectivity
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events=events, tmin=0.0, tmax=1.0-1.0/raw.info['sfreq'], baseline=None, preload=True, verbose=False)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method=method, sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        return {"shape": con.shape, "freqs": list(freqs)}
    except Exception as e:
        raise

# ---------- Microstate (if sklearn available) ----------
def microstate_analysis(data, sfreq, n_states=4):
    if not HAS_SKLEARN:
        return {"status":"sklearn_not_available"}
    try:
        gfp = data.std(axis=0)
        thr = np.percentile(gfp,95)
        peak_idx = np.where(gfp>=thr)[0]
        if peak_idx.size==0:
            return {"status":"no_peaks"}
        maps = data[:, peak_idx].T
        scaler = StandardScaler()
        maps_s = scaler.fit_transform(maps)
        kmeans = KMeans(n_clusters=n_states, random_state=42).fit(maps_s)
        centers = kmeans.cluster_centers_
        return {"status":"ok", "n_peaks":int(peak_idx.size), "centers": centers.tolist()}
    except Exception as e:
        return {"status":f"failed:{str(e)}"}

# ---------- Model & SHAP ----------
def load_model(path):
    try:
        if joblib and os.path.exists(path):
            return joblib.load(path)
    except Exception:
        return None
    return None

def explain_with_shap(model, X_df):
    if not HAS_SHAP:
        return None
    try:
        if HAS_XGBOOST and model.__class__.__name__.lower().startswith("xgb"):
            expl = shap.TreeExplainer(model)
            vals = expl.shap_values(X_df)
            return vals
        else:
            expl = shap.Explainer(model.predict, X_df)
            vals = expl(X_df)
            return vals
    except Exception as e:
        print("SHAP failed:", e)
        return None

# ---------- PDF generation (reportlab + Amiri support) ----------
def register_amiri_font(ttf_path=None):
    """
    Try to register Amiri font from provided path or ./Amiri-Regular.ttf or system fonts.
    """
    try:
        if ttf_path and os.path.exists(ttf_path):
            pdfmetrics.registerFont(TTFont("Amiri", ttf_path))
            return "Amiri"
        # try local
        local = Path("./Amiri-Regular.ttf")
        if local.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(local)))
            return "Amiri"
        # try common system path (Linux)
        sys_paths = ["/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf", "/usr/share/fonts/truetype/Amiri-Regular.ttf"]
        for p in sys_paths:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont("Amiri", p))
                return "Amiri"
    except Exception:
        pass
    # fallback to Helvetica
    return "Helvetica"

def reshape_ar(text):
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf_report(full_summary, lang='en', shap_local=None, amiri_path=None):
    """
    Returns bytes of PDF. If reportlab+arabic tools installed -> RTL Arabic with Amiri.
    """
    if not HAS_REPORTLAB:
        # fallback: return JSON bytes
        return json.dumps(full_summary, indent=2, ensure_ascii=False).encode('utf-8')
    # register Amiri or fallback
    font_name = register_amiri_font(amiri_path)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    x = margin
    y = height - margin
    # header
    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    if lang=='en':
        c.setFont(font_name, 16)
        c.drawCentredString(width/2, y, title_en)
        y -= 24
        met = f"Generated: {now_ts()}"
        c.setFont(font_name, 9)
        c.drawString(x, y, met); y -= 14
        p = full_summary.get("patient",{})
        c.drawString(x, y, f"Patient: {p.get('name','-')}  ID: {p.get('id','-')}  DOB: {p.get('dob','-')}"); y -= 16
    else:
        c.setFont(font_name, 16)
        header = reshape_ar(title_ar)
        c.drawCentredString(width/2, y, header)
        y -= 24
        p = full_summary.get("patient",{})
        lines = [
            reshape_ar(f"التاريخ: {now_ts()}"),
            reshape_ar(f"المريض: {p.get('name','-')}  المعرف: {p.get('id','-')}  الميلاد: {p.get('dob','-')}")
        ]
        c.setFont(font_name, 10)
        for ln in lines:
            c.drawRightString(width-margin, y, ln); y -= 14
    y -= 6
    # EEG summary
    c.setFont(font_name, 12)
    sec_en = "EEG / QEEG Summary"
    sec_ar = reshape_ar("ملخص EEG / QEEG")
    if lang=='en':
        c.drawString(x, y, sec_en); y -= 14
    else:
        c.drawRightString(width-margin, y, sec_ar); y -= 14
    c.setFont(font_name, 9)
    for f in full_summary.get("files",[]):
        txt_en = f"File: {f.get('filename')} | Channels: {f.get('raw_summary',{}).get('n_channels','-')} | sfreq: {f.get('raw_summary',{}).get('sfreq','-')}"
        if lang=='en':
            c.drawString(x+6, y, txt_en); y -= 12
            af = f.get("agg_features",{})
            c.drawString(x+12, y, f"Alpha mean (rel): {af.get('alpha_rel_mean',0):.4f}"); y -= 12
        else:
            ct = reshape_ar(f"الملف: {f.get('filename')}  القنوات: {f.get('raw_summary',{}).get('n_channels','-')}  التردد: {f.get('raw_summary',{}).get('sfreq','-')}")
            c.drawRightString(width-margin, y, ct); y -= 12
            af = f.get("agg_features",{})
            s = reshape_ar(f"متوسط ألفا (نسبي): {af.get('alpha_rel_mean',0):.4f}")
            c.drawRightString(width-margin, y, s); y -= 12
    y -= 6
    # Predictions
    preds = full_summary.get("predictions",{})
    if preds:
        c.setFont(font_name, 11)
        if lang=='en':
            c.drawString(x, y, "Model predictions:"); y -= 12
            for k,v in preds.items():
                c.setFont(font_name, 9)
                c.drawString(x+6, y, f"{k}: {v}"); y -= 10
        else:
            c.drawRightString(width-margin, y, reshape_ar("نتائج النموذج:")); y -= 12
            for k,v in preds.items():
                c.drawRightString(width-margin, y, reshape_ar(f"{k}: {v}")); y -= 10
    y -= 8
    # XAI (shap_local)
    if shap_local:
        c.setFont(font_name, 11)
        if lang=='en':
            c.drawString(x, y, "Explainable AI — top contributors:"); y -= 12
            c.setFont(font_name, 9)
            for feat,val in shap_local.items():
                c.drawString(x+6, y, f"{feat}: {val:.4f}"); y -= 10
        else:
            c.drawRightString(width-margin, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:")); y -= 12
            for feat,val in shap_local.items():
                ln = reshape_ar(f"{feat}: {val:.4f}")
                c.drawRightString(width-margin, y, ln); y -= 10
    # Footer branding
    footer_en = "Designed and developed by Golden Bird LLC — Vista Kaviani | Muscat, Sultanate of Oman"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني | مسقط، سلطنة عمان")
    c.setFont(font_name, 8)
    c.drawCentredString(width/2, 30, footer_en if lang=='en' else footer_ar)
    c.setFont(font_name, 8)
    disc = "Research/demo only — Not a clinical diagnosis."
    c.drawCentredString(width/2, 18, reshape_ar(disc) if lang!='en' and HAS_ARABIC_TOOLS else disc)
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------- UI ----------
# Basic CSS for nicer look
st.markdown("""
<style>
.main > .block-container { max-width: 1200px; }
h1 { color: #1b3b5f; }
.report-footer { color: #6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([4,1])
with col1:
    st.title("🧠 NeuroEarly Pro — Clinical XAI")
    st.markdown("EEG / QEEG / Connectivity / Microstates / Explainable AI — Research demo only")
with col2:
    # small logo placeholder
    st.markdown("<div style='text-align:right; font-size:12px;'>Golden Bird LLC</div>", unsafe_allow_html=True)

# Sidebar: language and patient
with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", options=["en","ar"], index=0)
    st.markdown("---")
    st.header("Patient info")
    patient_name = st.text_input("Name / اسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB")
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))
    st.markdown("---")
    st.write("Model files (optional): place model_depression.pkl and model_alzheimer.pkl in app root.")

# Upload EDF
st.subheader("1) Upload EDF file(s) (.edf)")
uploads = st.file_uploader("EDF files", type=["edf"], accept_multiple_files=True)

# PHQ-9 UI corrected
st.subheader("2) Depression screening — PHQ-9")
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
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe)")

# AD8
st.subheader("3) Cognitive screening — AD8")
AD8_ITEMS = [
 "Problems with judgment (e.g., problems making decisions, bad financial decisions)",
 "Less interest in hobbies/activities",
 "Repeats questions/stories/asks same thing over and over",
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

# options
st.markdown("---")
st.header("Processing options")
use_ica = st.checkbox("Attempt ICA artifact removal (if mne available)", value=False)
compute_conn = st.checkbox("Compute connectivity (if mne_connectivity available)", value=False)
compute_micro = st.checkbox("Microstate analysis (if sklearn available)", value=False)
run_models = st.checkbox("Run classification models (if model files present)", value=True)

# processing
results = []
if uploads:
    for up in uploads:
        st.write(f"File: {up.name}  ({up.size/1024/1024:.2f} MB)")
        try:
            tmp_path = save_uploaded_tmp(up)
            edf = read_edf(tmp_path)
            st.success(f"Loaded — backend: {edf.get('backend')}  channels: {len(edf.get('ch_names',[]))}  sfreq: {edf.get('sfreq')}")
            data_pre, sf = preprocess_data(edf)
            if use_ica:
                data_pre, ica_info = run_ica_if_possible(edf)
                st.write("ICA:", ica_info)
            df_bands = compute_psd_band(data_pre, sf)
            st.dataframe(df_bands.head(20))
            agg = aggregate_features(df_bands, ch_names=edf.get('ch_names'))
            st.write("Aggregated features:", agg)
            conn_summary = None
            if compute_conn and HAS_CONN and HAS_MNE and edf.get("backend")=="mne":
                try:
                    conn_summary = compute_connectivity_if_available(edf["raw"], sf)
                    st.write("Connectivity summary:", conn_summary)
                except Exception as e:
                    st.warning("Connectivity failed: " + str(e))
            micro_summary = None
            if compute_micro and HAS_SKLEARN:
                micro_summary = microstate_analysis(data_pre, sf)
                st.write("Microstate summary:", micro_summary)
            results.append({
                "filename": up.name,
                "raw_summary": {"n_channels": int(data_pre.shape[0]), "sfreq": float(sf)},
                "df_bands": df_bands,
                "agg_features": agg,
                "connectivity": conn_summary,
                "microstate": micro_summary
            })
        except Exception as e:
            st.error(f"Error processing file {up.name}: {e}")
            log_exc(e)

# prepare summary
full_summary = {
    "patient": {"name": patient_name, "id": patient_id, "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": phq},
    "ad8": {"total": ad8_total, "items": ad8},
    "files": results,
    "predictions": {}
}

# load models and run predictions
if run_models and results:
    model_dep = load_model("model_depression.pkl")
    model_ad = load_model("model_alzheimer.pkl")
    Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
    shap_local_for_pdf = None
    if model_dep is not None:
        try:
            predp = model_dep.predict_proba(Xdf)[:,1] if hasattr(model_dep, "predict_proba") else model_dep.predict(Xdf)
            full_summary["predictions"]["depression_probabilities"] = predp.tolist()
            st.write("Depression probabilities:", predp)
            if HAS_SHAP:
                sv = explain_with_shap(model_dep, Xdf)
                if sv is not None:
                    shap_mean = np.abs(sv).mean(axis=0) if isinstance(sv, np.ndarray) else np.abs(sv.values).mean(axis=0)
                    feat_imp = dict(zip(Xdf.columns, shap_mean.tolist()))
                    full_summary.setdefault("xai",{})["depression_global"] = feat_imp
                    shap_local_for_pdf = dict(sorted(feat_imp.items(), key=lambda x:-x[1])[:10])
        except Exception as e:
            st.warning("Depression model prediction failed.")
    if model_ad is not None:
        try:
            predp2 = model_ad.predict_proba(Xdf)[:,1] if hasattr(model_ad, "predict_proba") else model_ad.predict(Xdf)
            full_summary["predictions"]["alzheimers_probabilities"] = predp2.tolist()
            st.write("Alzheimer probabilities:", predp2)
            if HAS_SHAP:
                sv2 = explain_with_shap(model_ad, Xdf)
                if sv2 is not None:
                    shap_mean2 = np.abs(sv2).mean(axis=0) if isinstance(sv2, np.ndarray) else np.abs(sv2.values).mean(axis=0)
                    feat_imp2 = dict(zip(Xdf.columns, shap_mean2.tolist()))
                    full_summary.setdefault("xai",{})["alzheimers_global"] = feat_imp2
                    if shap_local_for_pdf is None:
                        shap_local_for_pdf = dict(sorted(feat_imp2.items(), key=lambda x:-x[1])[:10])
        except Exception as e:
            st.warning("Alzheimer model prediction failed.")

# Report generation UI
st.markdown("---")
st.header("Report generation")
col1, col2 = st.columns([2,1])
with col1:
    st.write("Summary (preview):")
    st.json(full_summary)
with col2:
    st.write("Actions")
    if results:
        lang_choice = st.selectbox("Report language", options=["en","ar"], index=0)
        amiri_path = st.text_input("Amiri TTF path (optional) — leave empty if Amiri-Regular.ttf is in app root", value="")
        if st.button("Generate PDF report"):
            try:
                pdf_bytes = generate_pdf_report(full_summary, lang=lang_choice, shap_local=shap_local_for_pdf, amiri_path=(amiri_path or None))
                st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("PDF generation failed.")
                log_exc(e)
    else:
        st.info("Upload at least one EDF and process to enable report generation.")

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis. Final clinical decisions must be made by a qualified physician.")
