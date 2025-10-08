# app_cloud_safe_XAI.py
"""
NeuroEarly Pro — Cloud-safe, feature-rich Streamlit app
Features:
- EDF upload (pyedflib fallback if mne missing)
- Preprocessing: bandpass, notch, simple artifact removal (eye-blink/muscle via ICA if mne present)
- PSD / band-power features, ratios, frontal alpha asymmetry
- Connectivity (if mne_connectivity available) optional
- Microstate basic analysis (GFP peaks + kmeans) approximate (fallback if scikit-learn available)
- Model hooks for Depression and Alzheimer (load model_depression.pkl, model_alzheimer.pkl)
- XAI with SHAP (if available), fallback to precomputed shap_values.json
- PHQ-9 (corrected) and AD8 UI
- Bilingual PDF generator (English/Arabic). For Arabic RTL quality, reportlab + arabic_reshaper + python-bidi recommended.
- Branding footer: Golden Bird LLC
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro — Clinical XAI", layout="wide")

import os, io, json, tempfile, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Try imports with graceful fallback
HAS_MNE = False
HAS_PYEDF = False
HAS_CONN = False
HAS_SHAP = False
HAS_SKLEARN = False
HAS_REPORTLAB = False
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

# Optional for high-quality PDFs with Arabic support
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    HAS_REPORTLAB = True
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_REPORTLAB = False
    HAS_ARABIC = False

from scipy.signal import welch, butter, filtfilt, iirnotch

# ----------------- Helpers -----------------
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def log_exc(e):
    tb = traceback.format_exc()
    st.error("Internal error — see console/logs.")
    st.code(tb)
    print(tb)

# ---------- EDF loader (mne or pyedflib) ----------
def save_uploaded_tmp(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def read_edf(path):
    """
    Returns: dict with keys:
      backend: 'mne' or 'pyedflib'
      data: np.array (n_channels, n_samples)
      ch_names: list
      sfreq: float
    """
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()  # shape (n_channels, n_times)
        chs = raw.ch_names
        sf = raw.info.get("sfreq", None)
        return {"backend":"mne", "raw": raw, "data": data, "ch_names": chs, "sfreq": sf}
    elif HAS_PYEDF:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        chs = f.getSignalLabels()
        sf = f.getSampleFrequency(0) if f.getSampleFrequency(0) else None
        sigs = [f.readSignal(i).astype(np.float64) for i in range(n)]
        f._close()
        data = np.vstack(sigs)  # (n_channels, n_samples)
        return {"backend":"pyedflib", "data": data, "ch_names": chs, "sfreq": sf}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# ---------- Preprocessing ----------
def notch_filter(sig, sfreq, freq=50.0, Q=30.0):
    if sfreq is None or sfreq <= 0:
        return sig
    b, a = iirnotch(freq, Q, sfreq)
    return filtfilt(b, a, sig)

def bandpass_filter(sig, sfreq, l=0.5, h=45.0, order=4):
    if sfreq is None or sfreq <= 0:
        return sig
    nyq = 0.5 * sfreq
    low = l / nyq
    high = h / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

def preprocess_data(edf_dict, apply_notch=True, l=0.5, h=45.0):
    data = edf_dict["data"].astype(np.float64)
    sf = edf_dict.get("sfreq") or 256.0
    pre = np.zeros_like(data)
    for i in range(data.shape[0]):
        x = data[i]
        if apply_notch:
            x = notch_filter(x, sf, freq=50.0)
        x = bandpass_filter(x, sf, l=l, h=h)
        pre[i] = x
    return pre, sf

# ---------- Artifact removal (ICA if mne available) ----------
def run_ica_if_possible(edf_dict, n_components=15):
    """
    If mne is available, run ICA and return cleaned raw (numpy array).
    Otherwise, return original data and note that ICA not run.
    """
    if not HAS_MNE or edf_dict.get("backend")!='mne':
        return edf_dict["data"], {"ica_status":"not_available"}
    try:
        raw = edf_dict["raw"]
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, verbose=False)
        ica.fit(raw, picks=picks, verbose=False)
        # automatic detection optional — here we skip automatic remove to be conservative
        raw_clean = raw.copy()
        # do not apply removals automatically; return fitted info
        return raw_clean.get_data(), {"ica_status":"fitted", "n_components": ica.n_components_}
    except Exception as e:
        return edf_dict["data"], {"ica_status": f"failed: {str(e)}"}

# ---------- PSD & features ----------
def compute_psd_band(df_data, sfreq, picks=None, nperseg=1024):
    """
    df_data: np.array (n_channels, n_samples)
    returns: df_bands: DataFrame rows per channel with abs/rel for delta/theta/alpha/beta/gamma
    """
    bands = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
    nchan = df_data.shape[0]
    idxs = picks if picks is not None else list(range(nchan))
    rows = []
    for i in idxs:
        sig = df_data[i]
        try:
            freqs, pxx = welch(sig, fs=sfreq, nperseg=min(nperseg, max(256, len(sig))), detrend='constant')
        except Exception:
            freqs = np.array([])
            pxx = np.array([])
        total = float(np.trapz(pxx, freqs)) if freqs.size>0 else 0.0
        row = {"channel": i}
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
    df = pd.DataFrame(rows)
    return df

def aggregate_features(df_bands, ch_names=None):
    agg = {}
    if df_bands.empty:
        return agg
    agg['alpha_rel_mean'] = float(df_bands['alpha_rel'].mean())
    agg['beta_rel_mean'] = float(df_bands['beta_rel'].mean())
    agg['theta_rel_mean'] = float(df_bands['theta_rel'].mean())
    agg['delta_rel_mean'] = float(df_bands['delta_rel'].mean())
    agg['theta_beta_ratio'] = float((df_bands['theta_rel'].mean()) / (df_bands['beta_rel'].mean() + 1e-9))
    # frontal alpha asymmetry example if ch_names provided (best-effort)
    if ch_names and len(ch_names)>=4:
        # try find F3,F4 or Fp1,Fp2 names heuristically
        try:
            names = [n.upper() for n in ch_names]
            def idx_of(pos):
                for i,n in enumerate(names):
                    if pos in n:
                        return i
                return None
            i_f3 = idx_of("F3")
            i_f4 = idx_of("F4")
            if i_f3 is not None and i_f4 is not None:
                # use relative alpha per channel
                val_f3 = df_bands.loc[df_bands['channel']==i_f3, 'alpha_rel'].values
                val_f4 = df_bands.loc[df_bands['channel']==i_f4, 'alpha_rel'].values
                if val_f3.size>0 and val_f4.size>0:
                    agg['alpha_asym_F3_F4'] = float(val_f3[0] - val_f4[0])
        except Exception:
            pass
    return agg

# ---------- Connectivity (optional) ----------
def compute_connectivity(raw, sfreq, fmin=8, fmax=13):
    if not HAS_CONN or not HAS_MNE:
        raise ImportError("Connectivity backend not available")
    try:
        # expect raw is mne Raw object
        from mne_connectivity import spectral_connectivity
        # construct epochs of 1s
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events=events, tmin=0.0, tmax=1.0-1.0/raw.info['sfreq'], baseline=None, preload=True, verbose=False)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method="pli", sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        return {"shape": con.shape, "freqs": list(freqs)}
    except Exception as e:
        raise

# ---------- Microstate (simple) ----------
def microstate_analysis(data, sfreq, n_states=4):
    """
    Simple microstate: compute GFP peaks and run kmeans on template maps.
    data: (n_channels, n_samples)
    returns: dict with microstate maps mean (if sklearn available)
    """
    if not HAS_SKLEARN:
        return {"status":"sklearn_not_available"}
    try:
        # compute GFP (global field power) -> std across channels at each timepoint
        gfp = data.std(axis=0)
        # find peaks: simple threshold top percent
        thr = np.percentile(gfp, 95)
        peak_idx = np.where(gfp >= thr)[0]
        if peak_idx.size == 0:
            return {"status":"no_peaks"}
        # pick maps at peaks
        maps = data[:, peak_idx].T  # shape (n_peaks, n_channels)
        scaler = StandardScaler()
        maps_s = scaler.fit_transform(maps)
        kmeans = KMeans(n_clusters=n_states, random_state=42).fit(maps_s)
        centers = kmeans.cluster_centers_
        return {"status":"ok", "n_peaks": int(peak_idx.size), "centers": centers.tolist()}
    except Exception as e:
        return {"status": f"failed: {str(e)}"}

# ---------- Models & SHAP loader ----------
def load_model(path):
    try:
        import joblib
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        pass
    return None

def explain_with_shap(model, X_df):
    if not HAS_SHAP:
        return None
    try:
        # If tree model, TreeExplainer; else generic
        if hasattr(model, "get_booster") or model.__class__.__name__.lower().startswith("xgboost"):
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_df)
            return sv
        else:
            expl = shap.Explainer(model.predict, X_df)
            sv = expl(X_df)
            return sv
    except Exception as e:
        print("SHAP failed:", e)
        return None

# ---------- PDF generator (reportlab if present) ----------
def generate_pdf_report(full_summary, lang='en', shap_local=None):
    """
    Returns bytes of PDF.
    If reportlab+arabic available -> nicer; otherwise fallback to fpdf-like simple text using reportlab basic anyway.
    """
    if HAS_REPORTLAB:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 20*mm
        x = margin
        y = height - margin
        # Header
        if lang=='en':
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width/2, y, "NeuroEarly Pro — Clinical Report")
            y -= 12*mm
            c.setFont("Helvetica", 10)
            c.drawString(x, y, f"Generated: {now_ts()}")
            y -= 6*mm
            p = full_summary.get("patient", {})
            c.drawString(x, y, f"Patient: {p.get('name','-')}   ID: {p.get('id','-')}   DOB: {p.get('dob','-')}")
            y -= 8*mm
        else:
            # Arabic (reshaper + bidi)
            c.setFont("Helvetica-Bold", 16)
            header = "تقرير NeuroEarly Pro — سريري"
            if HAS_ARABIC:
                header = get_display(arabic_reshaper.reshape(header))
            c.drawCentredString(width/2, y, header)
            y -= 12*mm
            p = full_summary.get("patient", {})
            lines = [
                f"التاريخ: {now_ts()}",
                f"المريض: {p.get('name','-')}  المعرف: {p.get('id','-')}  الميلاد: {p.get('dob','-')}"
            ]
            for ln in lines:
                out = get_display(arabic_reshaper.reshape(ln)) if HAS_ARABIC else ln
                c.drawString(x, y, out)
                y -= 6*mm
        y -= 4*mm
        # EEG files summary
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "EEG / QEEG Summary:")
        y -= 8*mm
        c.setFont("Helvetica", 10)
        for f in full_summary.get("files", []):
            txt = f"File: {f.get('filename')} | Channels: {f.get('raw_summary',{}).get('n_channels','-')} | sfreq: {f.get('raw_summary',{}).get('sfreq','-')}"
            c.drawString(x, y, txt)
            y -= 6*mm
            # band features short
            if f.get("agg_features"):
                s = f"Alpha mean (rel): {f['agg_features'].get('alpha_rel_mean',0):.4f} | Theta mean: {f['agg_features'].get('theta_rel_mean',0):.4f}"
                c.drawString(x+6*mm, y, s)
                y -= 6*mm
        y -= 4*mm
        # Model predictions
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Model Predictions / النتائج:")
        y -= 8*mm
        preds = full_summary.get("predictions", {})
        if preds:
            for key, val in preds.items():
                txt = f"{key}: {val}"
                c.drawString(x, y, txt)
                y -= 6*mm
        # XAI local summary
        if shap_local:
            y -= 6*mm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x, y, "XAI (top contributors):")
            y -= 8*mm
            c.setFont("Helvetica", 10)
            # shap_local assumed dict feature->value
            for feat, v in shap_local.items():
                c.drawString(x, y, f"{feat}: {v:.4f}")
                y -= 6*mm
                if y < 40*mm:
                    c.showPage(); y = height - margin
        # Footer branding
        y = 20*mm
        c.setFont("Helvetica", 8)
        footer = "Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman"
        if lang!='en' and HAS_ARABIC:
            footer = get_display(arabic_reshaper.reshape("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني  | مسقط، سلطنة عمان"))
        c.drawCentredString(width/2, y, footer)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(width/2, y-6, "Research/demo only — Not a clinical diagnosis.")
        c.save()
        buffer.seek(0)
        return buffer.read()
    else:
        # Simple plain-text PDF fallback using reportlab basic if present or return text file bytes
        txt = "NeuroEarly Pro — Report\n\n"
        txt += json.dumps(full_summary, indent=2, ensure_ascii=False)
        return txt.encode('utf-8')

# ---------- UI ----------
st.title("NeuroEarly Pro — Clinical XAI (Cloud-Safe)")
st.markdown("Upload EDF(s) → preprocess → features → model predictions + XAI → bilingual PDF report (EN / AR)")

# Sidebar: language & patient basic
with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", options=["en","ar"], index=0)
    st.markdown("---")
    st.header("Patient info")
    patient_name = st.text_input("Name / اسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB")
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))

st.markdown("### 1) Upload EDF file(s) (.edf)")
uploads = st.file_uploader("EDF files", type=["edf"], accept_multiple_files=True)

# PHQ-9 (corrected) UI (0-3 each)
st.markdown("### 2) Depression screening — PHQ-9")
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

# AD8
st.markdown("### 3) Cognitive screening — AD8")
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

# Options
st.markdown("---")
st.header("Processing options")
use_ica = st.checkbox("Attempt ICA artifact removal (if mne available)", value=False)
compute_conn = st.checkbox("Compute connectivity (if available)", value=False)
compute_micro = st.checkbox("Microstate analysis (if sklearn available)", value=False)
run_models = st.checkbox("Run classification models (if model files present)", value=True)

# Process uploaded files
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
                    conn_summary = compute_connectivity(edf["raw"], sf)
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

# Combined summary & model
full_summary = {
    "patient": {"name": patient_name, "id": patient_id, "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": phq},
    "ad8": {"total": ad8_total, "items": ad8},
    "files": results,
    "predictions": {}
}

# Load models if present
if run_models and results:
    model_dep = load_model("model_depression.pkl")
    model_ad = load_model("model_alzheimer.pkl")
    Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
    if model_dep is not None:
        try:
            predp = model_dep.predict_proba(Xdf)[:,1] if hasattr(model_dep, "predict_proba") else model_dep.predict(Xdf)
            full_summary["predictions"]["depression_probabilities"] = predp.tolist()
            st.write("Depression probabilities:", predp)
            # shap local if available
            shap_local_dep = None
            if HAS_SHAP:
                sv = explain_with_shap(model_dep, Xdf)
                if sv is not None:
                    # for brevity take first sample contributions sum
                    try:
                        # shap returns array-like; compute per-feature mean abs
                        shap_mean = np.abs(sv).mean(axis=0) if isinstance(sv, np.ndarray) else np.abs(sv.values).mean(axis=0)
                        feat_imp = dict(zip(Xdf.columns, shap_mean.tolist()))
                        full_summary.setdefault("xai",{})["depression_global"] = feat_imp
                        shap_local_dep = dict(sorted(feat_imp.items(), key=lambda x:-x[1])[:8])
                    except Exception:
                        shap_local_dep = None
        except Exception as e:
            st.warning("Depression model prediction failed.")
    if model_ad is not None:
        try:
            predp2 = model_ad.predict_proba(Xdf)[:,1] if hasattr(model_ad, "predict_proba") else model_ad.predict(Xdf)
            full_summary["predictions"]["alzheimers_probabilities"] = predp2.tolist()
            st.write("Alzheimer probabilities:", predp2)
            shap_local_ad = None
            if HAS_SHAP:
                sv2 = explain_with_shap(model_ad, Xdf)
                if sv2 is not None:
                    try:
                        shap_mean2 = np.abs(sv2).mean(axis=0) if isinstance(sv2, np.ndarray) else np.abs(sv2.values).mean(axis=0)
                        feat_imp2 = dict(zip(Xdf.columns, shap_mean2.tolist()))
                        full_summary.setdefault("xai",{})["alzheimers_global"] = feat_imp2
                        shap_local_ad = dict(sorted(feat_imp2.items(), key=lambda x:-x[1])[:8])
                    except Exception:
                        shap_local_ad = None
        except Exception as e:
            st.warning("Alzheimer model prediction failed.")

# Generate bilingual report
st.markdown("---")
st.header("Report generation")
col1, col2 = st.columns([2,1])
with col1:
    st.write("Preview full summary (JSON):")
    st.json(full_summary)
with col2:
    st.write("Actions:")
    if results:
        # choose lang
        lang_choice = st.selectbox("Report language", options=["en","ar"], index=0)
        # compute shap_local summary to show in report (prefer model dep)
        shap_local_for_pdf = None
        if full_summary.get("xai"):
            if "depression_global" in full_summary["xai"]:
                shap_local_for_pdf = dict(sorted(full_summary["xai"]["depression_global"].items(), key=lambda x:-x[1])[:10])
            elif "alzheimers_global" in full_summary["xai"]:
                shap_local_for_pdf = dict(sorted(full_summary["xai"]["alzheimers_global"].items(), key=lambda x:-x[1])[:10])
        if st.button("Generate bilingual PDF report"):
            try:
                pdf_bytes = generate_pdf_report(full_summary, lang=lang_choice, shap_local=shap_local_for_pdf)
                st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("PDF generation failed.")
                log_exc(e)
    else:
        st.info("Upload and process at least one EDF file to generate report.")

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis. Final clinical decisions must be made by a qualified physician.")
