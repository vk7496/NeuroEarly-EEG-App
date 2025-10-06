# app_full.py
"""
NeuroEarly Pro - Full robust Streamlit app for EDF (Depression + Alzheimer's screening)
Features:
- Safe EDF upload via temp files
- PSD and band-power feature extraction (handles broadcasting safely)
- Optional connectivity (mne-connectivity) if installed
- ICA try/except (non-fatal); ICA warnings suppressed
- PHQ-9 and AD8 forms
- Patient info, Labs, Medications inputs
- Model hook: if model.pkl present, will run predictions
- SHAP hook: if shap available, will compute explanations
- Export: JSON, CSV, PDF (if fpdf available)
- Caching to reduce recomputation
- Good error handling + user-facing messages in Persian/English
"""
import streamlit as st
import tempfile, os, io, json, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# EEG libs
import mne

# Optional libs - import safely
try:
    import mne_connectivity as mne_conn  # type: ignore
    HAS_MNE_CONN = True
except Exception:
    HAS_MNE_CONN = False

# PDF generation optional
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# SHAP optional
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------------------------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical XAI", layout="wide")

# ---------- Utilities ----------
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def log_and_show_exception(e: Exception):
    tb = traceback.format_exc()
    st.error("یک خطا رخ داد — لطفاً traceback را کپی کن و به من بده.")
    st.code(tb)
    print(tb)

# ---------- Temp EDF loader (safe) ----------
@st.cache_data(show_spinner=False, persist=True)
def save_bytes_to_tempfile(bytes_data: bytes, suffix=".edf"):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(bytes_data)
        tmp.flush()
        return tmp.name

@st.cache_data(show_spinner=False, persist=True)
def read_edf_from_temp(path: str, preload=True):
    # read using mne, handle some io exceptions
    raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    return raw

# ---------- Feature extraction ----------
@st.cache_data(show_spinner=False, persist=True)
def compute_psd_and_bandpowers(raw, picks=None, fmin=0.5, fmax=45.0, n_fft=2048):
    """
    Compute PSD using mne.time_frequency.psd_welch and then compute band powers per channel.
    Return: df_bands (channels x features), freqs, psd_mean (for plotting optionally)
    """
    try:
        if picks is None:
            picks = raw.ch_names[:min(16, len(raw.ch_names))]
        # ensure we work on selection
        picks = list(picks)
        # use raw.copy().pick to avoid modifying original
        raw_sel = raw.copy().pick(picks)
        # compute psd: returns (n_channels, n_freqs)
        psds, freqs = mne.time_frequency.psd_welch(raw_sel, fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
        # psds shape check
        if psds.ndim == 1:
            # shape is (n_freqs,) -> expand
            psds = psds[np.newaxis, :]
        n_chan = psds.shape[0]
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
        rows = []
        for ch_idx, ch_name in enumerate(picks):
            psd = psds[ch_idx, :]
            total_power = np.trapz(psd, freqs)
            row = {"channel": ch_name}
            for b_name, (bmin, bmax) in bands.items():
                mask = (freqs >= bmin) & (freqs <= bmax)
                if mask.sum() == 0 or total_power <= 0:
                    power = 0.0
                    rel = 0.0
                else:
                    power = float(np.trapz(psd[mask], freqs[mask]))
                    rel = float(power / total_power)
                row[f"{b_name}_power"] = power
                row[f"{b_name}_rel"] = rel
            rows.append(row)
        df = pd.DataFrame(rows)
        return df, freqs, psds
    except Exception as e:
        raise

# ---------- Connectivity (optional) ----------
@st.cache_data(show_spinner=False, persist=True)
def compute_connectivity_if_available(raw, method="pli", fmin=8, fmax=13, sfreq=None):
    """
    Tries to compute connectivity using mne_connectivity.spectral_connectivity
    Returns a small summary or raises if not available.
    """
    if not HAS_MNE_CONN:
        raise ImportError("mne_connectivity not installed")
    try:
        from mne_connectivity import spectral_connectivity  # local import for safety
        # create epochs or use raw to compute connectivity — simplest: use epochs of 1-second windows
        # convert to epochs by making fixed-length windows
        data = raw.copy().pick(raw.ch_names[:min(16, len(raw.ch_names))])
        # ensure sfreq
        if sfreq is None:
            sfreq = raw.info.get("sfreq", None)
        # create epochs
        events = mne.make_fixed_length_events(data, duration=1.0)
        epochs = mne.Epochs(data, events=events, tmin=0.0, tmax=1.0 - 1.0/sfreq if sfreq else 0.999, baseline=None, preload=True, verbose=False)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs, method=method, sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False
        )
        # con shape: (n_connections, n_freqs) maybe; trim to summary
        return {"connectivity_shape": con.shape, "freqs": freqs.tolist() if hasattr(freqs, "tolist") else list(freqs)}
    except Exception as e:
        raise

# ---------- ICA helper (non-fatal) ----------
def try_run_ica(raw, n_components=15, random_state=97):
    """
    Try to fit ICA. If fails or no components flagged, return message.
    """
    try:
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(raw, verbose=False)
        # automatic detection of ECG/EOG components is optional; skip here
        # we will not apply or exclude by default; return info
        n_comps = ica.n_components_
        return {"ica_n_components": int(n_comps), "ica_status": "fitted"}
    except Exception as e:
        # don't break pipeline; return info
        return {"ica_n_components": 0, "ica_status": f"ICA failed: {str(e)}"}

# ---------- Model & SHAP loaders ----------
@st.cache_resource
def load_model(path="model.pkl"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            return None
    return None

# ---------- PHQ-9 and AD8 utilities ----------
PHQ9_ITEMS = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling/staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure",
    "7. Trouble concentrating on things, such as reading or watching TV",
    "8. Moving or speaking slowly OR being fidgety/restless",
    "9. Thoughts that you would be better off dead or hurting yourself"
]

AD8_ITEMS = [
    "1. Problems with judgment (e.g., problems making decisions, bad financial decisions)",
    "2. Less interest in hobbies/activities",
    "3. Repeats questions/stories/asks the same thing over and over",
    "4. Trouble learning to use a tool, appliance, or gadget (e.g., VCR, microwave, remote)",
    "5. Forgetting the correct month or year",
    "6. Difficulty handling complicated financial affairs (e.g., balancing checkbook, taxes)",
    "7. Trouble remembering appointments",
    "8. Daily problems with thinking and memory"
]

# ---------- UI layout ----------
st.title("NeuroEarly Pro — Clinical XAI")
st.caption("EEG / QEEG / Connectivity / Microstates / Explainable Risk — Research demo only")

# Sidebar for language and patient short info
with st.sidebar:
    st.header("Language / اللغة")
    lang = st.radio("Choose / اختر", ("en", "ar", "fa (فارسی)"), index=0)
    st.markdown("---")
    st.header("Patient quick info")
    patient_name = st.text_input("Patient name / اسم بیمار")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth (if known)")
    sex = st.selectbox("Sex / جنسیت", ("Male", "Female", "Other", "Unknown"))
    st.markdown("---")
    st.write("App diagnostics:")
    st.write(f"mne_connectivity installed: {HAS_MNE_CONN}")
    st.write(f"fpdf installed: {HAS_FPDF}")
    st.write(f"shap installed: {HAS_SHAP}")
    st.markdown("---")
    st.write("Upload limits: 200MB per file (Streamlit Cloud)")

# Main - sections expandable
st.subheader("Optional: Patient information / معلومات المريض")
with st.expander("Patient details (expand)"):
    clinical_notes = st.text_area("Clinical notes (e.g., symptoms, duration)", height=120)
    occupation = st.text_input("Occupation / شغل")
    contact = st.text_input("Contact (phone/email)")

st.subheader("Optional: Recent lab tests / التحاليل")
with st.expander("Recent labs (expand)"):
    labs_text = st.text_area("Copy-paste recent lab results or notes here", height=120)

st.subheader("Current medications (one per line) / الأدوية الحالية")
meds = st.text_area("One medication per line", height=80)

st.markdown("---")
st.markdown("**1) Upload EEG file(s) (.edf)**")
uploaded = st.file_uploader("EDF files", type=["edf"], accept_multiple_files=True)

# PHQ-9 and AD8
st.markdown("---")
st.subheader("Depression screening — PHQ-9")
phq_scores = {}
for i, item in enumerate(PHQ9_ITEMS, 1):
    phq_scores[f"q{i}"] = st.radio(item, [0,1,2,3], index=0, key=f"phq{i}", horizontal=True)
phq_total = sum(phq_scores.values())
st.info(f"PHQ-9 total score: {phq_total} — (0–4 none, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe)")

st.markdown("---")
st.subheader("Cognitive screening — AD8")
ad8_scores = {}
for i, item in enumerate(AD8_ITEMS, 1):
    ad8_scores[f"a{i}"] = st.radio(item, [0,1], index=0, key=f"ad8_{i}", horizontal=True)
ad8_total = sum(ad8_scores.values())
st.info(f"AD8 total: {ad8_total} — Score ≥2 suggests cognitive impairment (sensitivity)")

# Process files area
if uploaded:
    col_main, col_side = st.columns([3,1])
    results = []
    with col_main:
        st.header("Processing uploaded EDF(s)")
        for up in uploaded:
            st.write(f"File: {up.name} — size: {up.size/1024/1024:.2f} MB")
            try:
                temp_path = save_bytes_to_tempfile(up.getbuffer(), suffix=".edf")
                # read with preload False first to save memory, then optionally preload later
                try:
                    raw = read_edf_from_temp(temp_path, preload=True)
                except MemoryError:
                    raw = read_edf_from_temp(temp_path, preload=False)
                st.success(f"Loaded `{up.name}` — channels: {len(raw.ch_names)}, sfreq: {raw.info.get('sfreq', 'unknown')}")
                st.write("Channels (first 24):", raw.ch_names[:24])

                # Try ICA (non-fatal)
                ica_info = try_run_ica(raw)
                if ica_info.get("ica_status", "").startswith("ICA failed"):
                    st.warning(f"ICA: {ica_info['ica_status']}")
                else:
                    st.info(f"ICA fitted: components={ica_info['ica_n_components']}")

                # Compute band powers safely
                try:
                    picks = raw.ch_names[:min(16, len(raw.ch_names))]
                    df_bands, freqs, psds = compute_psd_and_bandpowers(raw, picks=picks)
                    st.dataframe(df_bands)
                except Exception as e:
                    st.error("Error while computing band powers.")
                    log_and_show_exception(e)
                    df_bands = pd.DataFrame()

                # Optional connectivity
                conn_summary = None
                if st.checkbox("Compute connectivity (requires mne_connectivity)", value=False):
                    if not HAS_MNE_CONN:
                        st.warning("mne_connectivity not installed. Add to requirements.txt: mne-connectivity==0.6.0")
                    else:
                        try:
                            conn_summary = compute_connectivity_if_available(raw, method="pli", fmin=8, fmax=13, sfreq=raw.info.get("sfreq", None))
                            st.write("Connectivity summary:", conn_summary)
                        except Exception as e:
                            st.error("Connectivity calculation failed.")
                            log_and_show_exception(e)

                # Aggregate simple features for model
                agg_features = {}
                if not df_bands.empty:
                    agg_features = {
                        "alpha_rel_mean": float(df_bands["alpha_rel"].mean()),
                        "beta_rel_mean": float(df_bands["beta_rel"].mean()),
                        "theta_rel_mean": float(df_bands["theta_rel"].mean()),
                        "delta_rel_mean": float(df_bands["delta_rel"].mean()),
                        "n_channels": int(len(raw.ch_names)),
                        "sfreq": float(raw.info.get("sfreq", np.nan))
                    }
                    st.json(agg_features)
                else:
                    st.info("No band-power features available for aggregation.")

                # Save results container
                results.append({
                    "filename": up.name,
                    "temp_path": temp_path,
                    "raw_summary": {"n_channels": len(raw.ch_names), "sfreq": raw.info.get("sfreq", None), "ch_names": raw.ch_names[:40]},
                    "df_bands": df_bands,
                    "agg_features": agg_features,
                    "connectivity": conn_summary,
                    "ica": ica_info
                })

            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")
                log_and_show_exception(e)

    with col_side:
        st.header("Actions")
        if results:
            st.success(f"{len(results)} file(s) processed.")
            # Downloads: CSV of bands
            try:
                combined = pd.concat([r["df_bands"].assign(filename=r["filename"]) for r in results], ignore_index=True)
                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                st.download_button("Download combined bands (CSV)", data=csv_bytes, file_name=f"bands_{now_ts()}.csv", mime="text/csv")
            except Exception:
                st.info("No bands CSV available.")

            # JSON of aggregated features + PHQ/AD8 + patient info
            full_summary = {
                "patient": {"name": patient_name, "id": patient_id, "dob": str(dob), "sex": sex, "occupation": occupation, "contact": contact},
                "clinical_notes": clinical_notes,
                "labs": labs_text,
                "medications": meds.splitlines(),
                "phq9": {"total": phq_total, "items": phq_scores},
                "ad8": {"total": ad8_total, "items": ad8_scores},
                "files": []
            }
            for r in results:
                file_entry = {
                    "filename": r["filename"],
                    "raw_summary": r["raw_summary"],
                    "agg_features": r["agg_features"],
                    "ica": r["ica"],
                    "connectivity": r["connectivity"]
                }
                full_summary["files"].append(file_entry)

            json_bytes = json.dumps(full_summary, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button("Download full report (JSON)", data=json_bytes, file_name=f"report_{now_ts()}.json", mime="application/json")

            # Try PDF if available
            if HAS_FPDF and st.button("Generate PDF report"):
                try:
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", size=14)
                    pdf.cell(0, 8, "NeuroEarly Pro — Clinical Report", ln=True, align="C")
                    pdf.ln(6)
                    # patient summary
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 6, f"Patient: {patient_name}  |  ID: {patient_id}", ln=True)
                    pdf.cell(0, 6, f"DOB: {dob}  |  Sex: {sex}", ln=True)
                    pdf.ln(4)
                    pdf.multi_cell(0, 6, f"Clinical notes: {clinical_notes}")
                    pdf.ln(4)
                    pdf.cell(0, 6, f"PHQ-9 total: {phq_total}", ln=True)
                    pdf.cell(0, 6, f"AD8 total: {ad8_total}", ln=True)
                    pdf.ln(6)
                    # files summary
                    for fentry in full_summary["files"]:
                        pdf.set_font("Arial", size=12, style='B')
                        pdf.cell(0, 6, f"File: {fentry['filename']}", ln=True)
                        pdf.set_font("Arial", size=10)
                        pdf.cell(0, 6, f"Channels: {fentry['raw_summary'].get('n_channels')}, sfreq: {fentry['raw_summary'].get('sfreq')}", ln=True)
                        if fentry.get("agg_features"):
                            pdf.cell(0, 6, f"Alpha mean (rel): {fentry['agg_features'].get('alpha_rel_mean'):.4f}", ln=True)
                        pdf.ln(3)
                    # save to bytes
                    pdf_bytes = pdf.output(dest="S").encode("latin-1")
                    st.download_button("Download PDF report", data=pdf_bytes, file_name=f"report_{now_ts()}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error("PDF generation failed.")
                    log_and_show_exception(e)
            else:
                if not HAS_FPDF:
                    st.info("PDF generation requires 'fpdf'. Add to requirements.txt: fpdf==2.5.5")

            # Model prediction hook
            model = load_model("model.pkl")
            if model is not None:
                try:
                    X = pd.DataFrame([r["agg_features"] for r in results])
                    # Some models require feature order; ensure columns exist
                    preds = None
                    if hasattr(model, "predict_proba"):
                        preds = model.predict_proba(X)
                    else:
                        preds = model.predict(X)
                    st.write("Model predictions:")
                    st.write(preds)
                    try:
                        pred_csv = pd.DataFrame(preds).to_csv(index=False).encode("utf-8")
                        st.download_button("Download model predictions (CSV)", data=pred_csv, file_name=f"preds_{now_ts()}.csv", mime="text/csv")
                    except Exception:
                        pass

                    # SHAP explanation if available and model supports
                    if HAS_SHAP and hasattr(model, "predict"):
                        try:
                            explainer = shap.Explainer(model.predict, X)
                            shap_vals = explainer(X)
                            st.header("SHAP summary (approx)")
                            try:
                                st.pyplot(shap.plots.bar(shap_vals, show=False))
                            except Exception:
                                st.write("SHAP computed but plotting failed in this environment.")
                        except Exception as e:
                            st.warning("SHAP explanation failed.")
                            print(str(e))
                except Exception as e:
                    st.warning("Model prediction/explanation failed.")
                    log_and_show_exception(e)
            else:
                st.info("No model.pkl found. To enable predictions place a pickled model named model.pkl in the app root.")

        else:
            st.info("No processing results yet. Upload an EDF file to start.")

st.write("---")
st.caption("Design / Developed by Golden Bird LLC (Vista Kaviani) — Research demo only")
