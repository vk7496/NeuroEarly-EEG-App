import io
import os
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------- Page config ----------
st.set_page_config(page_title="NeuroEarly — EEG + PHQ9 + AD8 (Demo)", layout="centered")
st.title("🧠 NeuroEarly — EEG + Depression (PHQ-9) + AD8 (Demo)")
st.markdown(
    "Prototype for early Alzheimer's risk screening using EEG, questionnaires, and cognitive micro-tasks. "
    "**Research demo only — not a medical diagnosis.**"
)

# ---------- Constants ----------
BANDS = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–12 Hz)": (8, 12),
    "Beta (12–30 Hz)": (12, 30),
    "Gamma (30–45 Hz)": (30, 45),
}

PHQ9_QUESTIONS = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "7. Trouble concentrating on things, such as reading the newspaper or watching television",
    "8. Moving or speaking so slowly that other people could have noticed, or the opposite — being fidgety or restless",
    "9. Thoughts that you would be better off dead or of hurting yourself in some way",
]

PHQ9_OPTIONS = [
    "0 = Not at all",
    "1 = Several days",
    "2 = More than half the days",
    "3 = Nearly every day",
]

AD8_QUESTIONS = [
    "1. Problems with judgment (such as bad financial decisions)",
    "2. Reduced interest in hobbies/activities",
    "3. Repeats the same questions or stories",
    "4. Trouble learning how to use a tool, appliance, or gadget",
    "5. Forgets the correct month or year",
    "6. Difficulty handling complicated financial affairs (e.g., paying bills)",
    "7. Trouble remembering appointments",
    "8. Everyday thinking is getting worse",
]

AD8_OPTIONS = ["No", "Yes"]  # No=0, Yes=1


# ---------- Helper functions ----------
def compute_band_powers_from_raw(raw: mne.io.BaseRaw, fmin=0.5, fmax=45.0):
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    psd_mean = psds.mean(axis=0) if psds.ndim == 2 else psds

    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = float(np.trapz(psd_mean[mask], freqs[mask])) if mask.any() else 0.0
    return band_powers, freqs


def make_band_chart_png(band_powers: dict) -> bytes:
    labels = list(band_powers.keys())
    values = [band_powers[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.bar(labels, values)
    ax.set_ylabel("Integrated power (a.u.)")
    ax.set_title("EEG Band Powers (Welch PSD)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_signal_snippet_png(raw: mne.io.BaseRaw, seconds=10) -> bytes:
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks) if len(picks) > 0 else raw.get_data()
    ch0 = data[0] if data.ndim == 2 else data
    sf = int(raw.info["sfreq"])
    n = min(len(ch0), seconds * sf)
    t = np.arange(n) / sf
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=150)
    ax.plot(t, ch0[:n])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"EEG first channel snippet (~{n/sf:.1f}s)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def eeg_heuristics(band_powers: dict):
    alpha = band_powers.get("Alpha (8–12 Hz)", 1e-9)
    theta = band_powers.get("Theta (4–8 Hz)", 0.0)
    beta = band_powers.get("Beta (12–30 Hz)", 0.0)
    theta_alpha = float(theta / alpha) if alpha > 0 else 0.0
    beta_alpha = float(beta / alpha) if alpha > 0 else 0.0
    return {
        "Theta/Alpha": round(theta_alpha, 3),
        "Beta/Alpha": round(beta_alpha, 3),
        "Depression (EEG heuristic)": "Higher" if theta_alpha > 0.8 else "Lower",
        "Alzheimer (EEG heuristic)": "Possible concern" if beta_alpha < 0.5 else "Normal",
    }


def build_pdf_bytes(results: dict, band_png: bytes | None, sig_png: bytes | None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("NeuroEarly — EEG + PHQ-9 + AD8 Report", styles["Title"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"Timestamp: {results.get('timestamp', '')}", styles["Normal"]))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph("<b>EEG Summary</b>", styles["Heading3"]))
    eeg = results.get("EEG", {})
    rows = [["Metric", "Value"]]
    rows.append(["File", eeg.get("filename", "—")])
    rows.append(["Filesize (MB)", str(eeg.get("filesize_mb", "—"))])
    rows.append(["Sampling rate (Hz)", str(eeg.get("sfreq_hz", "—"))])
    for k, v in eeg.get("bands", {}).items():
        rows.append([k, f"{v:.6g}"])
    ratios = eeg.get("ratios", {})
    rows.append(["Theta/Alpha", str(ratios.get("Theta/Alpha", "—"))])
    rows.append(["Beta/Alpha", str(ratios.get("Beta/Alpha", "—"))])
    table = Table(rows, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 8))

    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=420, height=200))
        flow.append(Spacer(1, 6))
    if sig_png:
        flow.append(RLImage(io.BytesIO(sig_png), width=420, height=180))
        flow.append(Spacer(1, 8))

    flow.append(Paragraph("<b>PHQ-9 (Depression)</b>", styles["Heading3"]))
    flow.append(Paragraph(f"Score: {results['Depression']['score']} / 27", styles["Normal"]))
    flow.append(Paragraph(f"Risk: {results['Depression']['risk']}", styles["Normal"]))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("<b>AD8 (Cognition)</b>", styles["Heading3"]))
    flow.append(Paragraph(f"Score: {results['Alzheimer']['score']} / 8", styles["Normal"]))
    flow.append(Paragraph(f"Risk: {results['Alzheimer']['risk']}", styles["Normal"]))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph("Note: This report is a research demo and not a clinical diagnosis.", styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()


# ---------- 1) EEG Upload ----------
st.header("1) Upload EEG (.edf)")
uploaded = st.file_uploader("Upload an EDF file", type=["edf"])
raw = None
band_powers = {}
ratios = {}
band_png = sig_png = None
eeg_meta = {"filename": None, "filesize_mb": None, "sfreq_hz": None}

if uploaded is not None:
    try:
        eeg_meta["filename"] = uploaded.name
        eeg_meta["filesize_mb"] = round(len(uploaded.getbuffer()) / (1024 * 1024), 3)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        eeg_meta["sfreq_hz"] = float(raw.info.get("sfreq", np.nan))
        band_powers, freqs = compute_band_powers_from_raw(raw)
        ratios = eeg_heuristics(band_powers)

        st.subheader("EEG Band Powers")
        df_bands = pd.DataFrame({"Band": list(band_powers.keys()), "Power": list(band_powers.values())})
        st.bar_chart(df_bands.set_index("Band"))
        st.caption(f"Theta/Alpha = {ratios['Theta/Alpha']} | Beta/Alpha = {ratios['Beta/Alpha']}")

        st.subheader("EEG Signal Snippet (first channel)")
        sig_png = make_signal_snippet_png(raw, seconds=10)
        st.image(sig_png, use_column_width=True)

        band_png = make_band_chart_png(band_powers)

    except Exception as e:
        st.error(f"Error reading EEG file: {e}")

# ---------- 2) PHQ-9 ----------
st.header("2) Depression Screening — PHQ-9")
phq_answers = []
for i, q in enumerate(PHQ9_QUESTIONS, 1):
    ans = st.selectbox(q, PHQ9_OPTIONS, index=0, key=f"phq9_{i}")
    phq_answers.append(int(ans.split("=")[0].strip()))
phq_score = sum(phq_answers)
if phq_score < 5:
    phq_risk = "Minimal"
elif phq_score < 10:
    phq_risk = "Mild"
elif phq_score < 15:
    phq_risk = "Moderate"
elif phq_score < 20:
    phq_risk = "Moderately severe"
else:
    phq_risk = "Severe"
st.write(f"PHQ-9 Score: **{phq_score} / 27** → **{phq_risk}**")

# ---------- 3) AD8 ----------
st.header("3) Cognitive Screening — AD8 (informant-based demo)")
ad8_answers = []
for i, q in enumerate(AD8_QUESTIONS, 1):
    ans = st.selectbox(q, AD8_OPTIONS, index=0, key=f"ad8_{i}")
    ad8_answers.append(1 if ans == "Yes" else 0)
ad8_score = sum(ad8_answers)
ad8_risk = "Possible concern (≥2)" if ad8_score >= 2 else "Low"
st.write(f"AD8 Score: **{ad8_score} / 8** → **{ad8_risk}**")

# ---------- 4) Create Reports ----------
st.header("4) Generate Reports (PDF & JSON)")
if st.button("Create Reports"):
    if raw is None:
        st.warning("Please upload a valid EDF file before generating reports.")
    else:
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "EEG": {
                "filename": eeg_meta["filename"],
                "filesize_mb": eeg_meta["filesize_mb"],
                "sfreq_hz": eeg_meta["sfreq_hz"],
                "bands": band_powers,
                "ratios": ratios,
            },
            "Depression": {"score": phq_score, "risk": phq_risk, "answers": phq_answers},
            "Alzheimer": {"score": ad8_score, "risk": ad8_risk, "answers": ad8_answers},
            "note": "Demo synthesis of EEG heuristics + PHQ-9 + AD8. Not a diagnosis.",
        }

        json_bytes = io.BytesIO(json.dumps(results, indent=2).encode("utf-8"))
        st.download_button("⬇️ Download JSON", data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

        pdf_bytes = build_pdf_bytes(results, band_png, sig_png)
        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="neuroearly_report.pdf", mime="application/pdf")

st.caption("© NeuroEarly demo — research only. Not a clinical diagnostic tool.")
