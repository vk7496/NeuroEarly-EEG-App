import io
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

# ------------------ Page Config ------------------
st.set_page_config(page_title="NeuroEarly", layout="centered")

# ------------------ Language Selector ------------------
lang = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])

# ------------------ Texts ------------------
TEXTS = {
    "English": {
        "title": "🧠 NeuroEarly — EEG + Depression (PHQ-9) + AD8",
        "subtitle": "Prototype for early risk screening using EEG frequency bands, mood questionnaires, and cognitive tasks.",
        "upload": "1) Upload EEG (.edf)",
        "phq": "2) Depression Screening — PHQ-9",
        "ad8": "3) Cognitive Screening — AD8",
        "report": "4) Generate Reports",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "note": "⚠️ Research demo only — Not a clinical diagnosis."
    },
    "العربية": {
        "title": "🧠 نيوروإيرلي — تخطيط الدماغ + الاكتئاب (PHQ-9) + AD8",
        "subtitle": "نموذج أولي للفحص المبكر باستخدام ترددات EEG، واستبيانات المزاج، ومهام معرفية.",
        "upload": "١) رفع ملف EEG (.edf)",
        "phq": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) فحص معرفي — AD8",
        "report": "٤) إنشاء التقارير",
        "download_json": "⬇️ تحميل JSON",
        "download_pdf": "⬇️ تحميل PDF",
        "note": "⚠️ هذا نموذج بحث فقط — ليس تشخيصاً طبياً."
    }
}

# ------------------ Band Definitions ------------------
BANDS = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–12 Hz)": (8, 12),
    "Beta (12–30 Hz)": (12, 30),
    "Gamma (30–45 Hz)": (30, 45),
}

# ------------------ Helper Functions ------------------
def filter_raw(raw):
    raw.filter(1., 45., fir_design="firwin", verbose=False)
    raw.notch_filter(50., verbose=False)
    return raw

def compute_band_powers(raw):
    psd = raw.compute_psd(method="welch", fmin=0.5, fmax=45.0, n_fft=2048, n_overlap=1024, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    psd_mean = psds.mean(axis=0)
    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = float(np.trapz(psd_mean[mask], freqs[mask])) if mask.any() else 0.0
    return band_powers

def make_band_chart(band_powers):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.bar(band_powers.keys(), band_powers.values())
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("EEG Band Powers")
    plt.xticks(rotation=25)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def make_signal_chart(raw, seconds=10):
    data, times = raw[:1, :seconds * int(raw.info["sfreq"])]
    fig, ax = plt.subplots(figsize=(7, 2.5), dpi=150)
    ax.plot(times, data[0])
    ax.set_title("EEG Signal Snippet (First Channel)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def build_pdf(results, band_img, sig_img, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(TEXTS[lang]["title"], styles["Title"]))
    flow.append(Paragraph(TEXTS[lang]["subtitle"], styles["Normal"]))
    flow.append(Spacer(1, 12))

    eeg = results["EEG"]
    rows = [["Metric", "Value"],
            ["File", eeg["filename"]],
            ["Sampling Rate (Hz)", eeg["sfreq"]],
            *[(k, f"{v:.3f}") for k, v in eeg["bands"].items()]]

    table = Table(rows, colWidths=[200, 300])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                               ("BOX", (0,0), (-1,-1), 0.5, colors.black),
                               ("GRID", (0,0), (-1,-1), 0.25, colors.black)]))
    flow.append(table)
    flow.append(Spacer(1, 12))

    if band_img: flow.append(RLImage(io.BytesIO(band_img), width=400, height=160))
    if sig_img: flow.append(RLImage(io.BytesIO(sig_img), width=400, height=160))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"PHQ-9 Score: {results['PHQ9']['score']} ({results['PHQ9']['risk']})", styles["Normal"]))
    flow.append(Paragraph(f"AD8 Score: {results['AD8']['score']} ({results['AD8']['risk']})", styles["Normal"]))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(TEXTS[lang]["note"], styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ------------------ Layout ------------------
st.title(TEXTS[lang]["title"])
st.write(TEXTS[lang]["subtitle"])

# Upload EEG
st.header(TEXTS[lang]["upload"])
uploaded = st.file_uploader("Upload EDF", type=["edf"])
band_powers = {}
raw = None
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
        raw = filter_raw(raw)
        band_powers = compute_band_powers(raw)
        st.bar_chart(pd.DataFrame({"Power": band_powers.values()}, index=band_powers.keys()))
        st.image(make_signal_chart(raw), use_column_width=True)

# PHQ-9
st.header(TEXTS[lang]["phq"])
phq_score = 0
for i, q in enumerate([
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble sleeping (too much / too little)",
    "Feeling tired or low energy",
    "Overeating or loss of appetite",
    "Feeling bad about yourself",
    "Trouble concentrating",
    "Being restless or too quiet",
    "Thoughts of self-harm"
], 1):
    ans = st.selectbox(f"{i}. {q}", ["0 = Not at all", "1 = Several days", "2 = > half the days", "3 = Nearly every day"], key=f"phq{i}")
    phq_score += int(ans.split("=")[0].strip())
if phq_score < 5: phq_risk = "Minimal"
elif phq_score < 10: phq_risk = "Mild"
elif phq_score < 15: phq_risk = "Moderate"
elif phq_score < 20: phq_risk = "Moderately severe"
else: phq_risk = "Severe"

# AD8
st.header(TEXTS[lang]["ad8"])
ad8_score = 0
for i, q in enumerate([
    "Problems with judgment",
    "Reduced interest in hobbies",
    "Repeats the same questions",
    "Difficulty learning new tools",
    "Forgets month/year",
    "Trouble with finances",
    "Forgets appointments",
    "Thinking worse than before"
], 1):
    ans = st.selectbox(f"{i}. {q}", ["No", "Yes"], key=f"ad8{i}")
    ad8_score += 1 if ans == "Yes" else 0
ad8_risk = "Possible concern" if ad8_score >= 2 else "Low"

# Reports
st.header(TEXTS[lang]["report"])
if st.button("Generate Report"):
    if raw:
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "EEG": {"filename": uploaded.name, "sfreq": raw.info["sfreq"], "bands": band_powers},
            "PHQ9": {"score": phq_score, "risk": phq_risk},
            "AD8": {"score": ad8_score, "risk": ad8_risk},
        }
        json_bytes = io.BytesIO(json.dumps(results, indent=2).encode("utf-8"))
        st.download_button(TEXTS[lang]["download_json"], data=json_bytes, file_name="report.json", mime="application/json")

        pdf_bytes = build_pdf(results, make_band_chart(band_powers), make_signal_chart(raw), lang)
        st.download_button(TEXTS[lang]["download_pdf"], data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
    else:
        st.warning("Please upload an EEG file first.")

st.caption(TEXTS[lang]["note"])
