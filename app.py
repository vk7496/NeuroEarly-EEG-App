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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ---------------------------
# Fonts (for Arabic PDF)
# ---------------------------
AMIRI_PATH = "Amiri-Regular.ttf"
if os.path.exists(AMIRI_PATH):
    if "Amiri" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))

# ---------------------------
# Text dictionary
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro",
        "subtitle": "Prototype for early Alzheimer’s & Depression risk screening using EEG, questionnaires and cognitive micro-tasks.",
        "upload": "1) Upload EEG file (.edf)",
        "clean": "Apply ICA artifact removal (experimental, slow)",
        "phq9": "2) Depression Screening — PHQ-9",
        "ad8": "3) Cognitive Screening — AD8",
        "report": "4) Generate Report",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "note": "⚠️ Research demo only — Not a clinical diagnostic tool.",
        "phq9_questions": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating (e.g., reading, watching TV)",
            "Moving or speaking slowly, OR being fidgety/restless",
            "Thoughts of being better off dead or self-harm"
        ],
        "phq9_options": [
            "0 = Not at all",
            "1 = Several days",
            "2 = More than half the days",
            "3 = Nearly every day"
        ],
        "ad8_questions": [
            "Problems with judgment (e.g., poor financial decisions)",
            "Reduced interest in hobbies/activities",
            "Repeats questions or stories",
            "Trouble using a tool or gadget",
            "Forgets the correct month or year",
            "Difficulty managing finances (e.g., paying bills)",
            "Trouble remembering appointments",
            "Everyday thinking is getting worse"
        ],
        "ad8_options": ["No", "Yes"]
    },
    "ar": {
        "title": "🧠 نيوروإيرلي برو",
        "subtitle": "نموذج أولي للفحص المبكر لمخاطر الزهايمر والاكتئاب باستخدام EEG والاستبيانات والاختبارات المعرفية الصغيرة.",
        "upload": "١) تحميل ملف تخطيط الدماغ (.edf)",
        "clean": "إزالة التشويش باستخدام ICA (تجريبي، بطيء)",
        "phq9": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "note": "⚠️ هذا نموذج بحثي فقط — ليس أداة تشخيص سريري.",
        "phq9_questions": [
            "قلة الاهتمام أو المتعة في الأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم أو النوم المفرط",
            "الشعور بالتعب أو قلة الطاقة",
            "فقدان الشهية أو الإفراط في الأكل",
            "الشعور بأنك شخص سيء أو فاشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الحركة أو الكلام ببطء شديد، أو فرط الحركة",
            "أفكار بأنك أفضل حالاً ميتاً أو أفكار لإيذاء النفس"
        ],
        "phq9_options": [
            "0 = أبداً",
            "1 = عدة أيام",
            "2 = أكثر من نصف الأيام",
            "3 = كل يوم تقريباً"
        ],
        "ad8_questions": [
            "مشاكل في الحكم أو اتخاذ القرارات",
            "انخفاض الاهتمام بالهوايات أو الأنشطة",
            "تكرار الأسئلة أو القصص",
            "صعوبة في استخدام أداة أو جهاز",
            "نسيان الشهر أو السنة الصحيحة",
            "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
            "صعوبة في تذكر المواعيد",
            "تدهور التفكير اليومي"
        ],
        "ad8_options": ["لا", "نعم"]
    }
}

# ---------------------------
# EEG Helper
# ---------------------------
def clean_eeg(raw):
    raw.filter(l_freq=1.0, h_freq=40.0)
    raw.notch_filter(freqs=[50, 60])
    return raw

def compute_band_powers(raw: mne.io.BaseRaw):
    psd = raw.compute_psd(fmin=0.5, fmax=45, method="welch")
    psds, freqs = psd.get_data(return_freqs=True)
    mean_psd = psds.mean(axis=0)
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
    powers = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        powers[name] = float(np.trapz(mean_psd[mask], freqs[mask]))
    return powers

def plot_bands(powers: dict):
    fig, ax = plt.subplots()
    ax.bar(powers.keys(), powers.values())
    ax.set_title("EEG Band Powers")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# PDF Generator
# ---------------------------
def build_pdf_bytes(results: dict, lang="en", band_png=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()

    if lang == "ar" and os.path.exists(AMIRI_PATH):
        for s in ["Normal", "Title", "Heading2", "Italic"]:
            styles[s].fontName = "Amiri"

    flow = []
    t = TEXTS[lang]

    flow.append(Paragraph(t["title"], styles["Title"]))
    flow.append(Paragraph(t["subtitle"], styles["Normal"]))
    flow.append(Spacer(1, 12))

    eeg = results["EEG"]
    flow.append(Paragraph("EEG Band Powers:", styles["Heading2"]))
    rows = [["Band", "Power"]]
    for k, v in eeg["bands"].items():
        rows.append([k, f"{v:.4f}"])
    table = Table(rows, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black)
    ]))
    flow.append(table)
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(f"PHQ-9 Score: {results['Depression']['score']} → {results['Depression']['risk']}", styles["Normal"]))
    flow.append(Paragraph(f"AD8 Score: {results['Alzheimer']['score']} → {results['Alzheimer']['risk']}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=400, height=200))
        flow.append(Spacer(1, 12))

    # Recommendation (simple demo logic)
    rec = "Follow up with a neurologist and psychiatrist for combined evaluation."
    if lang == "ar":
        rec = "ينصح بمتابعة مع طبيب الأعصاب وطبيب نفسي لتقييم مشترك."
    flow.append(Paragraph("Recommendation:", styles["Heading2"]))
    flow.append(Paragraph(rec, styles["Normal"]))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(t["note"], styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Streamlit App
# ---------------------------
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"])
t = TEXTS[lang]

st.title(t["title"])
st.write(t["subtitle"])

# 1) EEG
st.header(t["upload"])
uploaded = st.file_uploader("EDF", type=["edf"])
apply_ica = st.checkbox(t["clean"])
bands = {}
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
        raw = clean_eeg(raw)
        if apply_ica:
            ica = mne.preprocessing.ICA(n_components=10, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)
        bands = compute_band_powers(raw)
        st.image(plot_bands(bands))

# 2) PHQ-9
st.header(t["phq9"])
phq_answers = []
for i, q in enumerate(t["phq9_questions"], 1):
    ans = st.selectbox(q, t["phq9_options"], key=f"phq{i}")
    phq_answers.append(int(ans.split("=")[0].strip()) if "=" in ans else t["phq9_options"].index(ans))
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

# 3) AD8
st.header(t["ad8"])
ad8_answers = []
for i, q in enumerate(t["ad8_questions"], 1):
    ans = st.selectbox(q, t["ad8_options"], key=f"ad8{i}")
    ad8_answers.append(1 if ans == t["ad8_options"][1] else 0)
ad8_score = sum(ad8_answers)
ad8_risk = "Low" if ad8_score < 2 else "Possible concern"

# 4) Report
st.header(t["report"])
if st.button("Generate"):
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "EEG": {"bands": bands},
        "Depression": {"score": phq_score, "risk": phq_risk},
        "Alzheimer": {"score": ad8_score, "risk": ad8_risk}
    }
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode())
    st.download_button(t["download_json"], json_bytes, file_name="report.json")

    pdf_bytes = build_pdf_bytes(results, lang=lang, band_png=plot_bands(bands))
    st.download_button(t["download_pdf"], pdf_bytes, file_name="report.pdf")
