import io
import json
import tempfile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import mne
import requests
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ---------------------------
# Language dictionary
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro — EEG + PHQ-9 + AD8",
        "subtitle": "Prototype for early Alzheimer’s & Depression risk screening using EEG, questionnaires and cognitive micro-tasks.",
        "upload": "1) Upload EEG file (.edf)",
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
            "Speaking slowly or being unusually restless",
            "Thoughts of being better off dead or self-harm"
        ],
        "phq9_options": [
            "0 = Not at all", "1 = Several days",
            "2 = More than half the days", "3 = Nearly every day"
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
        "title": "🧠 نيوروإيرلي برو — تخطيط الدماغ + PHQ-9 + AD8",
        "subtitle": "نموذج أولي للفحص المبكر لمخاطر الزهايمر والاكتئاب باستخدام EEG والاستبيانات والاختبارات المعرفية الصغيرة.",
        "upload": "١) تحميل ملف تخطيط الدماغ (.edf)",
        "phq9": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "note": "⚠️ هذا نموذج بحثي فقط — ليس أداة تشخيص سريري.",
        "phq9_questions": [
            "قلة الاهتمام أو المتعة في القيام بالأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم أو النوم المفرط",
            "الشعور بالتعب أو قلة الطاقة",
            "فقدان الشهية أو الإفراط في الأكل",
            "الشعور بأنك شخص سيء أو فاشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الكلام ببطء شديد أو فرط الحركة غير المعتاد",
            "أفكار بأنك أفضل حالاً ميتاً أو أفكار إيذاء النفس"
        ],
        "phq9_options": [
            "0 = أبداً", "1 = عدة أيام",
            "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"
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
# Arabic Font (in-memory)
# ---------------------------
def ensure_amiri():
    if "Amiri" in pdfmetrics.getRegisteredFontNames():
        return
    url = "https://github.com/alif-type/amiri-font/raw/master/ttf/Amiri-Regular.ttf"
    r = requests.get(url)
    font_bytes = io.BytesIO(r.content)
    pdfmetrics.registerFont(TTFont("Amiri", font_bytes))

# ---------------------------
# EEG Helper (with filtering)
# ---------------------------
def preprocess_eeg(raw: mne.io.BaseRaw):
    raw.filter(0.5, 45, fir_design="firwin", verbose=False)  # Band-pass
    raw.notch_filter([50, 60], verbose=False)  # Remove powerline noise
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
def build_pdf(results: dict, lang="en", band_png=None) -> bytes:
    ensure_amiri()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    if lang == "ar":
        for s in ["Normal", "Heading2", "Title", "Italic"]:
            styles[s].fontName = "Amiri"

    flow = []
    flow.append(Paragraph(TEXTS[lang]["title"], styles["Title"]))
    flow.append(Paragraph(TEXTS[lang]["subtitle"], styles["Normal"]))
    flow.append(Spacer(1, 12))

    # EEG Results
    flow.append(Paragraph("EEG Results:", styles["Heading2"]))
    rows = [["Band", "Power"]]
    for k, v in results["bands"].items():
        rows.append([k, f"{v:.4f}"])
    table = Table(rows, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(f"PHQ-9 Score: {results['phq_score']} / 27", styles["Normal"]))
    flow.append(Paragraph(f"AD8 Score: {results['ad8_score']} / 8", styles["Normal"]))
    flow.append(Spacer(1, 12))

    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=400, height=200))
        flow.append(Spacer(1, 12))

    flow.append(Paragraph(TEXTS[lang]["note"], styles["Italic"]))
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

# 1) EEG Upload
st.header(t["upload"])
uploaded = st.file_uploader("EDF", type=["edf"])
bands = {}
band_png = None
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
        raw = preprocess_eeg(raw)
        bands = compute_band_powers(raw)
        band_png = plot_bands(bands)
        st.image(band_png)

# 2) PHQ-9
st.header(t["phq9"])
phq_answers = []
for i, q in enumerate(t["phq9_questions"], 1):
    ans = st.selectbox(q, t["phq9_options"], key=f"phq{i}")
    phq_answers.append(int(ans.split("=")[0].strip()))
phq_score = sum(phq_answers)

# 3) AD8
st.header(t["ad8"])
ad8_answers = []
for i, q in enumerate(t["ad8_questions"], 1):
    ans = st.selectbox(q, t["ad8_options"], key=f"ad8{i}")
    ad8_answers.append(1 if ans == t["ad8_options"][1] else 0)
ad8_score = sum(ad8_answers)

# 4) Reports
st.header(t["report"])
if st.button("Generate"):
    results = {"bands": bands, "phq_score": phq_score, "ad8_score": ad8_score}
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode())
    st.download_button(t["download_json"], json_bytes, file_name="report.json")

    pdf_bytes = build_pdf(results, lang=lang, band_png=band_png)
    st.download_button(t["download_pdf"], pdf_bytes, file_name="report.pdf")
