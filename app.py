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

# ---------------------------
# Language dictionaries
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly — EEG + Depression (PHQ-9) + AD8 (Demo)",
        "subtitle": "Prototype for early screening using EEG, PHQ-9 (depression), and AD8 (cognition).",
        "upload_header": "1) Upload EEG (.edf)",
        "phq_header": "2) Depression Screening — PHQ-9",
        "ad8_header": "3) Cognitive Screening — AD8",
        "report_header": "4) Generate Reports",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "note": "This is a research demo — not a clinical diagnostic tool.",
        "phq_questions": [
            "1. Little interest or pleasure in doing things",
            "2. Feeling down, depressed, or hopeless",
            "3. Trouble falling or staying asleep, or sleeping too much",
            "4. Feeling tired or having little energy",
            "5. Poor appetite or overeating",
            "6. Feeling bad about yourself — or that you are a failure",
            "7. Trouble concentrating on reading or watching TV",
            "8. Moving or speaking noticeably slowly or being restless",
            "9. Thoughts that you would be better off dead or hurting yourself",
        ],
        "phq_options": [
            "0 = Not at all",
            "1 = Several days",
            "2 = More than half the days",
            "3 = Nearly every day",
        ],
        "ad8_questions": [
            "1. Problems with judgment (bad financial decisions)",
            "2. Reduced interest in hobbies/activities",
            "3. Repeats the same questions or stories",
            "4. Trouble learning how to use tools or appliances",
            "5. Forgets the correct month or year",
            "6. Difficulty handling finances (e.g., paying bills)",
            "7. Trouble remembering appointments",
            "8. Everyday thinking is getting worse",
        ],
        "ad8_options": ["No", "Yes"],
    },
    "ar": {
        "title": "🧠 نيوروإرلي — تخطيط الدماغ + الاكتئاب (PHQ-9) + الإدراك (AD8)",
        "subtitle": "نموذج أولي للفحص المبكر باستخدام EEG، اختبار PHQ-9 للاكتئاب، واختبار AD8 للذاكرة.",
        "upload_header": "١) تحميل ملف EEG (.edf)",
        "phq_header": "٢) فحص الاكتئاب — PHQ-9",
        "ad8_header": "٣) فحص الإدراك — AD8",
        "report_header": "٤) إنشاء التقارير",
        "download_json": "⬇️ تحميل ملف JSON",
        "download_pdf": "⬇️ تحميل ملف PDF",
        "note": "هذا نموذج بحثي — ليس أداة تشخيص طبية.",
        "phq_questions": [
            "١. قلة الاهتمام أو المتعة في القيام بالأشياء",
            "٢. الشعور بالحزن أو الاكتئاب أو اليأس",
            "٣. صعوبة في النوم أو النوم المفرط",
            "٤. الشعور بالتعب أو قلة الطاقة",
            "٥. فقدان الشهية أو الإفراط في الأكل",
            "٦. الشعور بأنك عديم القيمة أو فاشل",
            "٧. صعوبة في التركيز على القراءة أو مشاهدة التلفاز",
            "٨. بطء في الحركة أو الكلام أو العكس (توتر زائد)",
            "٩. أفكار عن الموت أو إيذاء النفس",
        ],
        "phq_options": [
            "٠ = أبداً",
            "١ = عدة أيام",
            "٢ = أكثر من نصف الأيام",
            "٣ = تقريباً كل يوم",
        ],
        "ad8_questions": [
            "١. مشاكل في الحكم واتخاذ القرارات",
            "٢. قلة الاهتمام بالهوايات أو الأنشطة",
            "٣. يكرر نفس الأسئلة أو القصص",
            "٤. صعوبة في تعلم استخدام أدوات أو أجهزة",
            "٥. ينسى الشهر أو السنة الصحيحة",
            "٦. صعوبة في إدارة الأمور المالية (مثل دفع الفواتير)",
            "٧. صعوبة في تذكر المواعيد",
            "٨. تدهور التفكير اليومي",
        ],
        "ad8_options": ["لا", "نعم"],
    },
}

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="NeuroEarly — Multilingual Demo", layout="centered")

# Language selector
lang = st.radio("🌐 Language / اللغة", options=["en", "ar"], format_func=lambda x: "English" if x=="en" else "العربية")
T = TEXTS[lang]

st.title(T["title"])
st.caption(T["subtitle"])

# ---------------------------
# EEG Upload
# ---------------------------
st.header(T["upload_header"])
uploaded = st.file_uploader("Upload EDF file" if lang=="en" else "تحميل ملف EDF", type=["edf"])

raw = None
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    st.success("EEG file loaded!" if lang=="en" else "تم تحميل ملف EEG بنجاح")

# ---------------------------
# PHQ-9
# ---------------------------
st.header(T["phq_header"])
phq_answers = []
for i, q in enumerate(T["phq_questions"], 1):
    ans = st.selectbox(q, T["phq_options"], index=0, key=f"phq9_{i}_{lang}")
    phq_answers.append(int(ans.split("=")[0].strip()) if lang=="en" else T["phq_options"].index(ans))
phq_score = sum(phq_answers)
st.write(f"PHQ-9 Score: {phq_score} / 27" if lang=="en" else f"درجة PHQ-9: {phq_score} / ٢٧")

# ---------------------------
# AD8
# ---------------------------
st.header(T["ad8_header"])
ad8_answers = []
for i, q in enumerate(T["ad8_questions"], 1):
    ans = st.selectbox(q, T["ad8_options"], index=0, key=f"ad8_{i}_{lang}")
    ad8_answers.append(1 if ans in ["Yes","نعم"] else 0)
ad8_score = sum(ad8_answers)
st.write(f"AD8 Score: {ad8_score} / 8" if lang=="en" else f"درجة AD8: {ad8_score} / ٨")

# ---------------------------
# Reports
# ---------------------------
st.header(T["report_header"])
if st.button("📑 Create Report" if lang=="en" else "📑 إنشاء تقرير"):
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Depression (PHQ-9)": phq_score,
        "Cognition (AD8)": ad8_score,
    }

    # JSON
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button(T["download_json"], data=json_bytes, file_name="report.json", mime="application/json")

    # PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(T["title"], styles["Title"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"Timestamp: {results['timestamp']}", styles["Normal"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"PHQ-9: {phq_score}", styles["Normal"]))
    flow.append(Paragraph(f"AD8: {ad8_score}", styles["Normal"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(T["note"], styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    st.download_button(T["download_pdf"], data=buf, file_name="report.pdf", mime="application/pdf")

st.caption(T["note"])
