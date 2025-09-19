# app.py — NeuroEarly Pro (final, fixed Q5 & Q8, Arabic + advanced preprocessing + bilingual PDF)
import io
import os
import json
import math
import tempfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

# PDF libs
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# optional Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except Exception:
    ARABIC_SUPPORT = False

# ---------- Config ----------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
st.sidebar.title("NeuroEarly Pro")
st.sidebar.info("Research demo — not a clinical diagnosis")

LANG_UI = st.sidebar.selectbox("Interface language / لغة الواجهة", ["English", "العربية"])
IS_EN = LANG_UI == "English"

# ---------- UI texts ----------
TEXT_UI = {
    "English": {
        "title": "🧠 NeuroEarly Pro — EEG Screening (Demo)",
        "subtitle": "Advanced preprocessing, PHQ-9, AD8 and professional bilingual PDF report (research only).",
        "upload": "1) Upload EDF file(s)",
        "upload_hint": "Upload one EDF for a session or multiple EDFs for longitudinal trend.",
        "phq": "PHQ-9 (Depression)",
        "ad8": "AD8 (Cognitive screening)",
        "generate": "Generate Reports (JSON / CSV / PDF)",
        "note": "Research demo. Not a clinical diagnostic tool."
    },
    "العربية": {
        "title": "🧠 نيوروإيرلي برو — فحص EEG (نموذج تجريبي)",
        "subtitle": "معالجة متقدمة، PHQ-9، AD8 وتقرير PDF احترافي ثنائي اللغة (للبحث فقط).",
        "upload": "١) تحميل ملفات EDF",
        "upload_hint": "ارفع ملف EDF واحد للجلسة أو عدة ملفات لتحليل طولي.",
        "phq": "PHQ-9 (الاكتئاب)",
        "ad8": "AD8 (الفحص المعرفي)",
        "generate": "إنشاء التقارير (JSON / CSV / PDF)",
        "note": "هذا نموذج بحثي. ليس تشخيصًا طبيًا."
    }
}
TUI = TEXT_UI["English"] if IS_EN else TEXT_UI["العربية"]

# ---------- Questionnaire texts (both languages) ----------
PHQ_QS = {
    "English": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating (specify type below)",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things (e.g., reading, watching TV)",
        "Moving or speaking slowly vs. being restless (choose best)",
        "Thoughts that you would be better off dead or of hurting yourself"
    ],
    "العربية": [
        "قلة الاهتمام أو المتعة في القيام بالأنشطة",
        "الشعور بالحزن أو الاكتئاب أو اليأس",
        "مشاكل في النوم أو النوم المفرط",
        "الشعور بالتعب أو قلة الطاقة",
        "فقدان الشهية أو الإفراط في الأكل (حدد النوع أدناه)",
        "الشعور بأنك شخص سيء أو فاشل",
        "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
        "بطء في الحركة أو الكلام مقابل فرط النشاط (اختر الأنسب)",
        "أفكار بأنك أفضل حالاً ميتاً أو أفكار إيذاء النفس"
    ]
}
PHQ_OPTS = {
    "English": ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"],
    "العربية": ["0 = أبداً", "1 = عدة أيام", "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"]
}
SPECIAL_Q8 = {
    "English": ["0 = Neither slow nor restless", "1 = Mostly calm/slow", "2 = Mostly restless", "3 = Both slow and restless"],
    "العربية": ["0 = لا بطيء ولا مفرط الحركة", "1 = غالباً هادئ / بطيء", "2 = غالباً مفرط النشاط", "3 = كلاهما بوضوح"]
}

AD8_QS = {
    "English": [
        "Problems with judgment (e.g., poor financial decisions)",
        "Reduced interest in hobbies/activities",
        "Repeats the same questions or stories",
        "Trouble learning to use a tool or gadget",
        "Forgets the correct month or year",
        "Difficulty handling finances (e.g., paying bills)",
        "Trouble remembering appointments",
        "Everyday thinking is getting worse"
    ],
    "العربية": [
        "مشاكل في الحكم أو اتخاذ القرارات",
        "انخفاض الاهتمام بالهوايات أو الأنشطة",
        "تكرار الأسئلة أو القصص",
        "صعوبة في تعلم استخدام أداة أو جهاز",
        "نسيان الشهر أو السنة الصحيحة",
        "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
        "صعوبة في تذكر المواعيد",
        "تدهور التفكير اليومي"
    ]
}
AD8_OPTS = {"English": ["No", "Yes"], "العربية": ["لا", "نعم"]}

# (کد ادامه دارد … شامل نویزگیری، محاسبات باند، PDF دو‌زبانه و بقیه‌ی بخش‌ها)
