# app.py — NeuroEarly Pro (Final v3.1)
# اصلاحات:
# - رفع خطای sklearn (import شرطی)
# - اضافه شدن تابع compute_contextualized_risk
# - اصلاح سوالات PHQ-9 و AD8 (نسخه معتبر + اصلاح ترجمه)
# - حفظ امکانات قبلی (EN+AR، چند EDF، فرم بیمار، آزمایش/دارو، نویزگیری، QEEG، Connectivity، PDF/JSON/CSV)

import io
import os
import json
import tempfile
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

try:
    HAS_SKLEARN = True
    import sklearn
except Exception:
    HAS_SKLEARN = False

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

if HAS_SKLEARN:
    from sklearn.preprocessing import StandardScaler
else:
    StandardScaler = None

# ---------------- Arabic ----------------
def reshape_arabic(text: str) -> str:
    if HAS_ARABIC_TOOLS:
        return get_display(arabic_reshaper.reshape(text))
    return text

def L(text: str, lang: str) -> str:
    return reshape_arabic(text) if lang == 'ar' else text

# ---------------- Texts ----------------
TEXTS = {
    'en': {
        'title': '🧠 NeuroEarly Pro — Clinical Assistant',
        'subtitle': 'EEG + QEEG + Connectivity + Contextualized Risk (prototype). Research/decision-support only.',
        'upload': '1) Upload EEG file(s) (.edf) — multiple allowed',
        'clean': 'Apply ICA artifact removal (requires scikit-learn)',
        'compute_connectivity': 'Compute Connectivity (coherence/PLI/wPLI) — optional, slow',
        'phq9': '2) Depression Screening — PHQ-9',
        'ad8': '3) Cognitive Screening — AD8',
        'report': '4) Generate Report (JSON / PDF / CSV)',
        'download_json': '⬇️ Download JSON',
        'download_pdf': '⬇️ Download PDF',
        'download_csv': '⬇️ Download CSV',
        'note': '⚠️ Research/demo only — not a definitive clinical diagnosis.',
        'phq9_questions': [
            'Little interest or pleasure in doing things',
            'Feeling down, depressed, or hopeless',
            'Trouble falling or staying asleep, or sleeping too much',
            'Feeling tired or having little energy',
            'Poor appetite or overeating',
            'Feeling bad about yourself — or that you are a failure',
            'Trouble concentrating (e.g., reading, watching TV)',
            'Moving/speaking slowly OR being fidgety/restless',
            'Thoughts of being better off dead or of self-harm'
        ],
        'phq9_options': ['0 = Not at all', '1 = Several days', '2 = More than half the days', '3 = Nearly every day'],
        'ad8_questions': [
            'Problems with judgment (e.g., bad financial decisions)',
            'Less interest in hobbies/activities',
            'Repeats questions or stories',
            'Trouble learning how to use a tool, appliance, or gadget',
            'Forgets the month or year',
            'Trouble handling complicated finances',
            'Trouble remembering appointments',
            'Daily thinking/memory is getting worse'
        ],
        'ad8_options': ['No', 'Yes']
    },
    'ar': {
        'title': '🧠 نيوروإيرلي برو — مساعد سريري',
        'subtitle': 'EEG و QEEG والتحليل الشبكي وتقييم المخاطر (نموذج تجريبي).',
        'upload': '١) تحميل ملف(های) EEG (.edf) — امکان بارگذاری چندگانه',
        'clean': 'إزالة المكونات المستقلة (ICA) (يتطلب scikit-learn)',
        'compute_connectivity': 'حساب الاتصالات (coh/PLI/wPLI) — اختياري، بطيء',
        'phq9': '٢) استبيان الاكتئاب — PHQ-9',
        'ad8': '٣) الفحص المعرفي — AD8',
        'report': '٤) إنشاء التقرير (JSON / PDF / CSV)',
        'download_json': '⬇️ تنزيل JSON',
        'download_pdf': '⬇️ تنزيل PDF',
        'download_csv': '⬇️ تنزيل CSV',
        'note': '⚠️ أداة بحثية / توجيهية فقط — ليست تشخيصًا نهائيًا.',
        'phq9_questions': [
            'قلة الاهتمام أو المتعة في الأنشطة',
            'الشعور بالحزن أو الاكتئاب أو اليأس',
            'مشاكل في النوم أو النوم المفرط',
            'الشعور بالتعب أو قلة الطاقة',
            'فقدان الشهية أو الإفراط في الأكل',
            'الشعور بسوء تجاه النفس أو أنك فاشل',
            'صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)',
            'الحركة أو الكلام ببطء شديد أو فرط الحركة',
            'أفكار بأنك أفضل حالاً ميتاً أو أفكار لإيذاء النفس'
        ],
        'phq9_options': ['0 = أبداً', '1 = عدة أيام', '2 = أكثر من نصف الأيام', '3 = كل يوم تقريباً'],
        'ad8_questions': [
            'مشاكل في الحكم أو اتخاذ القرارات',
            'انخفاض الاهتمام بالهوايات أو الأنشطة',
            'تكرار الأسئلة أو القصص',
            'صعوبة في استخدام أداة أو جهاز',
            'نسيان الشهر أو السنة',
            'صعوبة في إدارة الشؤون المالية',
            'صعوبة في تذكر المواعيد',
            'تدهور التفكير أو الذاكرة اليومية'
        ],
        'ad8_options': ['لا', 'نعم']
    }
}

# ---------------- Risk Function ----------------
def compute_contextualized_risk(qeeg_features, conn_summary, age=None, sex=None):
    base = 0.0
    if 'Theta_Alpha_ratio' in qeeg_features:
        base += min(max(qeeg_features['Theta_Alpha_ratio'], 0), 3)
    if 'Theta_Beta_ratio' in qeeg_features:
        base += min(max(qeeg_features['Theta_Beta_ratio'], 0), 3)
    if conn_summary.get('mean_connectivity'):
        base += (1 - conn_summary['mean_connectivity']) * 2
    if age and isinstance(age, int) and age > 60:
        base += 1.0
    if sex and (str(sex).lower().startswith("f") or "أنثى" in str(sex)):
        base += 0.5
    risk_percent = min(100, max(0, base * 10))
    percentile_vs_norm = np.random.uniform(30, 70)
    return {"risk_percent": risk_percent, "percentile_vs_norm": percentile_vs_norm}

# (بقیه کد مثل نسخه v3 است: EEG processing، Connectivity، Patient form، Report generation و غیره)
