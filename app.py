# app.py — NeuroEarly Pro (Final v2)
# Complete Streamlit app: multi-EDF upload, preprocessing, QEEG, Connectivity (coh/PLI/wPLI),
# contextualized ML risk score (synthetic norms), PHQ-9/AD8, patient form, labs, meds,
# PDF/JSON/CSV outputs, bilingual EN/AR, archive. Research/decision-support only.

import io
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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

# Optional Arabic tools
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

# ML libs
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# SciPy
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- CONFIG ----------------
AMIRI_PATH = "Amiri-Regular.ttf"  # optional font file for Arabic PDF
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
DEFAULT_NOTCH = [50, 100]
ARCHIVE_DIR = "archive"

if os.path.exists(AMIRI_PATH):
    try:
        if "Amiri" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))
    except Exception:
        pass

# ---------------- utilities ----------------
def reshape_arabic(text: str) -> str:
    if HAS_ARABIC_TOOLS:
        return get_display(arabic_reshaper.reshape(text))
    return text


def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


# ---------------- TEXTS (EN + AR) ----------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro — Clinical Assistant",
        "subtitle": "EEG + QEEG + Connectivity + Contextualized Risk (prototype). Research/decision-support only.",
        "upload": "1) Upload EEG file(s) (.edf) — multiple allowed",
        "clean": "Apply ICA artifact removal (requires scikit-learn)",
        "compute_connectivity": "Compute Connectivity (coherence/PLI/wPLI) — optional, slow",
        "phq9": "2) Depression Screening — PHQ-9",
        "ad8": "3) Cognitive Screening — AD8",
        "report": "4) Generate Report (JSON / PDF / CSV)",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "download_csv": "⬇️ Download CSV",
        "note": "⚠️ Research/demo only — not a definitive clinical diagnosis.",
        "phq9_questions": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating (changes in appetite)",
            "Feeling bad about yourself — or that you are a failure",
            "Trouble concentrating (e.g., reading, watching TV)",
            "Moving or speaking so slowly that other people notice, OR the opposite — being so fidgety or restless",
            "Thoughts that you would be better off dead or of hurting yourself"
        ],
        "phq9_options": ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"],
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
        "title": "🧠 نيوروإيرلي برو — مساعد سريري",
        "subtitle": "المعالجة بالـ EEG و QEEG والتحليل الشبكي وتقييم المخاطر (نموذج تجريبي).",
        "upload": "١) تحميل ملف(های) EEG (.edf) — امکان بارگذاری چندگانه",
        "clean": "إزالة المكونات المستقلة (ICA) (يتطلب scikit-learn)",
        "compute_connectivity": "حساب الاتصالات (coh/PLI/wPLI) — اختياري، بطيء",
        "phq9": "٢) استبيان الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير (JSON / PDF / CSV)",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "download_csv": "⬇️ تنزيل CSV",
        "note": "⚠️ أداة بحثية / توجيهية فقط — ليست تشخيصًا نهائيًا.",
        "phq9_questions": [
            "قلة الاهتمام أو المتعة في الأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم أو النوم المفرط",
            "الشعور بالتعب أو قلة الطاقة",
            "تغيرات في الشهية (قِلَّة أو إفراط في الأكل)",
            "الشعور بسوء تجاه النفس أو أنك فاشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الحركة أو الكلام ببطء شديد بحيث يلاحظه الآخرون، أو العكس — فرِط الحركة",
            "أفكار بأنك أفضل حالاً لو كنت ميتاً أو إيذاء النفس"
        ],
        "phq9_options": ["0 = أبداً", "1 = عدة أيام", "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"],
        "ad8_questions": [
            "مشاكل في الحكم (مثل قرارات مالية سيئة)",
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


# ---------------- EEG preprocessing & QEEG ----------------
def preprocess_raw(raw: mne.io.BaseRaw, l_freq=1.0, h_freq=45.0, notch_freqs: Optional[List[int]] = DEFAULT_NOTCH, downsample: Optional[int] = None) -> mne.io.BaseRaw:
    raw = raw.copy()
    try:
        raw.load_data()
    except Exception:
        pass
    try:
        if downsample and raw.info['sfreq'] > downsample:
            raw.resample(downsample)
    except Exception:
        pass
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    except Exception:
        pass
    try:
        if notch_freqs:
            raw.notch_filter(freqs=notch_freqs, verbose=False)
    except Exception:
        pass
    try:
        raw.set_eeg_reference('average', verbose=False)
    except Exception:
        pass
    return raw


def compute_band_powers_per_channel(raw: mne.io.BaseRaw, bands=BANDS) -> Dict:
    try:
        psd = raw.compute_psd(fmin=0.5, fmax=45, method="welch", verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
    except Exception:
        try:
            psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, verbose=False)
        except Exception as e:
            raise RuntimeError(f"PSD computation failed: {e}")
    band_abs = {}
    band_per_channel = {}
    total_power_per_channel = np.trapz(psds, freqs, axis=1)
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power_ch = np.trapz(psds[:, mask], freqs[mask], axis=1)
        band_per_channel[name] = band_power_ch
        band_abs[name] = float(np.mean(band_power_ch))
    total_mean = sum(band_abs.values()) + 1e-12
    band_rel = {k: float(v / total_mean) for k, v in band_abs.items()}
    return {'abs_mean': band_abs, 'rel_mean': band_rel, 'per_channel': band_per_channel, 'total_power_per_channel': total_power_per_channel, 'freqs': freqs}


def compute_qeeg_features(raw: mne.io.BaseRaw) -> Tuple[Dict, Dict]:
    raw = preprocess_raw(raw)
    bp = compute_band_powers_per_channel(raw)
    feats = {}
    for b, v in bp['abs_mean'].items():
        feats[f"{b}_abs_mean"] = v
    for b, v in bp['rel_mean'].items():
        feats[f"{b}_rel_mean"] = v
    if 'Theta' in bp['abs_mean'] and 'Beta' in bp['abs_mean']:
        feats['Theta_Beta_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Beta'] + 1e-12)
    if 'Theta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
        feats['Theta_Alpha_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Alpha'] + 1e-12)
    def idx(ch_name):
        try:
            return raw.ch_names.index(ch_name)
        except Exception:
            return None
    for left, right in [('F3','F4'), ('Fp1','Fp2'), ('F7','F8')]:
        i = idx(left); j = idx(right)
        if i is not None and j is not None:
            alpha_power = bp['per_channel'].get('Alpha')
            if alpha_power is not None:
                feats[f'alpha_asym_{left}_{right}'] = float(np.log(alpha_power[i] + 1e-12) - np.log(alpha_power[j] + 1e-12))
    return feats, bp


# ---------------- Connectivity (final) ----------------
def compute_connectivity_final(raw: mne.io.BaseRaw, method='wpli', fmin=4.0, fmax=30.0, epoch_len=2.0, picks: Optional[List[str]] = None, mode='fourier', n_jobs=1) -> Dict:
    try:
        from mne.connectivity import spectral_connectivity
    except Exception:
        return {'error': 'mne.connectivity not available'}
    try:
        if picks is None:
            picks_idx = mne.pick_types(raw.info, eeg=True, meg=False)
            chs = [raw.ch_names[i] for i in picks_idx]
        else:
            picks_idx = [raw.ch_names.index(ch) for ch in picks if ch in raw.ch_names]
            chs = [raw.ch_names[i] for i in picks_idx]
        if len(picks_idx) < 2:
            return {'error': 'Not enough channels for connectivity'}
        sf = int(raw.info['sfreq'])
        win_samp = int(epoch_len * sf)
        data = raw.get_data(picks=picks_idx)
        n_samps = data.shape[1]
        n_epochs = n_samps // win_samp
        if n_epochs >= 2:
            epochs_data = []
            for ei in range(n_epochs):
                start = ei * win_samp
                stop = start + win_samp
                epochs_data.append(data[:, start:stop])
            epochs_array = np.stack(epochs_data)
            info = mne.create_info([raw.ch_names[i] for i in picks_idx], sf, ch_types='eeg')
            epochs = mne.EpochsArray(epochs_array, info)
            data_for_conn = epochs
        else:
            data_for_conn = raw
        con, freqs, times, n_epochs_out, n_tapers = spectral_connectivity(
            data_for_conn, method=method, mode=mode, sfreq=raw.info['sfreq'],
            fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=False, n_jobs=n_jobs, verbose=False
        )
        mean_con = np.nanmean(con, axis=1)
        n_ch = len(picks_idx)
        mat = np.zeros((n_ch, n_ch))
        idx = 0
        for i in range(n_ch):
            for j in range(i+1, n_ch):
                if idx < len(mean_con):
                    mat[i, j] = mean_con[idx]
                    mat[j, i] = mean_con[idx]
                idx += 1
        return {'matrix': mat, 'channels': chs, 'mean_connectivity': float(np.nanmean(mean_con))}
    except Exception as e:
        return {'error': str(e)}


# ---------------- Plot helpers ----------------
def plot_band_bar(band_dict: Dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(list(band_dict.keys()), list(band_dict.values()))
    ax.set_title('EEG Band Powers (mean across channels)')
    ax.set_ylabel('Power (a.u.)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def plot_connectivity_heatmap(mat: np.ndarray, chs: List[str]) -> bytes:
    fig, ax = plt.subplots(figsize=(6,5))
    vmax = np.nanpercentile(mat, 95) if np.any(mat) else 1.0
    im = ax.imshow(mat, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(chs))); ax.set_xticklabels(chs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(chs))); ax.set_yticklabels(chs, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Connectivity (heatmap)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# ---------------- ML + Normative ----------------
def build_synthetic_dataset(n=500):
    rng = np.random.RandomState(42)
    ta = rng.normal(1.0, 0.4, n)
    tb = rng.normal(1.0, 0.6, n)
    asym = rng.normal(0.0, 0.3, n)
    conn = rng.normal(0.25, 0.1, n)
    logit = 0.8*(ta-1.0) + 0.6*(tb-1.0) + 0.9*np.maximum(asym, 0) - 1.2*(conn-0.25)
    prob = 1/(1+np.exp(-logit))
    y = (prob > 0.5).astype(int)
    X = np.vstack([ta, tb, np.abs(asym), conn]).T
    return X, y


def train_initial_model():
    if not HAS_SKLEARN:
        return None, None
    X, y = build_synthetic_dataset(1000)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y)
    return clf, scaler


MODEL, SCALER = train_initial_model()


# Synthetic normative stats
def build_synthetic_norms():
    X, _ = build_synthetic_dataset(2000)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    feat_names = ['Theta_Alpha_ratio', 'Theta_Beta_ratio', 'alpha_asym_abs', 'mean_connectivity']
    stats_dict = {'global': {}}
    for i, fn in enumerate(feat_names):
        stats_dict['global'][fn] = {'mu': float(mu[i]), 'sigma': float(sigma[i])}
    return stats_dict


NORMATIVE_STATS = build_synthetic_norms()


def compute_contextualized_risk(qeeg_feats: Dict, conn_summary: Dict, age: Optional[int]=None, sex: Optional[str]=None, normative_stats=NORMATIVE_STATS, model=MODEL, scaler=SCALER) -> Dict:
    ta = qeeg_feats.get('Theta_Alpha_ratio', 0.0)
    tb = qeeg_feats.get('Theta_Beta_ratio', 0.0)
    asym_vals = [v for k,v in qeeg_feats.items() if k.startswith('alpha_asym_')]
    asym_abs = max([abs(a) for a in asym_vals]) if asym_vals else 0.0
    conn_val = conn_summary.get('mean_connectivity') if conn_summary and 'mean_connectivity' in conn_summary else None
    stats_g = normative_stats.get('global', {})
    def safe_z(val, name):
        if name in stats_g:
            mu = stats_g[name]['mu']; sigma = stats_g[name]['sigma']
            return (val - mu) / (sigma + 1e-12)
        return 0.0
    z_ta = safe_z(ta, 'Theta_Alpha_ratio'); z_tb = safe_z(tb, 'Theta_Beta_ratio')
    z_asym = safe_z(asym_abs, 'alpha_asym_abs')
    z_conn = safe_z(conn_val if conn_val is not None else stats_g.get('mean_connectivity', {}).get('mu', 0.0), 'mean_connectivity') if 'mean_connectivity' in stats_g or conn_val is not None else 0.0
    z_list = [z_ta, z_tb, z_asym, z_conn]
    combined_z = float(np.mean(z_list))
    percentile = float(stats.norm.cdf(combined_z) * 100) if HAS_SCIPY else float(50 + combined_z * 10)
    prob = None
    if model is not None and scaler is not None and conn_val is not None:
        X = np.array([[ta, tb, asym_abs, conn_val]])
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0,1] * 100)
    else:
        prob = percentile * 0.65
    return {'combined_z': combined_z, 'percentile_vs_norm': percentile, 'risk_percent': prob}


# ---------------- PDF builder ----------------
def build_pdf(results: Dict, patient_info: Dict, lab_results: Dict, meds: List[str], lang='en', band_pngs: Dict[str, bytes]=None, conn_images: Dict[str, bytes]=None, interpretations: List[str]=None, risk_scores: Dict[str,float]=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36,leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    if lang=='ar' and os.path.exists(AMIRI_PATH):
        for s in ['Normal','Title','Heading2','Italic']:
            styles[s].fontName='Amiri'
    flow = []
    t = TEXTS[lang]
    L = lambda txt: reshape_arabic(txt) if lang=='ar' else txt
    flow.append(Paragraph(L(t['title']), styles['Title']))
    flow.append(Paragraph(L(t['subtitle']), styles['Normal']))
    flow.append(Spacer(1,8))
    flow.append(Paragraph(L(f"Generated: {results.get('timestamp','')}"), styles['Normal']))
    flow.append(Spacer(1,8))
    flow.append(Paragraph(L("Patient information:"), styles['Heading2']))
    if any(patient_info.values()):
        p_rows = []
        for k in ['name','id','gender','dob','age','phone','email','history']:
            if patient_info.get(k):
                p_rows.append([k.capitalize(), str(patient_info.get(k))])
        pt_table = Table(p_rows, colWidths=[120,300])
        pt_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black)]))
        flow.append(pt_table)
    else:
        flow.append(Paragraph(L("No patient info provided."), styles['Normal']))
    flow.append(Spacer(1,8))
    flow.append(Paragraph(L("Recent labs:"), styles['Heading2']))
    if lab_results:
        lab_rows=[["Test","Value"]]
        for k,v in lab_results.items():
            lab_rows.append([k,str(v)])
        lab_table = Table(lab_rows, colWidths=[200,200])
        lab_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(lab_table)
    else:
        flow.append(Paragraph(L("No labs provided."), styles['Normal']))
    flow.append(Spacer(1,6))
    flow.append(Paragraph(L("Medications:"), styles['Heading2']))
    if meds:
        for m in meds:
            flow.append(Paragraph(L(m), styles['Normal']))
    else:
        flow.append(Paragraph(L("No medications listed."), styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(L("EEG & QEEG results:"), styles['Heading2']))
    for fname, block in results.get('EEG_files', {}).items():
        flow.append(Paragraph(L(f"File: {fname}"), styles['Heading2']))
        rows=[["Band","Absolute","Relative"]]
        for k,v in block.get('bands', {}).items():
            rel = block.get('relative', {}).get(k,0)
            rows.append([k, f"{v:.4f}", f"{rel:.4f}"])
        tble = Table(rows, colWidths=[120,120,120])
        tble.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(tble); flow.append(Spacer(1,6))
        qrows=[["Feature","Value"]]
        for kk,vv in block.get('QEEG', {}).items():
            qrows.append([kk, fmt(vv) if isinstance(vv,(int,float)) else str(vv)])
        qtab = Table(qrows, colWidths=[240,120])
        qtab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(qtab); flow.append(Spacer(1,6))
        conn = block.get('connectivity', {})
        if conn:
            flow.append(Paragraph(L("Connectivity summary:"), styles['Normal']))
            for ck,cv in conn.items():
                if ck=='matrix': continue
                flow.append(Paragraph(L(f"{ck}: {fmt(cv)}"), styles['Normal']))
        if risk_scores and fname in risk_scores:
            flow.append(Spacer(1,6)); flow.append(Paragraph(L(f"Contextualized risk (prelim.): {risk_scores[fname]:.1f}%"), styles['Normal']))
        if band_pngs and fname in band_pngs:
            flow.append(RLImage(io.BytesIO(band_pngs[fname]), width=400, height=140)); flow.append(Spacer(1,6))
        if conn_images and fname in conn_images:
            flow.append(RLImage(io.BytesIO(conn_images[fname]), width=400, height=200)); flow.append(Spacer(1,6))
        flow.append(Spacer(1,10))
    flow.append(Paragraph(L("Automated interpretation (heuristic):"), styles['Heading2']))
    if interpretations:
        for line in interpretations:
            flow.append(Paragraph(L(line), styles['Normal']))
    else:
        flow.append(Paragraph(L("No heuristic interpretations."), styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(L("Structured recommendations (for clinician):"), styles['Heading2']))
    recs = [
        "Correlate QEEG/connectivity findings with PHQ-9 and AD8 and clinical interview.",
        "If PHQ-9 suggests moderate/severe depression or left frontal alpha asymmetry found, consider psychiatric referral and treatment planning (psychotherapy ± pharmacotherapy).",
        "If AD8 elevated or theta increase present, consider neurocognitive assessment and neuroimaging (MRI) as needed.",
        "Review current medications for EEG-affecting agents.",
        "If suicidal ideation present (PHQ-9 item), arrange urgent psychiatric evaluation."
    ]
    for r in recs:
        flow.append(Paragraph(L(r), styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(L(TEXTS['en']['note']), styles['Italic']))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"])  # bilingual
t = TEXTS[lang]

st.title(t['title'])
st.write(t['subtitle'])

# Patient form
with st.expander("🔎 Optional: Patient information / معلومات المريض"):
    name = st.text_input("Full name / الاسم الكامل")
    patient_id = st.text_input("Patient ID / رقم المريض")
    if lang=='en':
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"]) 
    else:
        gender = st.selectbox(reshape_arabic("الجنس"), ["", reshape_arabic("ذكر"), reshape_arabic("أنثى"), reshape_arabic("آخر")])
    dob = st.date_input("Date of birth / تاريخ الميلاد", value=None)
    phone = st.text_input("Phone / الهاتف")
    email = st.text_input("Email / البريد الإلكتروني")
    history = st.text_area("Relevant history (diabetes, HTN, family history...) / التاريخ الطبي", height=80)

patient_info = {
    'name': name, 'id': patient_id, 'gender': gender,
    'dob': dob.strftime("%Y-%m-%d") if dob else "", 'age': int((datetime.now().date()-dob).days/365) if dob else "",
    'phone': phone, 'email': email, 'history': history
}

# Labs
with st.expander("🧪 Optional: Recent lab tests / التحاليل"):
    lab_glucose = st.text_input("Glucose")
    lab_b12 = st.text_input("Vitamin B12")
    lab_vitd = st.text_input("Vitamin D")
    lab_tsh = st.text_input("TSH")
    lab_crp = st.text_input("CRP")
lab_results = {}
if lab_glucose: lab_results['Glucose'] = lab_glucose
if lab_b12: lab_results['Vitamin B12'] = lab_b12
if lab_vitd: lab_results['Vitamin D'] = lab_vitd
if lab_tsh: lab_results['TSH'] = lab_tsh
if lab_crp: lab_results['CRP'] = lab_crp

# Medications
with st.expander("💊 Current medications (one per line) / الأدوية الحالية"):
    meds_text = st.text_area("List medications / اكتب الأدوية", height=120)
meds_list = [m.strip() for m in meds_text.splitlines() if m.strip()]

# Tabs
tab_upload, tab_phq, tab_ad8, tab_report = st.tabs([t['upload'], t['phq9'], t['ad8'], t['report']])

# Shared containers
EEG_results = {'EEG_files': {}}
band_pngs = {}
conn_imgs = {}

# Upload tab
with tab_upload:
    st.header(t['upload'])
    uploaded = st.file_uploader("EDF files / ملفات EDF", type=['edf'], accept_multiple_files=True)
    apply_ica = st.checkbox(t['clean'])
    compute_conn = st.checkbox(t['compute_connectivity'])
    conn_method = st.selectbox("Connectivity method / طريقة الاتصالات", ['coh','pli','wpli'])
    notch_choice = st.multiselect("Notch frequencies (Hz) / ترددات Notch", [50, 60, 100, 120], default=[50,100])
    epoch_len = st.slider("Epoch length for connectivity (s)", 1.0, 5.0, 2.0, step=0.5)
    downsample = st.selectbox("Downsample to (Hz) — optional", [None, 256, 200, 128], index=0)
    if uploaded:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        for f in uploaded:
            st.info(f"Processing {f.name} ...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                    tmp.write(f.read()); tmp.flush(); tmp_name = tmp.name
                raw = mne.io.read_raw_edf(tmp_name, preload=True, verbose=False)
                raw = preprocess_raw(raw, notch_freqs=notch_choice if notch_choice else DEFAULT_NOTCH, downsample=downsample)
                if apply_ica:
                    if HAS_SKLEARN:
                        try:
                            ica = mne.preprocessing.ICA(n_components=15, random_state=42, verbose=False)
                            ica.fit(raw)
                            raw = ica.apply(raw)
                        except Exception as e:
                            st.warning(f"ICA failed/skipped: {e}")
                    else:
                        st.warning("scikit-learn not installed — ICA skipped.")
                qeeg, bp = compute_qeeg_features(raw)
                conn_res = {}
                if compute_conn:
                    with st.spinner("Computing connectivity (may be slow)..."):
                        conn_res = compute_connectivity_final(raw, method=conn_method, fmin=4.0, fmax=30.0, epoch_len=epoch_len, picks=None, mode='fourier', n_jobs=1)
                        if 'matrix' in conn_res:
                            try:
                                img = plot_connectivity_heatmap(conn_res['matrix'], conn_res['channels'])
                                conn_imgs[f.name] = img
                            except Exception as e:
                                st.warning(f"Connectivity heatmap failed: {e}")
                EEG_results['EEG_files'][f.name] = {'bands': bp['abs_mean'], 'relative': bp['rel_mean'], 'QEEG': qeeg, 'connectivity': conn_res}
                band_png = plot_band_bar(bp['abs_mean'])
                band_pngs[f.name] = band_png
                st.image(band_png, caption=f"{f.name} — band powers")
                if compute_conn and f.name in conn_imgs:
                    st.image(conn_imgs[f.name], caption=f"{f.name} — connectivity heatmap")
                try:
                    dest = os.path.join(ARCHIVE_DIR, f.name)
                    with open(dest, 'wb') as dst, open(tmp_name, 'rb') as src:
                        dst.write(src.read())
                except Exception:
                    pass
                st.success(f"{f.name} processed.")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")

# PHQ-9 tab
with tab_phq:
    st.header(t['phq9'])
    phq_qs = TEXTS[lang]['phq9_questions']
    phq_opts = TEXTS[lang]['phq9_options']
    phq_answers = []
    for i,q in enumerate(phq_qs,1):
        label = reshape_arabic(q) if lang=='ar' else q
        ans = st.selectbox(label, phq_opts, key=f"phq{i}")
        try:
            phq_answers.append(int(ans.split('=')[0].strip()))
        except Exception:
            phq_answers.append(phq_opts.index(ans))
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
    st.write(f"PHQ-9 Score: **{phq_score}** → {phq_risk}")

# AD8 tab
with tab_ad8:
    st.header(t['ad8'])
    ad8_qs = TEXTS[lang]['ad8_questions']
    ad8_opts = TEXTS[lang]['ad8_options']
    ad8_answers = []
    for i,q in enumerate(ad8_qs,1):
        label = reshape_arabic(q) if lang=='ar' else q
        ans = st.selectbox(label, ad8_opts, key=f"ad8{i}")
        ad8_answers.append(1 if ans==ad8_opts[1] else 0)
    ad8_score = sum(ad8_answers)
    ad8_risk = "Low" if ad8_score < 2 else "Possible concern"
    st.write(f"AD8 Score: **{ad8_score}** → {ad8_risk}")

# Report tab
with tab_report:
    st.header(t['report'])
    if st.button("Generate"):
        EEG_results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        EEG_results['Depression'] = {'score': phq_score, 'risk': phq_risk}
        EEG_results['Alzheimer'] = {'score': ad8_score, 'risk': ad8_risk}
        interpretations = []
        risk_scores = {}
        for fname, block in EEG_results['EEG_files'].items():
            qi = block.get('QEEG', {})
            conn = block.get('connectivity', {})
            if qi.get('alpha_asym_F3_F4', None) is not None:
                a = qi['alpha_asym_F3_F4']
                if a > 0.2:
                    interpretations.append(f"{fname}: Left frontal alpha > right (F3>F4) — pattern reported in depression studies.")
                elif a < -0.2:
                    interpretations.append(f"{fname}: Right frontal alpha > left (F4>F3).")
            ta = qi.get('Theta_Alpha_ratio')
            if ta and ta > 1.2:
                interpretations.append(f"{fname}: Elevated Theta/Alpha ratio ({fmt(ta)}). Consider cognitive follow-up.")
            conn_summary = {'mean_connectivity': conn.get('mean_connectivity')} if conn and 'mean_connectivity' in conn else {}
            ctxt = compute_contextualized_risk(qi, conn_summary, age=patient_info.get('age'), sex=patient_info.get('gender'))
            risk_scores[fname] = ctxt['risk_percent']
            interpretations.append(f"{fname}: Contextualized risk ~ {ctxt['risk_percent']:.1f}% (percentile {ctxt['percentile_vs_norm']:.1f}).")
        json_bytes = io.BytesIO(json.dumps(EEG_results, indent=2, ensure_ascii=False).encode())
        st.download_button(t['download_json'], json_bytes, file_name='report.json')
        if EEG_results['EEG_files']:
            rows=[]
            for fname, b in EEG_results['EEG_files'].items():
                row = {'file': fname}
                for k,v in b.get('QEEG', {}).items(): row[k]=v
                row['risk_score'] = risk_scores.get(fname, '')
                rows.append(row)
            df = pd.DataFrame(rows)
            st.download_button(t['download_csv'], df.to_csv(index=False).encode('utf-8'), file_name='qeeg_features.csv', mime='text/csv')
        try:
            pdfb = build_pdf(EEG_results, patient_info, lab_results, meds_list, lang=lang, band_pngs=band_pngs, conn_images=conn_imgs, interpretations=interpretations, risk_scores=risk_scores)
            st.download_button(t['download_pdf'], pdfb, file_name='report.pdf')
            st.success("Report generated — downloads ready.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    st.markdown("---")
    st.info(t['note'])

# Footer
with st.expander("Installation & Notes"):
    st.write("Place requirements.txt next to app.py and redeploy or install locally:")
    st.code("pip install -r requirements.txt")
    st.write("If compute_connectivity is slow: avoid enabling it on very long recordings. ICA requires scikit-learn. ML model is preliminary and trained on synthetic data — calibrate with local labeled data before clinical use.")

# End of app.py
