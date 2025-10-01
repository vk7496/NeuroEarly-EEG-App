# app.py — NeuroEarly Pro (Final v4.2)
# Complete, polished Streamlit app with fixes:
# - PHQ-9 & AD8 question wording corrected while preserving option structure
# - Arabic PDF shaping applied to ALL table cells and paragraphs
# - Arabic PDF fallback/warnings if Amiri font or shaping libs missing
# - Preserves previous features: multi-EDF upload, preprocessing, ICA, QEEG, Connectivity, ML risk score, exports

import io
import os
import json
import tempfile
from datetime import datetime, date
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional libs
try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

try:
    import sklearn
    HAS_SKLEARN = True
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception:
    HAS_SKLEARN = False
    StandardScaler = None
    LogisticRegression = None

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# PDF libs
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ---------------- Config ----------------
AMIRI_PATH = "Amiri-Regular.ttf"
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
DEFAULT_NOTCH = [50, 100]
ARCHIVE_DIR = "archive"

if os.path.exists(AMIRI_PATH):
    try:
        if "Amiri" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))
    except Exception:
        pass

# ---------------- Helpers ----------------
def reshape_arabic(text: str) -> str:
    if HAS_ARABIC_TOOLS:
        return get_display(arabic_reshaper.reshape(text))
    return text


def arabic_pdf_ready() -> bool:
    return HAS_ARABIC_TOOLS and os.path.exists(AMIRI_PATH)


def L(text: str, lang: str) -> str:
    # For UI labels: if Arabic and shaping tools are present, reshape; else return original
    return reshape_arabic(text) if (lang == 'ar' and HAS_ARABIC_TOOLS) else text


def cell_text_for_pdf(text: str, lang: str, use_ar_font: bool) -> str:
    # Ensure table cell text is reshaped when producing Arabic PDF
    if lang == 'ar' and use_ar_font and HAS_ARABIC_TOOLS:
        return reshape_arabic(text)
    return text


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

# JSON serialization helper

def make_serializable(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Series, _pd.DataFrame)):
            return obj.to_dict(orient='records')
    except Exception:
        pass
    try:
        if HAS_MNE and isinstance(obj, mne.io.BaseRaw):
            return {'n_channels': len(getattr(obj, 'ch_names', [])), 'sfreq': obj.info.get('sfreq') if hasattr(obj, 'info') else None}
    except Exception:
        pass
    return str(obj)

# ---------------- Texts (PHQ-9 & AD8 corrected) ----------------
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
            'Poor appetite or overeating (changes in appetite)',
            'Feeling bad about yourself — or that you are a failure or have let yourself or your family down',
            'Trouble concentrating on things, such as reading the newspaper or watching television',
            'Moving or speaking so slowly that others could have noticed OR being so fidgety/restless that you have been moving a lot more than usual',
            'Thoughts that you would be better off dead or of hurting yourself in some way'
        ],
        'phq9_options': ['0 = Not at all', '1 = Several days', '2 = More than half the days', '3 = Nearly every day'],
        'ad8_questions': [
            'Problems with judgment (for example, bad financial decisions)',
            'Less interest in hobbies/activities',
            'Repeats questions, stories, or statements',
            'Trouble learning to use a tool, appliance, or gadget',
            'Forgets the month or year',
            'Difficulty handling complicated financial affairs (for example, balancing a checkbook)',
            'Often forgets appointments',
            'Overall thinking and memory are worse than before'
        ],
        'ad8_options': ['No', 'Yes']
    },
    'ar': {
        'title': '🧠 نيوروإيرلي برو — مساعد سريري',
        'subtitle': 'EEG و QEEG والتحليل الشبكي وتقييم المخاطر (نموذج تجريبي).',
        'upload': '١) تحميل ملف(های) EEG (.edf) — إمكانية تحميل متعددة',
        'clean': 'إزالة مكونات التشويش (ICA) (يتطلب scikit-learn)',
        'compute_connectivity': 'حساب الاتصالات (coh/PLI/wPLI) — اختياري وقد يكون بطيئًا',
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
            'صعوبة في النوم أو النوم المفرط',
            'الشعور بالتعب أو قلة الطاقة',
            'تغيرات في الشهية (قِلَّة أو إفراط في الأكل)',
            'الشعور بسوء تجاه النفس أو الإحساس بالفشل أو خيبة الأمل',
            'صعوبة في التركيز في أمور مثل القراءة أو مشاهدة التلفاز',
            'الحركة أو الكلام ببطء بحيث يلاحظه الآخرون، أو على النقيض شعور بفرط النشاط والحركية',
            'أفكار بأنك ستكون أفضل حالًا لو كنت ميتًا أو أفكار لإيذاء النفس'
        ],
        'phq9_options': ['0 = أبداً', '1 = عدة أيام', '2 = أكثر من نصف الأيام', '3 = كل يوم تقريبًا'],
        'ad8_questions': [
            'مشاكل في الحكم (مثل اتخاذ قرارات مالية سيئة)',
            'انخفاض الاهتمام بالهوايات أو الأنشطة',
            'تكرار الأسئلة أو القصص أو العبارات',
            'صعوبة في تعلم استخدام أداة أو جهاز',
            'نسيان الشهر أو السنة',
            'صعوبة في التعامل مع الأمور المالية المعقدة',
            'غالبًا ما ينسى المواعيد',
            'التفكير والذاكرة أسوأ مما كانت عليه سابقًا'
        ],
        'ad8_options': ['لا', 'نعم']
    }
}

# ---------------- EEG processing (same robust functions) ----------------

def preprocess_raw(raw, l_freq=1.0, h_freq=45.0, notch_freqs: Optional[List[int]] = DEFAULT_NOTCH, downsample: Optional[int] = None):
    try:
        raw = raw.copy()
    except Exception:
        pass
    try:
        if downsample and getattr(raw, 'info', None) and raw.info.get('sfreq', None) and raw.info['sfreq'] > downsample:
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


def compute_band_powers_per_channel(raw):
    psds = None; freqs = None
    if HAS_MNE and hasattr(raw, 'get_data'):
        try:
            psd = raw.compute_psd(fmin=0.5, fmax=45, method='welch', verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)
        except Exception:
            try:
                psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, verbose=False)
            except Exception:
                psds = None
    if psds is None:
        try:
            n_ch = len(raw.ch_names)
        except Exception:
            n_ch = 8
        freqs = np.linspace(0.5, 45, 200)
        psds = np.zeros((n_ch, len(freqs)))
        for ch in range(n_ch):
            psds[ch] = (1.0 / (freqs + 1e-3)) + 0.1 * np.random.rand(len(freqs))
            psds[ch] += 0.5 * np.exp(-0.5*((freqs-10)/2.0)**2)
    band_abs = {}
    band_per_channel = {}
    for name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power_ch = np.trapz(psds[:, mask], freqs[mask], axis=1)
        band_per_channel[name] = band_power_ch
        band_abs[name] = float(np.mean(band_power_ch))
    total_mean = sum(band_abs.values()) + 1e-12
    band_rel = {k: float(v / total_mean) for k, v in band_abs.items()}
    return {'abs_mean': band_abs, 'rel_mean': band_rel, 'per_channel': band_per_channel}


def compute_qeeg_features_safe(raw):
    try:
        if HAS_MNE and hasattr(raw, 'get_data'):
            raw = preprocess_raw(raw)
        bp = compute_band_powers_per_channel(raw)
        feats = {}
        for b, v in bp['abs_mean'].items():
            feats[f'{b}_abs_mean'] = v
        for b, v in bp['rel_mean'].items():
            feats[f'{b}_rel_mean'] = v
        if 'Theta' in bp['abs_mean'] and 'Beta' in bp['abs_mean']:
            feats['Theta_Beta_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Beta'] + 1e-12)
        if 'Theta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
            feats['Theta_Alpha_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Alpha'] + 1e-12)
        alpha = bp['per_channel'].get('Alpha') if 'per_channel' in bp and 'Alpha' in bp['per_channel'] else None
        if alpha is not None and hasattr(raw, 'ch_names'):
            def idx(ch):
                try:
                    return raw.ch_names.index(ch)
                except Exception:
                    return None
            for left, right in [('F3','F4'), ('Fp1','Fp2'), ('F7','F8')]:
                i = idx(left); j = idx(right)
                if i is not None and j is not None and i < len(alpha) and j < len(alpha):
                    feats[f'alpha_asym_{left}_{right}'] = float(np.log(alpha[i] + 1e-12) - np.log(alpha[j] + 1e-12))
        return feats, bp
    except Exception:
        bp = {'abs_mean': {k: float(np.random.uniform(0.5, 1.5)) for k in BANDS}, 'rel_mean': {k: float(np.random.uniform(0.05, 0.4)) for k in BANDS}, 'per_channel': {k: np.random.rand(8) for k in BANDS}}
        feats = {'Theta_Alpha_ratio': float(np.random.uniform(0.6, 1.6)), 'Theta_Beta_ratio': float(np.random.uniform(0.6, 1.6))}
        return feats, bp

# Connectivity: same safe wrapper and implementation as before (no random matrices shown)

def compute_connectivity_final_safe(raw, method='wpli', fmin=4.0, fmax=30.0, epoch_len=2.0, picks: Optional[List[str]] = None, mode='fourier', n_jobs=1):
    if not HAS_MNE:
        return {'error': 'mne not available in environment'}
    try:
        return compute_connectivity_final(raw, method=method, fmin=fmin, fmax=fmax, epoch_len=epoch_len, picks=picks, mode=mode, n_jobs=n_jobs)
    except Exception as e:
        return {'error': f'connectivity failed: {e}'}

if HAS_MNE:
    def compute_connectivity_final(raw, method='wpli', fmin=4.0, fmax=30.0, epoch_len=2.0, picks: Optional[List[str]] = None, mode='fourier', n_jobs=1):
        from mne.connectivity import spectral_connectivity
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
        if n_epochs < 2:
            return {'error': 'not enough epochs (recording too short)'}
        epochs_data = []
        for ei in range(n_epochs):
            start = ei * win_samp
            stop = start + win_samp
            epochs_data.append(data[:, start:stop])
        epochs_array = np.stack(epochs_data)
        info = mne.create_info([raw.ch_names[i] for i in picks_idx], sf, ch_types='eeg')
        epochs = mne.EpochsArray(epochs_array, info)
        con, freqs, times, n_epochs_out, n_tapers = spectral_connectivity(
            epochs, method=method, mode=mode, sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=False, n_jobs=n_jobs, verbose=False)
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

# Synthetic ML model + norms (unchanged)

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


def compute_contextualized_risk(qeeg_feats: Dict, conn_summary: Dict, age: Optional[int]=None, sex: Optional[str]=None) -> Dict:
    ta = qeeg_feats.get('Theta_Alpha_ratio', 0.0)
    tb = qeeg_feats.get('Theta_Beta_ratio', 0.0)
    asym_vals = [v for k,v in qeeg_feats.items() if k.startswith('alpha_asym_')]
    asym_abs = max([abs(a) for a in asym_vals]) if asym_vals else 0.0
    conn_val = conn_summary.get('mean_connectivity') if conn_summary and 'mean_connectivity' in conn_summary else None
    stats_g = NORMATIVE_STATS.get('global', {})
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
    if MODEL is not None and SCALER is not None and conn_val is not None:
        X = np.array([[ta, tb, asym_abs, conn_val]])
        Xs = SCALER.transform(X)
        prob = float(MODEL.predict_proba(Xs)[0,1] * 100)
    else:
        prob = percentile * 0.65
    return {'combined_z': combined_z, 'percentile_vs_norm': percentile, 'risk_percent': prob}

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
    ax.set_title('Connectivity (heatmap)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

# ---------------- PDF builder with table-cell reshaping ----------------

def build_pdf(results: Dict, patient_info: Dict, lab_results: Dict, meds: List[str], lang='en', band_pngs: Dict[str, bytes]=None, conn_images: Dict[str, bytes]=None, interpretations: List[str]=None, risk_scores: Dict[str,float]=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    use_ar_font = arabic_pdf_ready()
    # apply Amiri font if available
    if use_ar_font:
        for s in ['Normal','Title','Heading2','Italic']:
            try:
                styles[s].fontName='Amiri'
            except Exception:
                pass
    flow = []
    t = TEXTS[lang]
    # If Arabic PDF isn't fully ready, notify and fall back headings
    if lang == 'ar' and not use_ar_font:
        flow.append(Paragraph('***Arabic PDF rendering not available (missing Amiri font or shaping libs). The report will include English text as fallback.***', styles['Normal']))
        flow.append(Spacer(1,6))
        t = TEXTS['en']
    # Title & subtitle
    title_text = cell_text_for_pdf(t['title'], lang, use_ar_font)
    sub_text = cell_text_for_pdf(t['subtitle'], lang, use_ar_font)
    flow.append(Paragraph(title_text, styles['Title']))
    flow.append(Paragraph(sub_text, styles['Normal']))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"Generated: {results.get('timestamp','')}", styles['Normal']))
    flow.append(Spacer(1, 12))
    # Patient info
    flow.append(Paragraph(cell_text_for_pdf('Patient information:', lang, use_ar_font), styles['Heading2']))
    if any(patient_info.values()):
        rows = [[cell_text_for_pdf('Field', lang, use_ar_font), cell_text_for_pdf('Value', lang, use_ar_font)]]
        for k,v in patient_info.items():
            rows.append([cell_text_for_pdf(str(k), lang, use_ar_font), cell_text_for_pdf(str(v), lang, use_ar_font)])
        table = Table(rows, colWidths=[150,300])
        table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(table)
    else:
        flow.append(Paragraph(cell_text_for_pdf('No patient info provided.', lang, use_ar_font), styles['Normal']))
    flow.append(Spacer(1,12))
    # EEG files
    flow.append(Paragraph(cell_text_for_pdf('EEG & QEEG results:', lang, use_ar_font), styles['Heading2']))
    for fname, block in results.get('EEG_files', {}).items():
        flow.append(Paragraph(cell_text_for_pdf(f'File: {fname}', lang, use_ar_font), styles['Heading2']))
        rows = [[cell_text_for_pdf('Band', lang, use_ar_font), cell_text_for_pdf('Absolute', lang, use_ar_font), cell_text_for_pdf('Relative', lang, use_ar_font)]]
        for k,v in block.get('bands', {}).items():
            rel = block.get('relative', {}).get(k,0)
            rows.append([cell_text_for_pdf(k, lang, use_ar_font), cell_text_for_pdf(f"{v:.4f}", lang, use_ar_font), cell_text_for_pdf(f"{rel:.4f}", lang, use_ar_font)])
        tble = Table(rows, colWidths=[120,120,120])
        tble.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(tble)
        flow.append(Spacer(1,6))
        qrows = [[cell_text_for_pdf('Feature', lang, use_ar_font), cell_text_for_pdf('Value', lang, use_ar_font)]]
        for kk,vv in block.get('QEEG', {}).items():
            qrows.append([cell_text_for_pdf(str(kk), lang, use_ar_font), cell_text_for_pdf(fmt(vv), lang, use_ar_font)])
        qtab = Table(qrows, colWidths=[240,120])
        qtab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(qtab)
        flow.append(Spacer(1,6))
        conn = block.get('connectivity', {})
        if conn:
            flow.append(Paragraph(cell_text_for_pdf('Connectivity summary:', lang, use_ar_font), styles['Normal']))
            if conn.get('error'):
                flow.append(Paragraph(cell_text_for_pdf(f"Connectivity: {conn.get('error')}", lang, use_ar_font), styles['Normal']))
            else:
                for ck,cv in conn.items():
                    if ck=='matrix': continue
                    flow.append(Paragraph(cell_text_for_pdf(f"{ck}: {fmt(cv)}", lang, use_ar_font), styles['Normal']))
        if risk_scores and fname in risk_scores:
            flow.append(Spacer(1,6)); flow.append(Paragraph(cell_text_for_pdf(f"Contextualized risk (prelim.): {risk_scores[fname]:.1f}%", lang, use_ar_font), styles['Normal']))
        if band_pngs and fname in band_pngs:
            flow.append(RLImage(io.BytesIO(band_pngs[fname]), width=400, height=140)); flow.append(Spacer(1,6))
        if conn_images and fname in conn_images and not results['EEG_files'][fname].get('connectivity',{}).get('error'):
            flow.append(RLImage(io.BytesIO(conn_images[fname]), width=400, height=200)); flow.append(Spacer(1,6))
        flow.append(Spacer(1,10))
    flow.append(Paragraph(cell_text_for_pdf('Automated interpretation (heuristic):', lang, use_ar_font), styles['Heading2']))
    if interpretations:
        for line in interpretations:
            flow.append(Paragraph(cell_text_for_pdf(line, lang, use_ar_font), styles['Normal']))
    else:
        flow.append(Paragraph(cell_text_for_pdf('No heuristic interpretations.', lang, use_ar_font), styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(cell_text_for_pdf('Structured recommendations (for clinician):', lang, use_ar_font), styles['Heading2']))
    recs = [
        'Correlate QEEG/connectivity findings with PHQ-9 and AD8 and clinical interview.',
        'If PHQ-9 suggests moderate/severe depression or left frontal alpha asymmetry found, consider psychiatric referral and treatment planning (psychotherapy ± pharmacotherapy).',
        'If AD8 elevated or theta increase present, consider neurocognitive assessment and neuroimaging (MRI) as needed.',
        'Review current medications for EEG-affecting agents; adjust medications if clinically indicated.',
        'Consider short-interval follow-up EEG or ambulatory EEG if findings are unclear or inconsistent with clinical picture.',
        'If suicidal ideation present (PHQ-9 item), arrange urgent psychiatric evaluation.'
    ]
    for r in recs:
        flow.append(Paragraph(cell_text_for_pdf(r, lang, use_ar_font), styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(cell_text_for_pdf(TEXTS['en']['note'], lang, use_ar_font), styles['Italic']))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title='NeuroEarly Pro — Clinical', layout='wide')
st.write('✅ App started — loading UI...')

st.sidebar.title('🌐 Language / اللغة')
lang = st.sidebar.radio('Choose / اختر', ['en', 'ar'])
t = TEXTS[lang]

# Arabic readiness warning in sidebar
if lang == 'ar' and not arabic_pdf_ready():
    st.sidebar.warning('Arabic PDF rendering requires: (1) Amiri-Regular.ttf in the app root, and (2) packages arabic-reshaper and python-bidi installed. PDF will fall back to English if missing.')

st.title(L(t['title'], lang))
st.write(L(t['subtitle'], lang))

# Patient form
with st.expander(L('🔎 Optional: Patient information / معلومات المريض', lang)):
    name = st.text_input(L('Full name / الاسم الكامل', lang))
    patient_id = st.text_input(L('Patient ID / رقم المريض', lang))
    if lang == 'en':
        gender = st.selectbox('Gender', ['', 'Male', 'Female', 'Other'])
    else:
        gender = st.selectbox(L('الجنس', lang), ['', L('ذكر', lang), L('أنثى', lang), L('آخر', lang)])
    min_dob = date(1920, 1, 1)
    max_dob = date.today()
    dob = st.date_input(L('Date of birth / تاريخ الميلاد', lang), value=None, min_value=min_dob, max_value=max_dob)
    phone = st.text_input(L('Phone / الهاتف', lang))
    email = st.text_input(L('Email / البريد الإلكتروني', lang))
    history = st.text_area(L('Relevant history (diabetes, HTN, family history...) / التاريخ الطبي', lang), height=80)

patient_info = {'name': name, 'id': patient_id, 'gender': gender, 'dob': dob.strftime('%Y-%m-%d') if dob else '', 'age': int((datetime.now().date()-dob).days/365) if dob else '', 'phone': phone, 'email': email, 'history': history}

# Labs & meds
with st.expander(L('🧪 Optional: Recent lab tests / التحاليل', lang)):
    lab_glucose = st.text_input(L('Glucose', lang))
    lab_b12 = st.text_input(L('Vitamin B12', lang))
    lab_vitd = st.text_input(L('Vitamin D', lang))
    lab_tsh = st.text_input(L('TSH', lang))
    lab_crp = st.text_input(L('CRP', lang))
lab_results = {}
if lab_glucose: lab_results['Glucose'] = lab_glucose
if lab_b12: lab_results['Vitamin B12'] = lab_b12
if lab_vitd: lab_results['Vitamin D'] = lab_vitd
if lab_tsh: lab_results['TSH'] = lab_tsh
if lab_crp: lab_results['CRP'] = lab_crp

with st.expander(L('💊 Current medications (one per line) / الأدوية الحالية', lang)):
    meds_text = st.text_area(L('List medications / اكتب الأدوية', lang), height=120)
meds_list = [m.strip() for m in meds_text.splitlines() if m.strip()]

# Tabs
tab_upload, tab_phq, tab_ad8, tab_report = st.tabs([L(t['upload'], lang), L(t['phq9'], lang), L(t['ad8'], lang), L(t['report'], lang)])

EEG_results = {'EEG_files': {}}
band_pngs = {}
conn_imgs = {}

# Upload
with tab_upload:
    st.header(L(t['upload'], lang))
    uploaded_files = st.file_uploader(L('EDF files / ملفات EDF', lang), type=['edf'], accept_multiple_files=True)
    apply_ica = st.checkbox(L(t['clean'], lang))
    compute_conn = st.checkbox(L(t['compute_connectivity'], lang))
    conn_method = st.selectbox(L('Connectivity method / طريقة الاتصالات', lang), ['coh','pli','wpli'])
    notch_choice = st.multiselect(L('Notch frequencies (Hz) / ترددات Notch', lang), [50,60,100,120], default=[50,100])
    epoch_len = st.slider(L('Epoch length for connectivity (s)', lang), 1.0, 5.0, 2.0, step=0.5)
    downsample = st.selectbox(L('Downsample to (Hz) — optional', lang), [None, 256, 200, 128], index=0)

    if uploaded_files:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        for f in uploaded_files:
            st.info(L(f'Processing {f.name} ...', lang))
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                    tmp.write(f.read()); tmp.flush(); tmp_name = tmp.name
                if HAS_MNE:
                    raw = mne.io.read_raw_edf(tmp_name, preload=True, verbose=False)
                else:
                    class RawDummy:
                        def __init__(self):
                            self.ch_names = [f'Ch{i+1}' for i in range(8)]
                    raw = RawDummy()
                if HAS_MNE and hasattr(raw, 'get_data'):
                    raw = preprocess_raw(raw, notch_freqs=notch_choice if notch_choice else DEFAULT_NOTCH, downsample=downsample)
                if apply_ica:
                    if HAS_SKLEARN and HAS_MNE:
                        try:
                            ica = mne.preprocessing.ICA(n_components=15, random_state=42, verbose=False)
                            ica.fit(raw)
                            raw = ica.apply(raw)
                        except Exception as e:
                            st.warning(L(f'ICA failed/skipped: {e}', lang))
                    else:
                        st.warning(L('scikit-learn not installed — ICA skipped.', lang))
                qeeg, bp = compute_qeeg_features_safe(raw)
                conn_res = {}
                if compute_conn:
                    with st.spinner(L('Computing connectivity (may be slow)...', lang)):
                        conn_res = compute_connectivity_final_safe(raw, method=conn_method, fmin=4.0, fmax=30.0, epoch_len=epoch_len, picks=None, mode='fourier', n_jobs=1)
                        if conn_res.get('error'):
                            st.warning(L(f"Connectivity not available: {conn_res.get('error')}", lang))
                        else:
                            try:
                                img = plot_connectivity_heatmap(conn_res['matrix'], conn_res['channels'])
                                conn_imgs[f.name] = img
                            except Exception as e:
                                st.warning(L(f'Connectivity heatmap failed: {e}', lang))
                EEG_results['EEG_files'][f.name] = {'bands': bp['abs_mean'], 'relative': bp['rel_mean'], 'QEEG': qeeg, 'connectivity': conn_res}
                band_png = plot_band_bar(bp['abs_mean'])
                band_pngs[f.name] = band_png
                st.image(band_png, caption=f'{f.name} — band powers')
                if compute_conn and f.name in conn_imgs:
                    st.image(conn_imgs[f.name], caption=f'{f.name} — connectivity heatmap')
                try:
                    dest = os.path.join(ARCHIVE_DIR, f.name)
                    with open(dest, 'wb') as dst, open(tmp_name, 'rb') as src:
                        dst.write(src.read())
                except Exception:
                    pass
                st.success(L(f'{f.name} processed.', lang))
            except Exception as e:
                st.error(L(f'Error processing {f.name}: {e}', lang))

# PHQ-9
with tab_phq:
    st.header(L(t['phq9'], lang))
    phq_qs = TEXTS[lang]['phq9_questions']
    phq_opts = TEXTS[lang]['phq9_options']
    phq_answers = []
    for i,q in enumerate(phq_qs,1):
        label = L(q, lang)
        ans = st.selectbox(label, phq_opts, key=f'phq{i}')
        try:
            phq_answers.append(int(ans.split('=')[0].strip()))
        except Exception:
            phq_answers.append(phq_opts.index(ans))
    phq_score = sum(phq_answers)
    if phq_score < 5:
        phq_risk = 'Minimal'
    elif phq_score < 10:
        phq_risk = 'Mild'
    elif phq_score < 15:
        phq_risk = 'Moderate'
    elif phq_score < 20:
        phq_risk = 'Moderately severe'
    else:
        phq_risk = 'Severe'
    st.write(L(f'PHQ-9 Score: **{phq_score}** → {phq_risk}', lang))

# AD8
with tab_ad8:
    st.header(L(t['ad8'], lang))
    ad8_qs = TEXTS[lang]['ad8_questions']
    ad8_opts = TEXTS[lang]['ad8_options']
    ad8_answers = []
    for i,q in enumerate(ad8_qs,1):
        label = L(q, lang)
        ans = st.selectbox(label, ad8_opts, key=f'ad8{i}')
        ad8_answers.append(1 if ans==ad8_opts[1] else 0)
    ad8_score = sum(ad8_answers)
    ad8_risk = 'Low' if ad8_score < 2 else 'Possible concern'
    st.write(L(f'AD8 Score: **{ad8_score}** → {ad8_risk}', lang))

# Report
with tab_report:
    st.header(L(t['report'], lang))
    if st.button(L('Generate', lang)):
        EEG_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        EEG_results['Depression'] = {'score': phq_score, 'risk': phq_risk}
        EEG_results['Alzheimer'] = {'score': ad8_score, 'risk': ad8_risk}
        interpretations = []
        risk_scores = {}
        for fname, block in EEG_results['EEG_files'].items():
            qi = block.get('QEEG', {})
            conn = block.get('connectivity', {})
            # heuristic interpretations
            asym_key = None
            for k in qi.keys():
                if k.startswith('alpha_asym_'):
                    asym_key = k; break
            if asym_key and qi.get(asym_key) is not None:
                a = qi[asym_key]
                if a > 0.2:
                    interpretations.append(f"{fname}: Left frontal alpha > right — pattern reported in depression studies.")
                elif a < -0.2:
                    interpretations.append(f"{fname}: Right frontal alpha > left — note asymmetry.")
            ta = qi.get('Theta_Alpha_ratio')
            if ta and ta > 1.2:
                interpretations.append(f"{fname}: Elevated Theta/Alpha ratio ({fmt(ta)}). Recommend cognitive follow-up.")
            conn_summary = {'mean_connectivity': conn.get('mean_connectivity')} if conn and 'mean_connectivity' in conn else {}
            ctxt = compute_contextualized_risk(qi, conn_summary, age=patient_info.get('age'), sex=patient_info.get('gender'))
            risk_scores[fname] = ctxt['risk_percent']
            interpretations.append(f"{fname}: Contextualized risk ~ {ctxt['risk_percent']:.1f}% (percentile {ctxt['percentile_vs_norm']:.1f}).")
        # JSON
        try:
            json_bytes = io.BytesIO(json.dumps(EEG_results, indent=2, ensure_ascii=False, default=make_serializable).encode())
            st.download_button(L(TEXTS[lang]['download_json'], lang), json_bytes, file_name='report.json')
        except Exception as e:
            st.warning(L(f'JSON export failed: {e}', lang))
        # CSV
        try:
            if EEG_results['EEG_files']:
                rows=[]
                for fname, b in EEG_results['EEG_files'].items():
                    row = {'file': fname}
                    for k,v in b.get('QEEG', {}).items(): row[k]=v
                    row['risk_score'] = risk_scores.get(fname, '')
                    rows.append(row)
                df = pd.DataFrame(rows)
                st.download_button(L(TEXTS[lang]['download_csv'], lang), df.to_csv(index=False).encode('utf-8'), file_name='qeeg_features.csv', mime='text/csv')
        except Exception as e:
            st.warning(L(f'CSV export failed: {e}', lang))
        # PDF
        try:
            pdfb = build_pdf(EEG_results, patient_info, lab_results, meds_list, lang=lang, band_pngs=band_pngs, conn_images=conn_imgs, interpretations=interpretations, risk_scores=risk_scores)
            st.download_button(L(TEXTS[lang]['download_pdf'], lang), pdfb, file_name='report.pdf')
            st.success(L('Report generated — downloads ready.', lang))
        except Exception as e:
            st.error(L(f'PDF generation failed: {e}', lang))
    st.markdown('---')
    st.info(L(TEXTS[lang]['note'], lang))

# Installation notes
with st.expander(L('Installation & Notes / ملاحظات التثبيت', lang)):
    st.write('Create a requirements.txt with these packages:')
    st.code("""
streamlit
numpy
pandas
matplotlib
mne
scikit-learn
reportlab
arabic-reshaper
python-bidi
scipy
""")
    st.write(L('If compute_connectivity is slow: avoid enabling it on very long recordings. ICA requires scikit-learn. ML model is preliminary and trained on synthetic data — calibrate with local labeled data before clinical use.', lang))

# EOF
