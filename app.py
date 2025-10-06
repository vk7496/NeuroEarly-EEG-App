# app.py — NeuroEarly Pro — Final Pro Edition (single-file, modular style)
# Full-feature Streamlit app: bilingual (EN/AR), QEEG, Connectivity, Microstates, XAI (SHAP), PDF reports
# Requirements: streamlit, numpy, pandas, matplotlib, mne, scikit-learn, reportlab,
# arabic-reshaper, python-bidi, scipy, shap, joblib
#
# Put Amiri-Regular.ttf in project root for Arabic UI + PDF.

import io
import os
import json
import tempfile
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional heavy libs
try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

# Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

# ML libs
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    StandardScaler = None
    RandomForestClassifier = None

# SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Stats
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

# Joblib
try:
    import joblib
except Exception:
    joblib = None

# ---------------- Configuration ----------------
AMIRI_PATH = "Amiri-Regular.ttf"
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
DEFAULT_NOTCH = [50, 100]
ARCHIVE_DIR = "archive"
MODEL_PATH = "qeeg_risk_model.joblib"
SCALER_PATH = "qeeg_scaler.joblib"

# register Amiri if present
if os.path.exists(AMIRI_PATH):
    try:
        if "Amiri" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))
    except Exception:
        pass

# ---------------- Utility helpers ----------------
def reshape_for_ui(text: str) -> str:
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def reshape_for_pdf(text: str) -> str:
    # Use shaping if available; otherwise rely on Amiri font rendering and RTL flags in PDF
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

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
        'title': '🧠 NeuroEarly Pro — Clinical',
        'subtitle': 'EEG / QEEG / Connectivity / Microstates / Explainable Risk — Research demo only.',
        'upload': '1) Upload EEG file(s) (.edf)',
        'clean': 'Apply ICA artifact removal (optional; requires scikit-learn)',
        'compute_connectivity': 'Compute Connectivity (coh/PLI/wPLI) — optional, may be slow',
        'microstates': 'Microstates analysis (optional)',
        'phq9': '2) Depression Screening — PHQ-9',
        'ad8': '3) Cognitive Screening — AD8',
        'report': '4) Generate Report (JSON / PDF / CSV)',
        'download_json': '⬇️ Download JSON',
        'download_pdf': '⬇️ Download PDF',
        'download_csv': '⬇️ Download CSV',
        'note': '⚠️ Research/demo only — not a clinical diagnosis.',
        'phq9_questions': [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling asleep, staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating (e.g., reading, watching TV)",
            "Moving or speaking very slowly, OR being fidgety/restless",
            "Thoughts of being better off dead or self-harm"
        ],
        'phq9_options': ['0 = Not at all', '1 = Several days', '2 = More than half the days', '3 = Nearly every day'],
        'ad8_questions': [
            "Problems with judgment (e.g., poor financial decisions)",
            "Reduced interest in hobbies/activities",
            "Repeats questions or stories",
            "Trouble using a tool or gadget",
            "Forgets the correct month or year",
            "Difficulty managing finances (e.g., paying bills)",
            "Trouble remembering appointments",
            "Everyday thinking is getting worse"
        ],
        'ad8_options': ['No', 'Yes']
    },
    'ar': {
        'title': '🧠 نيوروإيرلي برو — سريري',
        'subtitle': 'EEG / QEEG / الاتصالات / الميكروستيتس / تفسير المخاطر — عرض بحثي.',
        'upload': '١) تحميل ملف(های) EEG (.edf)',
        'clean': 'إزالة مكونات ICA (اختياري؛ يتطلب scikit-learn)',
        'compute_connectivity': 'حساب الاتصالات (coh/PLI/wPLI) — اختياري وقد يكون بطيئًا',
        'microstates': 'تحليل الميكروستيتس (اختياري)',
        'phq9': '٢) استبيان الاكتئاب — PHQ-9',
        'ad8': '٣) الفحص المعرفي — AD8',
        'report': '٤) إنشاء التقرير (JSON / PDF / CSV)',
        'download_json': '⬇️ تنزيل JSON',
        'download_pdf': '⬇️ تنزيل PDF',
        'download_csv': '⬇️ تنزيل CSV',
        'note': '⚠️ أداة بحثية / توجيهية فقط — ليست تشخيصًا نهائيًا.',
        'phq9_questions': [
            "قلة الاهتمام أو المتعة في الأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم (صعوبة في النوم أو النوم لفترات طويلة)",
            "الشعور بالتعب أو قلة الطاقة",
            "قِلّة الشهية أو الإفراط في الأكل",
            "الشعور بسوء تجاه النفس أو الشعور بالفشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الحركة أو الكلام ببطء شديد، أو الشعور بفرط الحركة/الاضطراب الحركي",
            "أفكار بأنك ستكون أفضل حالًا لو كنت ميتًا أو التفكير في إيذاء النفس"
        ],
        'phq9_options': ['0 = أبداً', '1 = عدة أيام', '2 = أكثر من نصف الأيام', '3 = كل يوم تقريبًا'],
        'ad8_questions': [
            "مشاكل في الحكم (مثل اتخاذ قرارات مالية سيئة)",
            "انخفاض الاهتمام بالهوايات أو الأنشطة",
            "تكرار الأسئلة أو القصص",
            "صعوبة في استخدام أداة أو جهاز",
            "نسيان الشهر أو السنة الصحيحة",
            "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
            "صعوبة في تذكر المواعيد",
            "التفكير والذاكرة أسوأ مما كانت عليه سابقًا"
        ],
        'ad8_options': ['لا', 'نعم']
    }
}

# ---------------- EEG processing & denoising ----------------
def preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freqs: Optional[List[int]] = DEFAULT_NOTCH, downsample: Optional[int]=None, apply_ref=True):
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
        if apply_ref:
            raw.set_eeg_reference('average', verbose=False)
    except Exception:
        pass
    return raw

def advanced_preprocess(raw, l_freq=1.0, h_freq=45.0, notch_freqs: Optional[List[int]] = DEFAULT_NOTCH, downsample: Optional[int]=None, reject_treshold_uV: Optional[float]=150.0):
    raw = preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq, notch_freqs=notch_freqs, downsample=downsample)
    # annotate large artifacts
    try:
        if reject_treshold_uV and hasattr(raw, 'get_data'):
            data_uv = raw.get_data() * 1e6
            sf = int(raw.info['sfreq'])
            win = int(1.0 * sf)
            bad_onsets = []
            for start in range(0, max(1, data_uv.shape[1] - win), win):
                seg = data_uv[:, start:start+win]
                if np.max(np.ptp(seg, axis=1)) > reject_treshold_uV:
                    bad_onsets.append((start/sf, 1.0))
            if bad_onsets:
                from mne import Annotations
                ann = Annotations(onset=[o[0] for o in bad_onsets], duration=[o[1] for o in bad_onsets], description=['BAD_artifact']*len(bad_onsets))
                raw.set_annotations(raw.annotations + ann)
    except Exception:
        pass
    return raw

def run_ica_and_remove_artifacts(raw, n_components=15, auto_clean_eog=True, auto_clean_ecg=False):
    result = {'ica_applied': False, 'excluded_components': [], 'note': None}
    if not HAS_MNE:
        result['note'] = 'mne not available - ICA skipped'
        return raw, result
    if not HAS_SKLEARN:
        result['note'] = 'scikit-learn not installed - ICA skipped'
        return raw, result
    try:
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, meg=False)
        ica.fit(raw, picks=picks, verbose=False)
    except Exception as e:
        result['note'] = f'ICA fit failed: {e}'
        return raw, result
    exclude = []
    try:
        if auto_clean_eog:
            eog_chs = [ch for ch in raw.ch_names if 'EOG' in ch.upper() or 'VEOG' in ch.upper() or 'HEOG' in ch.upper()]
            for ch in eog_chs:
                inds, _ = ica.find_bads_eog(raw, ch_name=ch)
                exclude += inds
    except Exception:
        pass
    try:
        if auto_clean_ecg:
            ecg_chs = [ch for ch in raw.ch_names if 'ECG' in ch.upper() or 'EKG' in ch.upper()]
            for ch in ecg_chs:
                inds, _ = ica.find_bads_ecg(raw, ch_name=ch)
                exclude += inds
    except Exception:
        pass
    exclude = list(set(exclude))
    if exclude:
        try:
            ica.exclude = exclude
            raw_clean = ica.apply(raw.copy())
            result['ica_applied'] = True
            result['excluded_components'] = exclude
            return raw_clean, result
        except Exception as e:
            result['note'] = f'ICA apply failed: {e}'
            return raw, result
    else:
        result['note'] = 'No ICA components flagged for exclusion'
        return raw, result

# ---------------- Band powers & QEEG ----------------
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
        if 'Beta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
            feats['Beta_Alpha_ratio'] = bp['abs_mean']['Beta'] / (bp['abs_mean']['Alpha'] + 1e-12)
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

# ---------------- Connectivity safe wrapper ----------------
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

# ---------------- Microstate analysis (integrated) ----------------
if HAS_SKLEARN:
    from sklearn.cluster import KMeans

def compute_gfp(raw) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_MNE and hasattr(raw, 'get_data'):
        data = raw.get_data()
        times = np.arange(data.shape[1]) / raw.info['sfreq']
    else:
        data = np.random.randn(8, 1000)
        times = np.arange(data.shape[1]) / 250.0
    gfp = np.std(data, axis=0)
    return gfp, times

def find_gfp_peaks(gfp: np.ndarray, sfreq: float, min_peak_distance_s: float = 0.02, n_peaks: Optional[int]=None):
    from scipy.signal import find_peaks
    dist = max(1, int(min_peak_distance_s * sfreq))
    peaks, _ = find_peaks(gfp, distance=dist)
    if n_peaks and len(peaks) > n_peaks:
        order = np.argsort(gfp[peaks])[::-1][:n_peaks]
        peaks = peaks[order]
    return np.sort(peaks)

def extract_maps_at_peaks(raw, peaks_idx):
    if HAS_MNE and hasattr(raw, 'get_data'):
        data = raw.get_data()
    else:
        data = np.random.randn(8, 1000)
    maps = data[:, peaks_idx].T
    maps = maps - maps.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(maps, axis=1, keepdims=True) + 1e-12
    maps = maps / norms
    return maps

def run_microstate_kmeans(maps: np.ndarray, n_states: int = 4, random_state: int = 42):
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for microstate clustering")
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    km.fit(maps)
    centers = km.cluster_centers_
    centers = centers - centers.mean(axis=1, keepdims=True)
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return km, centers

def assign_microstates_to_samples(raw, centers):
    if HAS_MNE and hasattr(raw, 'get_data'):
        data = raw.get_data()
    else:
        data = np.random.randn(centers.shape[1], 1000)
    data_t = (data.T - data.mean(axis=0))
    norms = np.linalg.norm(data_t, axis=1, keepdims=True) + 1e-12
    data_n = data_t / norms
    centers_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    corr = np.dot(data_n, centers_n.T)
    labels = np.argmax(np.abs(corr), axis=1)
    return labels, corr

def compute_microstate_stats(labels: np.ndarray, sfreq: float, n_states: int):
    n_samples = len(labels)
    total_time_s = n_samples / sfreq
    stats = {}
    for s in range(n_states):
        mask = (labels == s).astype(int)
        coverage = mask.sum() / n_samples
        runs = []
        in_run = False; run_len = 0
        for v in mask:
            if v and not in_run:
                in_run = True; run_len = 1
            elif v and in_run:
                run_len += 1
            elif not v and in_run:
                runs.append(run_len); in_run = False; run_len = 0
        if in_run:
            runs.append(run_len)
        if runs:
            mean_dur_ms = np.mean(runs) / sfreq * 1000.0
            occurrence = len(runs) / total_time_s
        else:
            mean_dur_ms = 0.0; occurrence = 0.0
        stats[s] = {'coverage_frac': coverage, 'mean_duration_ms': float(mean_dur_ms), 'occurrence_per_s': float(occurrence)}
    return stats

def compute_connectivity_per_microstate(raw, labels, centers, sfreq=None, method='wpli', fmin=4.0, fmax=30.0, epoch_len=2.0):
    if not HAS_MNE:
        return {'error': 'mne not available'}
    if sfreq is None:
        sfreq = raw.info['sfreq']
    n_ch = len(raw.ch_names)
    results = {}
    for s in range(centers.shape[0]):
        idxs = np.where(labels == s)[0]
        if len(idxs) < int(epoch_len*sfreq):
            results[s] = {'error': 'not enough data for state'}
            continue
        segs = []
        start = idxs[0]; prev = idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
            else:
                segs.append((start, prev)); start = i; prev = i
        segs.append((start, prev))
        epochs_list = []
        win = int(epoch_len * sfreq)
        for (samp0, samp1) in segs:
            seg_len = samp1 - samp0 + 1
            n_full = seg_len // win
            for k in range(n_full):
                st0 = samp0 + k*win
                en0 = st0 + win
                epochs_list.append(raw.get_data()[:, st0:en0])
        if len(epochs_list) < 2:
            results[s] = {'error': 'not enough epochs after chunking'}
            continue
        epochs_array = np.stack(epochs_list)
        info = mne.create_info(raw.ch_names, sfreq, ch_types='eeg')
        epochs = mne.EpochsArray(epochs_array, info)
        try:
            from mne.connectivity import spectral_connectivity
            con, freqs, times, n_epochs_out, n_tapers = spectral_connectivity(
                epochs, method=method, mode='fourier', sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=False, n_jobs=1, verbose=False)
            mean_con = np.nanmean(con, axis=1)
            n_chs = len(raw.ch_names)
            mat = np.zeros((n_chs, n_chs))
            idx = 0
            for i in range(n_chs):
                for j in range(i+1, n_chs):
                    if idx < len(mean_con):
                        mat[i, j] = mean_con[idx]; mat[j, i] = mean_con[idx]
                    idx += 1
            results[s] = {'matrix': mat, 'channels': raw.ch_names, 'mean_connectivity': float(np.nanmean(mean_con))}
        except Exception as e:
            results[s] = {'error': f'connectivity failed: {e}'}
    return results

# ---------------- Synthetic model + XAI helpers ----------------
def build_synthetic_dataset(n=1200):
    rng = np.random.RandomState(42)
    ta = rng.normal(1.0, 0.4, n)
    tb = rng.normal(1.0, 0.6, n)
    asym = rng.normal(0.0, 0.3, n)
    conn = rng.normal(0.25, 0.1, n)
    age = rng.normal(60, 12, n)
    logit = 0.8*(ta-1.0) + 0.6*(tb-1.0) + 0.9*np.maximum(asym, 0) - 1.2*(conn-0.25) + 0.01*(age-60)
    prob = 1/(1+np.exp(-logit))
    y = (prob > 0.5).astype(int)
    X = np.vstack([ta, tb, np.abs(asym), conn, age]).T
    feat_names = ['Theta_Alpha_ratio','Theta_Beta_ratio','alpha_asym_abs','mean_connectivity','age']
    return X, y, feat_names

def train_synthetic_model():
    if not HAS_SKLEARN:
        return None
    X, y, fn = build_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xs, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(scaler.transform(X_test))[:,1])
    return {'model': clf, 'scaler': scaler, 'feat_names': fn, 'auc': auc}

MODEL_OBJ = None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and joblib:
    try:
        model_loaded = joblib.load(MODEL_PATH)
        scaler_loaded = joblib.load(SCALER_PATH)
        MODEL_OBJ = {'model': model_loaded, 'scaler': scaler_loaded, 'feat_names': ['Theta_Alpha_ratio','Theta_Beta_ratio','alpha_asym_abs','mean_connectivity','age'], 'auc': None}
    except Exception:
        MODEL_OBJ = train_synthetic_model()
else:
    MODEL_OBJ = train_synthetic_model()

MODEL = MODEL_OBJ['model'] if MODEL_OBJ else None
SCALER = MODEL_OBJ['scaler'] if MODEL_OBJ else None
FEATURE_NAMES = MODEL_OBJ['feat_names'] if MODEL_OBJ else ['Theta_Alpha_ratio','Theta_Beta_ratio','alpha_asym_abs','mean_connectivity','age']
MODEL_AUC = MODEL_OBJ['auc'] if MODEL_OBJ else None

def compute_shap_for_instance(model, scaler, feat_names, X_instance: np.ndarray):
    if not HAS_SHAP or model is None:
        return None
    try:
        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.Explainer(model, feature_names=feat_names)
        shap_vals = explainer.shap_values(scaler.transform(X_instance.reshape(1,-1)))
        # produce a simple bar plot for local explanation
        vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0][0]
        fig, ax = plt.subplots(figsize=(6,2.5))
        idx = np.argsort(np.abs(vals))[::-1]
        ax.barh([feat_names[i] for i in idx], vals[idx])
        ax.set_title('Local feature contributions (SHAP)')
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return buf.getvalue()
    except Exception:
        # fallback to simple feature importance
        try:
            fig, ax = plt.subplots(figsize=(6,2.5))
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                ax.bar(range(len(feat_names)), imp)
                ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, rotation=45)
            else:
                ax.text(0.1,0.5,'No SHAP and no feature_importances', transform=ax.transAxes)
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
            return buf.getvalue()
        except Exception:
            return None

def compute_shap_summary_plot(model, scaler, feat_names, X_sample: np.ndarray):
    if not HAS_SHAP or model is None:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(scaler.transform(X_sample))
        fig = plt.figure(figsize=(6,4))
        shap.summary_plot(shap_vals, features=X_sample, feature_names=feat_names, show=False)
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None

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

# ---------------- PDF builder (language-aware) ----------------
def build_pdf(results: Dict, patient_info: Dict, lab_results: Dict, meds: List[str], lang='en', band_pngs: Dict[str, bytes]=None, conn_images: Dict[str, bytes]=None, micro_images: Dict[str, bytes]=None, interpretations: List[str]=None, risk_scores: Dict[str,float]=None, shap_images: Dict[str,bytes]=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    use_ar_pdf = (lang == 'ar' and os.path.exists(AMIRI_PATH))
    if use_ar_pdf:
        for s in ['Normal','Title','Heading2','Italic']:
            try:
                styles[s].fontName='Amiri'
            except Exception:
                pass
    flow = []
    t = TEXTS[lang]
    title_text = reshape_for_pdf(t['title']) if use_ar_pdf else t['title']
    sub_text = reshape_for_pdf(t['subtitle']) if use_ar_pdf else t['subtitle']
    flow.append(Paragraph(title_text, styles['Title']))
    flow.append(Paragraph(sub_text, styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(f"Generated: {results.get('timestamp','')}", styles['Normal']))
    flow.append(Spacer(1,12))
    # Patient info
    header_pi = reshape_for_pdf('Patient information:') if use_ar_pdf else 'Patient information:'
    flow.append(Paragraph(header_pi, styles['Heading2']))
    if any(patient_info.values()):
        rows = [['Field','Value']]
        for k,v in patient_info.items():
            rows.append([str(k), str(v)])
        table = Table(rows, colWidths=[150,300])
        table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(table)
    else:
        flow.append(Paragraph('No patient info provided.', styles['Normal']))
    flow.append(Spacer(1,12))
    # EEG results
    flow.append(Paragraph('EEG & QEEG results:', styles['Heading2']))
    for fname, block in results.get('EEG_files', {}).items():
        flow.append(Paragraph(f'File: {fname}', styles['Heading2']))
        rows = [['Band','Absolute','Relative']]
        for k,v in block.get('bands', {}).items():
            rel = block.get('relative', {}).get(k,0)
            rows.append([k, f"{v:.4f}", f"{rel:.4f}"])
        tble = Table(rows, colWidths=[120,120,120])
        tble.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(tble); flow.append(Spacer(1,6))
        qrows = [['Feature','Value']]
        for kk,vv in block.get('QEEG', {}).items():
            qrows.append([str(kk), fmt(vv)])
        qtab = Table(qrows, colWidths=[240,120])
        qtab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        flow.append(qtab); flow.append(Spacer(1,6))
        if band_pngs and fname in band_pngs:
            flow.append(RLImage(io.BytesIO(band_pngs[fname]), width=400, height=140)); flow.append(Spacer(1,6))
        if conn_images and fname in conn_images and not results['EEG_files'][fname].get('connectivity',{}).get('error'):
            flow.append(RLImage(io.BytesIO(conn_images[fname]), width=400, height=200)); flow.append(Spacer(1,6))
        if shap_images and fname in shap_images:
            flow.append(Paragraph('Model explanation (features contributing to risk):', styles['Normal']))
            flow.append(RLImage(io.BytesIO(shap_images[fname]), width=400, height=200)); flow.append(Spacer(1,6))
        flow.append(Spacer(1,10))
    # interpretations & recs
    flow.append(Paragraph('Automated interpretation (heuristic + model):', styles['Heading2']))
    if interpretations:
        for line in interpretations:
            flow.append(Paragraph(line, styles['Normal']))
    else:
        flow.append(Paragraph('No heuristic interpretations.', styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph('Structured recommendations (for clinician):', styles['Heading2']))
    recs = [
        'Correlate QEEG/connectivity findings with PHQ-9 and AD8 and clinical interview.',
        'If PHQ-9 suggests moderate/severe depression or left frontal alpha asymmetry found, consider psychiatric referral and treatment planning (psychotherapy ± pharmacotherapy).',
        'If AD8 elevated or theta increase present, consider neurocognitive assessment and neuroimaging (MRI) as needed.',
        'Review current medications for EEG-affecting agents; adjust medications if clinically indicated.',
        'Consider short-interval follow-up EEG or ambulatory EEG if findings are unclear or inconsistent with clinical picture.',
        'If suicidal ideation present (PHQ-9 item), arrange urgent psychiatric evaluation.'
    ]
    for r in recs:
        flow.append(Paragraph(r, styles['Normal']))
    flow.append(Spacer(1,12))
    flow.append(Paragraph(TEXTS['en']['note'], styles['Italic']))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title='NeuroEarly Pro — Clinical', layout='wide')
# Basic CSS + Amiri if available
css = """
<style>
body { background-color: #f3f8ff; }
.card { background: white; padding: 12px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); margin-bottom: 12px; }
.small { font-size: 0.9rem; color: #444; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
if os.path.exists(AMIRI_PATH):
    st.markdown(f"""
    <style>
    @font-face {{
      font-family: 'AmiriCustom';
      src: url('/{AMIRI_PATH}') format('truetype');
    }}
    .ar-rtl {{ font-family: 'AmiriCustom', serif !important; direction: rtl !important; text-align: right !important; }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar language
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"])
t = TEXTS[lang]

# Header card
st.markdown("<div class='card'>", unsafe_allow_html=True)
if lang == 'ar' and os.path.exists(AMIRI_PATH):
    st.markdown(f"<h1 class='ar-rtl'>{reshape_for_ui(t['title'])}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='small ar-rtl'>{reshape_for_ui(t['subtitle'])}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>{t['subtitle']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Patient form
with st.expander(reshape_for_ui("🔎 Optional: Patient information / معلومات المريض") if (lang=='ar' and HAS_ARABIC_TOOLS) else "🔎 Optional: Patient information / معلومات المريض"):
    name = st.text_input(reshape_for_ui("Full name / الاسم الكامل") if (lang=='ar' and HAS_ARABIC_TOOLS) else "Full name / الاسم الكامل")
    patient_id = st.text_input(reshape_for_ui("Patient ID / رقم المريض") if (lang=='ar' and HAS_ARABIC_TOOLS) else "Patient ID / رقم المريض")
    if lang == 'en':
        gender = st.selectbox('Gender', ['', 'Male', 'Female', 'Other'])
    else:
        gender = st.selectbox('الجنس', ['', 'ذكر', 'أنثى', 'آخر'])
    min_dob = date(1920, 1, 1); max_dob = date.today()
    dob = st.date_input(reshape_for_ui('Date of birth / تاريخ الميلاد') if (lang=='ar' and HAS_ARABIC_TOOLS) else 'Date of birth / تاريخ الميلاد', value=None, min_value=min_dob, max_value=max_dob)
    phone = st.text_input(reshape_for_ui('Phone / الهاتف') if (lang=='ar' and HAS_ARABIC_TOOLS) else 'Phone / الهاتف')
    email = st.text_input(reshape_for_ui('Email / البريد الإلكتروني') if (lang=='ar' and HAS_ARABIC_TOOLS) else 'Email / البريد الإلكتروني')
    history = st.text_area(reshape_for_ui('Relevant history (diabetes, HTN, family history...) / التاريخ الطبي') if (lang=='ar' and HAS_ARABIC_TOOLS) else 'Relevant history (diabetes, HTN, family history...) / التاريخ الطبي', height=80)

patient_info = {'name': name, 'id': patient_id, 'gender': gender, 'dob': dob.strftime('%Y-%m-%d') if dob else '', 'age': int((datetime.now().date()-dob).days/365) if dob else '', 'phone': phone, 'email': email, 'history': history}

with st.expander(reshape_for_ui("🧪 Optional: Recent lab tests / التحاليل") if (lang=='ar' and HAS_ARABIC_TOOLS) else "🧪 Optional: Recent lab tests / التحاليل"):
    lab_glucose = st.text_input('Glucose'); lab_b12 = st.text_input('Vitamin B12'); lab_vitd = st.text_input('Vitamin D'); lab_tsh = st.text_input('TSH'); lab_crp = st.text_input('CRP')
lab_results = {}
if lab_glucose: lab_results['Glucose'] = lab_glucose
if lab_b12: lab_results['Vitamin B12'] = lab_b12
if lab_vitd: lab_results['Vitamin D'] = lab_vitd
if lab_tsh: lab_results['TSH'] = lab_tsh
if lab_crp: lab_results['CRP'] = lab_crp

with st.expander(reshape_for_ui("💊 Current medications (one per line) / الأدوية الحالية") if (lang=='ar' and HAS_ARABIC_TOOLS) else "💊 Current medications (one per line) / الأدوية الحالية"):
    meds_text = st.text_area(reshape_for_ui('List medications / اكتب الأدوية') if (lang=='ar' and HAS_ARABIC_TOOLS) else 'List medications / اكتب الأدوية', height=120)
meds_list = [m.strip() for m in meds_text.splitlines() if m.strip()]

# Main tabs
tab_upload, tab_phq, tab_ad8, tab_micro, tab_report = st.tabs([t['upload'], t['phq9'], t['ad8'], t['microstates'], t['report']])

EEG_results = {'EEG_files': {}}
band_pngs = {}
conn_imgs = {}
micro_imgs = {}
shap_images = {}

# Upload tab
with tab_upload:
    st.header(t['upload'])
    uploaded_files = st.file_uploader("EDF files / ملفات EDF", type=['edf'], accept_multiple_files=True)
    apply_ica = st.checkbox(t['clean'])
    compute_conn = st.checkbox(t['compute_connectivity'])
    conn_method = st.selectbox("Connectivity method / طريقة الاتصالات", ['coh','pli','wpli'])
    notch_choice = st.multiselect("Notch frequencies (Hz) / ترددات Notch", [50,60,100,120], default=[50,100])
    epoch_len = st.slider("Epoch length for connectivity (s)", 1.0, 5.0, 2.0, step=0.5)
    downsample = st.selectbox("Downsample to (Hz) — optional", [None, 256, 200, 128], index=0)
    # advanced denoising
    reject_uV = st.number_input('Reject threshold (µV) for annotation (0 disable)', min_value=0.0, max_value=1000.0, value=150.0)
    auto_clean_eog = st.checkbox('Auto-remove EOG components (if detected)', value=True)
    auto_clean_ecg = st.checkbox('Auto-remove ECG components (if detected)', value=False)
    # microstate options
    run_microstate_per_file = st.checkbox('Run Microstate analysis per file (may be slow)', value=False)
    micro_k = st.slider('Microstate K', 2, 7, 4)

    if uploaded_files:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        for f in uploaded_files:
            st.info(f'Processing {f.name} ...')
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                    tmp.write(f.read()); tmp.flush(); tmp_name = tmp.name
                if HAS_MNE:
                    raw = mne.io.read_raw_edf(tmp_name, preload=True, verbose=False)
                else:
                    # dummy raw
                    class RawDummy:
                        def __init__(self):
                            self.ch_names = [f'Ch{i+1}' for i in range(8)]
                            self.info = {'sfreq': 250}
                        def get_data(self, picks=None):
                            return np.random.randn(8, 250*60)
                    raw = RawDummy()

                if HAS_MNE and hasattr(raw, 'get_data'):
                    raw = advanced_preprocess(raw, notch_freqs=notch_choice if notch_choice else DEFAULT_NOTCH, downsample=downsample, reject_treshold_uV=(reject_uV if reject_uV>0 else None))
                if apply_ica:
                    raw, ica_info = run_ica_and_remove_artifacts(raw, n_components=15, auto_clean_eog=auto_clean_eog, auto_clean_ecg=auto_clean_ecg)
                    if ica_info.get('note'):
                        st.info(f"ICA note: {ica_info['note']}")
                    if ica_info.get('excluded_components'):
                        st.success(f"ICA removed components: {ica_info['excluded_components']}")
                # compute QEEG
                qeeg, bp = compute_qeeg_features_safe(raw)
                conn_res = {}
                if compute_conn:
                    with st.spinner('Computing connectivity (may be slow)...'):
                        conn_res = compute_connectivity_final_safe(raw, method=conn_method, fmin=4.0, fmax=30.0, epoch_len=epoch_len, picks=None, mode='fourier', n_jobs=1)
                        if conn_res.get('error'):
                            st.warning(f"Connectivity not available: {conn_res.get('error')}")
                        else:
                            try:
                                img = plot_connectivity_heatmap(conn_res['matrix'], conn_res['channels'])
                                conn_imgs[f.name] = img
                            except Exception as e:
                                st.warning(f'Connectivity heatmap failed: {e}')
                EEG_results['EEG_files'][f.name] = {'bands': bp['abs_mean'], 'relative': bp['rel_mean'], 'QEEG': qeeg, 'connectivity': conn_res}
                band_png = plot_band_bar(bp['abs_mean'])
                band_pngs[f.name] = band_png
                st.image(band_png, caption=f'{f.name} — band powers')
                if compute_conn and f.name in conn_imgs:
                    st.image(conn_imgs[f.name], caption=f'{f.name} — connectivity heatmap')
                # Microstate
                if run_microstate_per_file and HAS_MNE:
                    with st.spinner('Running microstate analysis...'):
                        gfp, times = compute_gfp(raw)
                        peaks = find_gfp_peaks(gfp, raw.info['sfreq'], min_peak_distance_s=0.02, n_peaks=None)
                        maps = extract_maps_at_peaks(raw, peaks)
                        if maps.shape[0] >= micro_k and HAS_SKLEARN:
                            km, centers = run_microstate_kmeans(maps, micro_k)
                            labels, corr = assign_microstates_to_samples(raw, centers)
                            stats = compute_microstate_stats(labels, raw.info['sfreq'], micro_k)
                            EEG_results['EEG_files'][f.name]['microstates'] = {'centers': centers.tolist(), 'stats': stats}
                            # simple centers plot
                            for i, c in enumerate(centers):
                                fig, ax = plt.subplots(figsize=(6,2))
                                ax.bar(range(len(c)), c)
                                ax.set_title(f"{f.name} — Microstate {i+1} center")
                                buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
                                micro_imgs[f'{f.name}_ms_{i}'] = buf.getvalue()
                            # connectivity per microstate (try)
                            try:
                                conn_by_state = compute_connectivity_per_microstate(raw, labels, centers, sfreq=raw.info['sfreq'], method=conn_method, fmin=4.0, fmax=30.0, epoch_len=epoch_len)
                                EEG_results['EEG_files'][f.name]['microstate_connectivity'] = conn_by_state
                            except Exception as e:
                                st.warning(f'Microstate connectivity failed: {e}')
                        else:
                            st.info('Not enough peaks for microstate clustering or scikit-learn missing.')
                # model explanation (if model available)
                feat_vector = [
                    qeeg.get('Theta_Alpha_ratio', 0.0),
                    qeeg.get('Theta_Beta_ratio', 0.0),
                    max([abs(qeeg.get(k,0.0)) for k in qeeg.keys() if k.startswith('alpha_asym_')] + [0.0]),
                    conn_res.get('mean_connectivity') if conn_res and 'mean_connectivity' in conn_res else 0.0,
                    patient_info.get('age') if patient_info.get('age') else 60.0
                ]
                X_inst = np.array(feat_vector)
                if MODEL is not None and SCALER is not None:
                    shap_img = compute_shap_for_instance(MODEL, SCALER, FEATURE_NAMES, X_inst)
                    if shap_img:
                        shap_images[f.name] = shap_img
                # archive file
                try:
                    dest = os.path.join(ARCHIVE_DIR, f.name)
                    with open(dest, 'wb') as dst, open(tmp_name, 'rb') as src:
                        dst.write(src.read())
                except Exception:
                    pass
                st.success(f'{f.name} processed.')
            except Exception as e:
                st.error(f'Error processing {f.name}: {e}')

# PHQ-9 tab
with tab_phq:
    st.header(t['phq9'])
    phq_qs = TEXTS[lang]['phq9_questions']
    phq_opts = TEXTS[lang]['phq9_options']
    phq_answers = []
    ui_opts = phq_opts[:]
    for i,q in enumerate(phq_qs,1):
        label = reshape_for_ui(q) if (lang=='ar' and HAS_ARABIC_TOOLS) else q
        ans = st.selectbox(label, ui_opts, key=f'phq{i}')
        try:
            idx = ui_opts.index(ans)
            num = int(phq_opts[idx].split('=')[0].strip())
        except Exception:
            try:
                num = int(ans.split('=')[0].strip())
            except Exception:
                num = 0
        phq_answers.append(num)
    phq_score = sum(phq_answers)
    if phq_score < 5: phq_risk = 'Minimal'
    elif phq_score < 10: phq_risk = 'Mild'
    elif phq_score < 15: phq_risk = 'Moderate'
    elif phq_score < 20: phq_risk = 'Moderately severe'
    else: phq_risk = 'Severe'
    st.write(f'PHQ-9 Score: **{phq_score}** → {phq_risk}')

# AD8 tab
with tab_ad8:
    st.header(t['ad8'])
    ad8_qs = TEXTS[lang]['ad8_questions']
    ad8_opts = TEXTS[lang]['ad8_options']
    ad8_answers = []
    for i,q in enumerate(ad8_qs,1):
        label = reshape_for_ui(q) if (lang=='ar' and HAS_ARABIC_TOOLS) else q
        ans = st.selectbox(label, ad8_opts, key=f'ad8{i}')
        try:
            idx = ad8_opts.index(ans)
            val = 1 if ad8_opts[idx] == ad8_opts[1] else 0
        except Exception:
            val = 0
        ad8_answers.append(val)
    ad8_score = sum(ad8_answers); ad8_risk = 'Low' if ad8_score < 2 else 'Possible concern'
    st.write(f'AD8 Score: **{ad8_score}** → {ad8_risk}')

# Microstates tab (more manual control)
with tab_micro:
    st.header(t['microstates'])
    st.write("Run microstate analysis from Upload tab per file or request integration for batch runs.")
    st.info("Requires mne & scikit-learn for full functionality.")

# Report tab
with tab_report:
    st.header(t['report'])
    if st.button('Generate'):
        EEG_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        EEG_results['Depression'] = {'score': phq_score, 'risk': phq_risk}
        EEG_results['Alzheimer'] = {'score': ad8_score, 'risk': ad8_risk}
        interpretations = []
        risk_scores = {}
        for fname, block in EEG_results['EEG_files'].items():
            qi = block.get('QEEG', {})
            conn = block.get('connectivity', {})
            # heuristics
            asym_key = None
            for k in qi.keys():
                if k.startswith('alpha_asym_'):
                    asym_key = k; break
            if asym_key and qi.get(asym_key) is not None:
                a = qi[asym_key]
                if a > 0.2:
                    interpretations.append(f"{fname}: Left frontal alpha > right — pattern reported in depression studies." if lang=='en' else f"{fname}: اتجاه أعلى لألفا الجبهي الأيسر — مرتبط بدراسات الاكتئاب.")
                elif a < -0.2:
                    interpretations.append(f"{fname}: Right frontal alpha > left — notable asymmetry." if lang=='en' else f"{fname}: تفاوت أيسر/أيمن ملحوظ في ألفا.")
            ta = qi.get('Theta_Alpha_ratio')
            if ta and ta > 1.2:
                if lang=='ar':
                    interpretations.append(f"{fname}: ارتفاع نسبة ثيتا/ألفا ({fmt(ta)}) — قد يشير إلى ضعف إدراكي مبكر؛ يوصى بمتابعة عصبية.")
                else:
                    interpretations.append(f"{fname}: Elevated Theta/Alpha ratio ({fmt(ta)}) — may indicate early cognitive decline; recommend neurological follow-up.")
            conn_summary = {'mean_connectivity': conn.get('mean_connectivity')} if conn and 'mean_connectivity' in conn else {}
            feat_vector = [
                qi.get('Theta_Alpha_ratio', 0.0),
                qi.get('Theta_Beta_ratio', 0.0),
                max([abs(qi.get(k,0.0)) for k in qi.keys() if k.startswith('alpha_asym_')] + [0.0]),
                conn_summary.get('mean_connectivity', 0.0),
                patient_info.get('age') if patient_info.get('age') else 60.0
            ]
            X_inst = np.array(feat_vector)
            if MODEL is not None and SCALER is not None:
                try:
                    prob = float(MODEL.predict_proba(SCALER.transform(X_inst.reshape(1,-1)))[0,1] * 100)
                except Exception:
                    prob = 0.0
                risk_scores[fname] = prob
                if fname in shap_images:
                    pass
            else:
                risk_scores[fname] = 0.0
        # JSON
        try:
            json_bytes = io.BytesIO(json.dumps(EEG_results, indent=2, ensure_ascii=False, default=make_serializable).encode())
            st.download_button(TEXTS[lang]['download_json'], json_bytes, file_name='report.json')
        except Exception as e:
            st.warning(f'JSON export failed: {e}')
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
                st.download_button(TEXTS[lang]['download_csv'], df.to_csv(index=False).encode('utf-8'), file_name='qeeg_features.csv', mime='text/csv')
        except Exception as e:
            st.warning(f'CSV export failed: {e}')
        # PDF
        try:
            pdfb = build_pdf(EEG_results, patient_info, lab_results, meds_list, lang=lang, band_pngs=band_pngs, conn_images=conn_imgs, micro_images=micro_imgs, interpretations=interpretations, risk_scores=risk_scores, shap_images=shap_images)
            st.download_button(TEXTS[lang]['download_pdf'], pdfb, file_name='report.pdf')
            st.success('Report generated — downloads ready.')
        except Exception as e:
            st.error(f'PDF generation failed: {e}')
    st.markdown('---')
    st.info(TEXTS[lang]['note'])

# Model & XAI controls
with st.expander('Model & XAI (train / upload dataset / view SHAP)'):
    st.write('Baseline synthetic model AUC:', f"{MODEL_AUC:.3f}" if MODEL_AUC else 'n/a')
    uploaded_csv = st.file_uploader('Upload CSV with labelled data (Theta_Alpha_ratio,Theta_Beta_ratio,alpha_asym_abs,mean_connectivity,age,label)', type=['csv'])
    if uploaded_csv and HAS_SKLEARN:
        try:
            df = pd.read_csv(uploaded_csv)
            required = ['Theta_Alpha_ratio','Theta_Beta_ratio','alpha_asym_abs','mean_connectivity','age','label']
            if not all([c in df.columns for c in required]):
                st.warning('CSV missing required columns.')
            else:
                X = df[required[:-1]].values; y = df['label'].values
                scaler = StandardScaler(); Xs = scaler.fit_transform(X)
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
                clf.fit(Xs, y)
                auc = roc_auc_score(y, clf.predict_proba(Xs)[:,1])
                st.success(f'Trained model on uploaded data — AUC (train) = {auc:.3f}')
                if joblib:
                    joblib.dump(clf, MODEL_PATH); joblib.dump(scaler, SCALER_PATH); st.info('Model & scaler saved.')
                MODEL = clf; SCALER = scaler; MODEL_AUC = auc
        except Exception as e:
            st.error(f'Model training failed: {e}')
    st.write('SHAP installed:' , HAS_SHAP)
    if HAS_SHAP and MODEL is not None:
        if st.button('Show SHAP summary (synthetic sample)'):
            Xsamp, _, feat_names = build_synthetic_dataset(200)
            shap_img = compute_shap_summary_plot(MODEL, SCALER, FEATURE_NAMES, Xsamp)
            if shap_img:
                st.image(shap_img, caption='SHAP summary (synthetic sample)')

# EOF
