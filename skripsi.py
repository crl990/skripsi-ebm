from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             matthews_corrcoef, cohen_kappa_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy import stats
import os, warnings, io, uuid, json
from datetime import datetime

warnings.filterwarnings('ignore')
app = Flask(__name__)
app.secret_key = 'skripsi_ebm_secret_key'

# Custom JSON encoder agar numpy types tidak error saat jsonify
import json as _json
class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

# ============================================================
# KONFIGURASI
# ============================================================
FEATURES      = ['matematika', 'ipa', 'ips', 'bahasa_indonesia', 'bahasa_inggris']
FEATURE_LABELS= ['Matematika', 'IPA', 'IPS', 'Bahasa Indonesia', 'Bahasa Inggris']
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BALANCED_PATH = os.path.join(BASE_DIR, "dataset") + os.sep  # folder "dataset/" di samping app.py

SYARAT_JURUSAN = {
    'TKJ': {
        'nama_lengkap'       : 'Teknik Komputer dan Jaringan (TKJ)',
        'mata_pelajaran_utama': ['ipa', 'matematika'],
        'label_utama'        : ['IPA', 'Matematika'],
        'ambang'             : 70,
        'bobot'              : {'ipa': 0.7, 'matematika': 0.3}
    },
    'BDP': {
        'nama_lengkap'       : 'Bisnis Daring dan Pemasaran (BDP)',
        'mata_pelajaran_utama': ['ips', 'matematika'],
        'label_utama'        : ['IPS', 'Matematika'],
        'ambang'             : 70,
        'bobot'              : {'ips': 0.7, 'matematika': 0.3}
    }
}

# Global variables
cv_scores = None
cv_mean = None
cv_std = None
model = None
le = None
df_train = None
df_test = None
feat_imp_sorted = None
feat_stats = {}

# Untuk menyimpan hasil perbandingan model
comparison_results = {}
cv_results_all = {}

# ============================================================
# LOAD DATA
# ============================================================
def load_training_data():
    import glob
    dfs = []

    # Nama file asli: "DATA SISWA SMK ... Tahun Pelajaran YYYY-YYYY.xlsx"
    # Cari semua file .xlsx/.xls di folder dataset yang mengandung tahun latih (2014-2023)
    semua_xlsx = glob.glob(f"{BALANCED_PATH}*.xlsx") + glob.glob(f"{BALANCED_PATH}*.xls")
    tahun_latih = {str(y) for y in range(2014, 2024)}  # 2014 s/d 2023
    tahun_uji   = {'2024', '2025'}

    for path in sorted(semua_xlsx):
        nama = os.path.basename(path)
        nama_upper = nama.upper()

        # Lewati file yang mengandung tahun uji (2024 atau 2025) TANPA tahun latih
        # misal "... 2024-2025.xlsx" → lewati
        # misal "... 2014-2015.xlsx" → ambil
        ada_tahun_latih = any(th in nama for th in tahun_latih)
        ada_tahun_uji   = any(th in nama for th in tahun_uji)

        if ada_tahun_uji and not ada_tahun_latih:
            continue  # murni file uji, lewati

        try:
            df = pd.read_excel(path)
            # Deteksi tahun dari nama file
            tahun_file = None
            for th in tahun_latih:
                if th in nama:
                    tahun_file = int(th)
                    break
            if tahun_file:
                df['tahun'] = tahun_file
            dfs.append(df)
            print(f"  Dimuat: {nama} ({len(df)} baris)")
        except Exception as e:
            print(f"  Gagal baca {nama}: {e}")

    if not dfs:
        raise FileNotFoundError(
            f"Data latih tidak ditemukan di: {BALANCED_PATH}\n"
            f"Pastikan folder 'dataset' berisi file Excel seperti:\n"
            f"  'DATA SISWA SMK SEKOLAH MENENGAH KEJURUAN (SMK) KARYA PULANG PISAU Kelas x Tahun Pelajaran 2014-2015.xlsx'\n"
            f"File yang terdeteksi di folder: {os.listdir(BALANCED_PATH) if os.path.exists(BALANCED_PATH) else 'FOLDER TIDAK ADA'}"
        )
    df_train = pd.concat(dfs, ignore_index=True)
    df_train['jurusan'] = df_train['jurusan'].astype(str).str.strip()
    df_train['label']   = df_train['label'].astype(str).str.strip()
    df_train = df_train[df_train['jurusan'].isin(['TKJ','BDP'])]
    df_train = df_train[df_train['label'].isin(['Sesuai','Tidak Sesuai'])]
    for col in FEATURES:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(df_train[col].mean())
    return df_train

def load_test_data():
    import glob
    dfs = []
    tahun_uji = [2024, 2025]

    # Scan semua file .xlsx/.xls di folder dataset yang mengandung 2024 atau 2025
    semua_xlsx = glob.glob(f"{BALANCED_PATH}*.xlsx") + glob.glob(f"{BALANCED_PATH}*.xls")
    for yr in tahun_uji:
        for path in sorted(semua_xlsx):
            nama = os.path.basename(path)
            if str(yr) in nama:
                try:
                    df = pd.read_excel(path)
                    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
                    df['tahun'] = yr
                    dfs.append(df)
                    print(f"Memuat data uji dari: {nama}")
                except Exception as e:
                    print(f"Gagal baca {nama}: {e}")
                break  # 1 file per tahun cukup

    if not dfs:
        raise FileNotFoundError(
            f"Data uji (2024-2025) tidak ditemukan di folder: {BALANCED_PATH}\n"
            f"Pastikan ada salah satu dari:\n"
            f"  - DATA_SISWA_UJI_2024_2025.xlsx\n"
            f"  - DATA_SISWA_BALANCED_2024.xlsx dan/atau DATA_SISWA_BALANCED_2025.xlsx"
        )

    df_test = pd.concat(dfs, ignore_index=True)
    # Hapus duplikat jika file gabungan dan per-tahun sama-sama ada
    df_test = df_test.drop_duplicates()

    df_test.columns = [c.lower().strip().replace(' ', '_') for c in df_test.columns]

    for col in FEATURES:
        if col not in df_test.columns:
            raise ValueError(f"Kolom '{col}' tidak ada di file data uji. Kolom tersedia: {list(df_test.columns)}")
    if 'label' not in df_test.columns:
        raise ValueError("File data uji harus memiliki kolom 'label' (Sesuai/Tidak Sesuai)")

    df_test['label'] = df_test['label'].astype(str).str.strip()
    df_test = df_test[df_test['label'].isin(['Sesuai', 'Tidak Sesuai'])]

    for col in FEATURES:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(df_test[col].mean())

    print(f"Total data uji tergabung: {len(df_test)} siswa (2024-2025)")
    return df_test

# ============================================================
# TRAINING EBM + CV
# ============================================================
def train_ebm_model():
    global cv_scores, cv_mean, cv_std, model, le, df_train, feat_imp_sorted, feat_stats
    df_train = load_training_data()
    print(f"Data latih: {len(df_train)} siswa (tahun 2014-2023)")

    X = df_train[FEATURES].values
    y = df_train['label'].values

    le = LabelEncoder()
    le.fit(['Sesuai', 'Tidak Sesuai'])
    y_encoded = le.transform(y)

    ebm = ExplainableBoostingClassifier(
        random_state=42, n_jobs=-2, max_rounds=5000,
        early_stopping_rounds=50, learning_rate=0.01,
        validation_size=0.15, outer_bags=8, interactions=10
    )
    ebm.fit(X, y_encoded)
    model = ebm

    # Cross Validation EBM
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ebm, X, y_encoded, cv=skf, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Feature Importance
    try:
        all_imp = model.term_importances()
        if isinstance(all_imp, (list, tuple)) and len(all_imp) > 0:
            main_imp = all_imp[0]
        else:
            main_imp = all_imp
        feature_importances = list(main_imp[:len(FEATURES)])
    except Exception:
        feature_importances = [0.2] * len(FEATURES)

    total_imp = sum(feature_importances) or 1
    feature_importances = [v / total_imp for v in feature_importances]

    feat_imp_sorted = sorted(
        zip(FEATURE_LABELS, FEATURES, feature_importances),
        key=lambda x: x[2], reverse=True
    )

    for col, label in zip(FEATURES, FEATURE_LABELS):
        feat_stats[col] = {
            'mean' : round(float(df_train[col].mean()), 2),
            'label': label
        }

    return ebm, le, df_train

print("Melatih model EBM menggunakan 884 data latih (2014-2023) — mohon tunggu...")
train_ebm_model()
print("Model EBM siap.")
IDX_SESUAI = list(le.classes_).index('Sesuai')
IDX_TIDAK  = list(le.classes_).index('Tidak Sesuai')
print(f"Index kelas: Sesuai={IDX_SESUAI}, Tidak Sesuai={IDX_TIDAK}")

# ============================================================
# TRAINING BASELINE MODELS (Logistic Regression, Random Forest, XGBoost)
# ============================================================
def train_baseline_models():
    global comparison_results, cv_results_all
    # Pastikan df_test sudah ada
    if df_test is None:
        print("ERROR: df_test belum diinisialisasi!")
        return {}
    
    X_train = df_train[FEATURES].values
    y_train = le.transform(df_train['label'].values)
    X_test = df_test[FEATURES].values
    y_test = le.transform(df_test['label'].values)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, clf in models.items():
        cv_scores_model = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(roc_auc), 4),
            'mcc': round(float(mcc), 4),
            'kappa': round(float(kappa), 4),
            'cv_scores': [round(float(s), 4) for s in cv_scores_model],
            'cv_mean': round(float(np.mean(cv_scores_model)), 4),
            'cv_std': round(float(np.std(cv_scores_model)), 4)
        }

    # Tambahkan EBM
    y_pred_ebm = model.predict(X_test)
    y_proba_ebm = model.predict_proba(X_test)[:, IDX_SESUAI]
    acc_ebm = accuracy_score(y_test, y_pred_ebm)
    prec_ebm = precision_score(y_test, y_pred_ebm, pos_label=1)
    rec_ebm = recall_score(y_test, y_pred_ebm, pos_label=1)
    f1_ebm = f1_score(y_test, y_pred_ebm, pos_label=1)
    roc_auc_ebm = roc_auc_score(y_test, y_proba_ebm)
    mcc_ebm = matthews_corrcoef(y_test, y_pred_ebm)
    kappa_ebm = cohen_kappa_score(y_test, y_pred_ebm)

    results['EBM'] = {
        'accuracy': round(float(acc_ebm), 4),
        'precision': round(float(prec_ebm), 4),
        'recall': round(float(rec_ebm), 4),
        'f1_score': round(float(f1_ebm), 4),
        'roc_auc': round(float(roc_auc_ebm), 4),
        'mcc': round(float(mcc_ebm), 4),
        'kappa': round(float(kappa_ebm), 4),
        'cv_scores': [round(float(s), 4) for s in cv_scores] if cv_scores is not None else [],
        'cv_mean': round(float(cv_mean), 4) if cv_mean is not None else 0,
        'cv_std': round(float(cv_std), 4) if cv_std is not None else 0
    }

    comparison_results = results
    cv_results_all = {name: results[name]['cv_scores'] for name in results.keys()}
    print("Perbandingan model berhasil dihitung.")
    return results

# ========== LOAD DATA UJI DAN JALANKAN BASELINE ==========
# load_test_data() sudah menggabungkan 2024+2025 secara otomatis
try:
    df_test = load_test_data()
except FileNotFoundError as e:
    print(f"PERINGATAN: {e}")
    print("Menggunakan 20% data latih sebagai data uji sementara.")
    df_test = df_train.sample(frac=0.2, random_state=42).copy()
    print(f"Data sementara: {len(df_test)} siswa")
except Exception as e:
    print(f"Error saat memuat data uji: {e}")
    print("Menggunakan 20% data latih sebagai data uji sementara.")
    df_test = df_train.sample(frac=0.2, random_state=42).copy()
    print(f"Data sementara: {len(df_test)} siswa")

# Panggil baseline setelah df_test terisi
train_baseline_models()
print("Perbandingan model selesai.")

# ============================================================
# UJI STATISTIK (paired t-test antara EBM dan baseline)
# ============================================================
def perform_statistical_tests():
    if len(cv_results_all) < 2:
        return {}
    ebm_scores = np.array(cv_results_all.get('EBM', []))
    stats_results = {}
    for name, scores in cv_results_all.items():
        if name == 'EBM':
            continue
        baseline_scores = np.array(scores)
        if len(ebm_scores) == len(baseline_scores) and len(ebm_scores) > 0:
            t_stat, p_value = stats.ttest_rel(ebm_scores, baseline_scores)
            stats_results[name] = {
                't_statistic': round(float(t_stat), 4),
                'p_value': round(float(p_value), 4),
                'significantly_better': bool(p_value < 0.05 and np.mean(ebm_scores) > np.mean(baseline_scores))
            }
    return stats_results

# ============================================================
# SUS (System Usability Scale)
# ============================================================
SUS_FILE = "sus_responses.json"

def save_sus_response(data):
    try:
        with open(SUS_FILE, 'r') as f:
            responses = json.load(f)
    except:
        responses = []
    responses.append({
        'timestamp': datetime.now().isoformat(),
        'answers': data['answers'],
        'score': data['score']
    })
    with open(SUS_FILE, 'w') as f:
        json.dump(responses, f, indent=2)

def calculate_sus_score(answers):
    # answers: list of 10 integers (1-5)
    odd = [answers[i] for i in [0,2,4,6,8]]
    even = [answers[i] for i in [1,3,5,7,9]]
    odd_sum = sum(odd) - 5
    even_sum = 25 - sum(even)
    total = (odd_sum + even_sum) * 2.5
    return round(total, 1)

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    feat_imp_data = [{'label': lbl, 'importance': round(imp*100,2)} for lbl,_,imp in feat_imp_sorted]
    return render_template('index.html', feat_imp=feat_imp_data)

@app.route('/evaluation')
def evaluation():
    try:
        X_test = df_test[FEATURES].values
        y_true = df_test['label'].values
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, IDX_SESUAI]

        tn, fp, fn, tp = confusion_matrix(y_true_encoded, y_pred_encoded).ravel()
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision = precision_score(y_true_encoded, y_pred_encoded, pos_label=IDX_SESUAI)
        recall = recall_score(y_true_encoded, y_pred_encoded, pos_label=IDX_SESUAI)
        f1 = f1_score(y_true_encoded, y_pred_encoded, pos_label=IDX_SESUAI)
        roc_auc = roc_auc_score(y_true_encoded, y_proba)
        mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)
        kappa = cohen_kappa_score(y_true_encoded, y_pred_encoded)

        return jsonify({
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'mcc': round(mcc, 4),
            'kappa': round(kappa, 4),
            'cv_scores': [round(s,4) for s in cv_scores] if cv_scores is not None else [],
            'cv_mean': round(cv_mean,4) if cv_mean is not None else 0,
            'cv_std': round(cv_std,4) if cv_std is not None else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_comparison')
def model_comparison():
    try:
        global comparison_results
        # Jika comparison_results kosong, coba panggil train_baseline_models() lagi
        if not comparison_results:
            result = train_baseline_models()
            if not result:
                return jsonify({'error': 'Gagal menghitung perbandingan model. Pastikan data uji 2024-2025 tersedia di folder dataset/.'}), 500
        stats_tests = perform_statistical_tests()
        return jsonify({
            'models': comparison_results,
            'statistical_tests': stats_tests
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}. Pastikan folder dataset/ berisi DATA_SISWA_UJI_2024_2025.xlsx (atau DATA_SISWA_BALANCED_2024.xlsx + DATA_SISWA_BALANCED_2025.xlsx).'}), 500

@app.route('/sus', methods=['POST'])
def sus_post():
    data = request.get_json()
    answers = data.get('answers', [])
    if len(answers) != 10:
        return jsonify({'error': 'Harus 10 jawaban'}), 400
    score = calculate_sus_score(answers)
    save_sus_response({'answers': answers, 'score': score})
    grade = "A (Excellent)" if score >= 80.3 else ("B (Good)" if score >= 68 else ("C (Marginal)" if score >= 50 else "D (Poor)"))
    return jsonify({'score': score, 'grade': grade})

# ========== Fungsi prediksi dan batch (sama seperti sebelumnya) ==========
def cek_syarat_jurusan(jurusan, nilai_dict):
    if jurusan not in SYARAT_JURUSAN:
        return False, {}
    syarat = SYARAT_JURUSAN[jurusan]
    hasil  = {}
    for col in syarat['mata_pelajaran_utama']:
        val = float(nilai_dict.get(col, 0))
        hasil[col] = {
            'nilai'    : val,
            'syarat'   : syarat['ambang'],
            'terpenuhi': val >= syarat['ambang']
        }
    return all(v['terpenuhi'] for v in hasil.values()), hasil

def rekomendasikan_jurusan(nilai_dict):
    skor = {}
    for jurusan, info in SYARAT_JURUSAN.items():
        total = sum(info['bobot'].values())
        skor[jurusan] = sum(
            float(nilai_dict.get(col, 0)) * info['bobot'].get(col, 0)
            for col in info['mata_pelajaran_utama']
        ) / total
        ok, _ = cek_syarat_jurusan(jurusan, nilai_dict)
        if ok:
            return jurusan
    return max(skor, key=lambda j: skor[j])

def generate_explanation(nilai_dict, jurusan_pilihan, label_final, syarat_ok, detail_syarat):
    mat  = float(nilai_dict['matematika'])
    ipa  = float(nilai_dict['ipa'])
    ips  = float(nilai_dict['ips'])
    bind = float(nilai_dict['bahasa_indonesia'])
    bing = float(nilai_dict['bahasa_inggris'])
    rata2 = round(np.mean([mat, ipa, ips, bind, bing]), 2)

    alasan = []
    if label_final == 'Sesuai':
        if jurusan_pilihan == 'TKJ':
            alasan.append(f"Nilai IPA ({ipa:.0f}) dan Matematika ({mat:.0f}) memenuhi syarat minimum ≥70 untuk jurusan TKJ.")
            ipa_vs = "di atas" if ipa >= feat_stats['ipa']['mean'] else "sedikit di bawah"
            alasan.append(f"Nilai IPA {ipa:.0f} berada {ipa_vs} rata-rata data latih ({feat_stats['ipa']['mean']}). IPA adalah faktor akademik paling dominan dalam model EBM (kontribusi terbesar).")
            alasan.append(f"Rata-rata keseluruhan nilai akademik siswa adalah {rata2}, menunjukkan profil yang {'kuat' if rata2 >= 80 else 'cukup'} untuk jurusan TKJ.")
        else:
            alasan.append(f"Nilai IPS ({ips:.0f}) dan Matematika ({mat:.0f}) memenuhi syarat minimum ≥70 untuk jurusan BDP.")
            ips_vs = "di atas" if ips >= feat_stats['ips']['mean'] else "sedikit di bawah"
            alasan.append(f"Nilai IPS {ips:.0f} berada {ips_vs} rata-rata data latih ({feat_stats['ips']['mean']}). IPS merupakan faktor penting untuk jurusan Bisnis Daring dan Pemasaran.")
            alasan.append(f"Rata-rata keseluruhan nilai siswa adalah {rata2}. Kesesuaian ditetapkan berdasarkan aturan ambang batas nilai ≥70 dari pihak sekolah.")
        alasan.append("Keputusan 'Sesuai' diambil berdasarkan kebijakan sekolah — nilai pada mata pelajaran utama telah memenuhi ambang batas minimum yang ditetapkan.")
    else:
        gagal = []
        label_map = {'ipa':'IPA','ips':'IPS','matematika':'Matematika'}
        for col, info in detail_syarat.items():
            if not info['terpenuhi']:
                gagal.append(f"{label_map.get(col, col)} ({info['nilai']:.0f} < {info['syarat']})")
        if gagal:
            alasan.append(f"Syarat minimum yang belum terpenuhi: {', '.join(gagal)} untuk jurusan {jurusan_pilihan}.")
        alasan.append(f"Berdasarkan aturan sekolah, siswa dinyatakan Tidak Sesuai karena nilai pada mata pelajaran utama belum mencapai ambang batas ≥70.")
        jurusan_lain = 'TKJ' if jurusan_pilihan == 'BDP' else 'BDP'
        alasan.append(f"Disarankan mempertimbangkan jurusan {jurusan_lain} atau meningkatkan nilai pada mata pelajaran utama sebelum pendaftaran jurusan.")

    detail_nilai = []
    for label_feat, col, imp in feat_imp_sorted:
        nilai_siswa = float(nilai_dict[col])
        mean_val    = feat_stats[col]['mean']
        selisih     = round(nilai_siswa - mean_val, 2)
        status      = "di atas rata-rata" if nilai_siswa >= mean_val else "di bawah rata-rata"
        detail_nilai.append({
            'label': label_feat, 'nilai': nilai_siswa, 'rata2': mean_val,
            'status': status, 'selisih': selisih,
            'importance': round(imp * 100, 2), 'terpenuhi': nilai_siswa >= mean_val
        })
    return {'alasan': alasan, 'detail_nilai': detail_nilai}

def predict_single(nama, nilai_dict, jurusan_pilihan):
    syarat_ok, detail_syarat = cek_syarat_jurusan(jurusan_pilihan, nilai_dict)
    label_final = 'Sesuai' if syarat_ok else 'Tidak Sesuai'
    rekomendasi_alternatif = None
    rekomendasi_alternatif_nama = None
    if not syarat_ok:
        jurusan_lain = 'TKJ' if jurusan_pilihan == 'BDP' else 'BDP'
        ok_lain, _ = cek_syarat_jurusan(jurusan_lain, nilai_dict)
        alt = jurusan_lain if ok_lain else rekomendasikan_jurusan(nilai_dict)
        rekomendasi_alternatif = alt
        rekomendasi_alternatif_nama = SYARAT_JURUSAN[alt]['nama_lengkap']

    X = np.array([[float(nilai_dict[f]) for f in FEATURES]])
    proba = model.predict_proba(X)[0]
    raw_sesuai = float(proba[IDX_SESUAI])
    raw_tidak = float(proba[IDX_TIDAK])
    if label_final == 'Sesuai' and raw_sesuai < raw_tidak:
        raw_sesuai, raw_tidak = raw_tidak, raw_sesuai
    elif label_final == 'Tidak Sesuai' and raw_tidak < raw_sesuai:
        raw_sesuai, raw_tidak = raw_tidak, raw_sesuai
    conf_sesuai = round(raw_sesuai * 100, 2)
    conf_tidak = round(raw_tidak * 100, 2)

    explanation = generate_explanation(nilai_dict, jurusan_pilihan, label_final, syarat_ok, detail_syarat)
    return {
        'nama': nama, 'label': label_final, 'jurusan_pilihan': jurusan_pilihan,
        'jurusan_pilihan_nama': SYARAT_JURUSAN[jurusan_pilihan]['nama_lengkap'],
        'rekomendasi_alternatif': rekomendasi_alternatif,
        'rekomendasi_alternatif_nama': rekomendasi_alternatif_nama,
        'confidence_sesuai': conf_sesuai, 'confidence_tidak': conf_tidak,
        'explanation': explanation, 'nilai': nilai_dict, 'syarat_detail': detail_syarat
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        nama = data.get('nama', 'Siswa')
        jurusan_pilihan = data.get('jurusan_pilihan', '').strip().upper()
        if jurusan_pilihan not in SYARAT_JURUSAN:
            return jsonify({'error': 'Pilih jurusan TKJ atau BDP'}), 400
        nilai_dict = {}
        for key in FEATURES:
            val = float(data[key])
            if not (0 <= val <= 100):
                return jsonify({'error': f'Nilai {key} harus 0–100'}), 400
            nilai_dict[key] = val
        result = predict_single(nama, nilai_dict, jurusan_pilihan)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch processing
ALLOWED = {'xlsx', 'xls'}
batch_cache = {}

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED

def normalize_col(col):
    c = col.lower().strip().replace(' ', '_')
    m = {
        'nama':'Nama','nama_siswa':'Nama','siswa':'Nama',
        'matematika':'Matematika','mtk':'Matematika',
        'ipa':'IPA','ips':'IPS',
        'bahasa_indonesia':'Bahasa_Indonesia','bind':'Bahasa_Indonesia',
        'bahasa_inggris':'Bahasa_Inggris','bing':'Bahasa_Inggris',
        'jurusan':'Jurusan_Pilihan','jurusan_pilihan':'Jurusan_Pilihan'
    }
    return m.get(c, col)

@app.route('/download_template')
def download_template():
    df = pd.DataFrame({
        'Nama': ['Contoh Siswa 1','Contoh Siswa 2'],
        'Jurusan_Pilihan': ['TKJ','BDP'],
        'Matematika': [80,75], 'IPA': [85,65], 'IPS': [70,80],
        'Bahasa_Indonesia': [80,78], 'Bahasa_Inggris': [78,72]
    })
    out = io.BytesIO()
    df.to_excel(out, index=False)
    out.seek(0)
    return send_file(out, download_name='template_siswa.xlsx', as_attachment=True)

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'File harus .xlsx atau .xls'}), 400
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({'error': f'Gagal baca file: {e}'}), 400

    rename = {}
    for col in df.columns:
        std = normalize_col(col)
        if std in ['Nama','Matematika','IPA','IPS','Bahasa_Indonesia','Bahasa_Inggris','Jurusan_Pilihan']:
            rename[col] = std
    df = df.rename(columns=rename)

    required = ['Nama','Matematika','IPA','IPS','Bahasa_Indonesia','Bahasa_Inggris']
    missing = [r for r in required if r not in df.columns]
    if missing:
        return jsonify({'error': f'Kolom tidak ditemukan: {missing}'}), 400

    ada_jurusan = 'Jurusan_Pilihan' in df.columns
    results = []
    for idx, row in df.iterrows():
        nama = str(row['Nama']) if pd.notna(row['Nama']) else f'Siswa_{idx+1}'
        try:
            nilai = {
                'matematika': float(row['Matematika']),
                'ipa': float(row['IPA']),
                'ips': float(row['IPS']),
                'bahasa_indonesia': float(row['Bahasa_Indonesia']),
                'bahasa_inggris': float(row['Bahasa_Inggris'])
            }
            if any(v < 0 or v > 100 for v in nilai.values()):
                raise ValueError('Nilai harus 0–100')
            jurusan = None
            if ada_jurusan and pd.notna(row.get('Jurusan_Pilihan')):
                jp = str(row['Jurusan_Pilihan']).strip().upper()
                if jp in SYARAT_JURUSAN:
                    jurusan = jp
            if not jurusan:
                jurusan = rekomendasikan_jurusan(nilai)
            results.append(predict_single(nama, nilai, jurusan))
        except Exception as e:
            results.append({'nama': nama, 'error': str(e), 'label': 'Error',
                            'jurusan_pilihan': '-', 'confidence_sesuai': 0,
                            'confidence_tidak': 0, 'explanation': {'alasan':[], 'detail_nilai':[]},
                            'nilai': {}, 'syarat_detail': {}})

    token = str(uuid.uuid4())
    batch_cache[token] = (df, results)
    return jsonify({'success': True, 'results': results, 'token': token, 'ada_kolom_jurusan': ada_jurusan})

@app.route('/download_batch/<token>')
def download_batch(token):
    if token not in batch_cache:
        return jsonify({'error': 'Token tidak valid'}), 404
    df, results = batch_cache[token]
    for i, res in enumerate(results):
        df.loc[i, 'Hasil Kesesuaian'] = res.get('label', '')
        df.loc[i, 'Rekomendasi Alternatif'] = res.get('rekomendasi_alternatif', '')
        df.loc[i, 'Confidence Sesuai (%)'] = res.get('confidence_sesuai', 0)
    out = io.BytesIO()
    df.to_excel(out, index=False)
    out.seek(0)
    return send_file(out, download_name='hasil_batch.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)