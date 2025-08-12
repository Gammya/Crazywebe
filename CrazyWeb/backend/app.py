# backend/app.py
from flask import Flask, send_from_directory, jsonify, request
from datetime import datetime, timezone
import os, random, json, traceback
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Пути (подставь, если у тебя структура иная)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "crazy_time_spin_model.h5")
JSON_PATH = os.path.join(BASE_DIR, "crazy-time.json")

# Попытка импортировать tensorflow (опционально)
try:
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    tf_available = True
except Exception:
    tf_available = False

model = None
label_enc = None
scaler = None
sector_order = ['one','two','five','ten','coinflip','cashhunt','pachinko','crazytime']  # порядок секторов на колесе (8 секторов)
# Если у тебя другой порядок — измени sector_order в frontend и тут.

def load_encoders_and_scaler():
    global label_enc, scaler
    # Загружаем crazy-time.json и строим LabelEncoder / StandardScaler
    if not os.path.exists(JSON_PATH):
        print("[WARN] crazy-time.json не найден:", JSON_PATH)
        return
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = data.get('data', {}).get('resultsData') if isinstance(data.get('data'), dict) else data.get('data')
    if rows is None:
        rows = data.get('data', {}).get('resultsData', [])
    import pandas as pd
    df = pd.DataFrame(rows)
    # Берём колонки, которые использовались при обучении в crazy_time_nn.py
    # В примере использовались: 'multiplier', 'slotResultSymbol', 'spinResultSymbol', 'totalPayout'
    df = df[['multiplier','slotResultSymbol','spinResultSymbol','totalPayout']].copy()
    # Заполняем пропуски
    df['multiplier'] = df['multiplier'].fillna('1X').astype(str)
    df['slotResultSymbol'] = df['slotResultSymbol'].fillna('unknown').astype(str)
    df['spinResultSymbol'] = df['spinResultSymbol'].fillna('unknown').astype(str)
    # LabelEncode categorical columns
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le_mult = LabelEncoder().fit(df['multiplier'])
    le_slot = LabelEncoder().fit(df['slotResultSymbol'])
    le_spin = LabelEncoder().fit(df['spinResultSymbol'])
    # store encoders in dict
    label_encoders = {
        'multiplier': le_mult,
        'slotResultSymbol': le_slot,
        'spinResultSymbol': le_spin
    }
    # Build a scaler for numeric features (we used 'multiplier' encoded, 'slotResultSymbol' encoded and 'totalPayout')
    X = pd.DataFrame({
        'multiplier': le_mult.transform(df['multiplier']),
        'slotResultSymbol': le_slot.transform(df['slotResultSymbol']),
        'totalPayout': df['totalPayout'].fillna(0).astype(float)
    })
    scaler_local = StandardScaler().fit(X)
    return label_encoders, scaler_local

# Попытка загрузить модель и энкодеры
if tf_available and os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Модель загружена:", MODEL_PATH)
    except Exception as e:
        print("[WARN] Не удалось загрузить модель:", e)
        model = None
else:
    print("[INFO] TensorFlow недоступен или модель отсутствует — будем работать в fallback режиме.")

try:
    label_encoders, scaler = load_encoders_and_scaler()
    print("[INFO] Энкодеры и scaler успешно созданы.")
except Exception:
    label_encoders = None
    scaler = None
    print("[WARN] Ошибка создания энкодеров/скейлера:", traceback.format_exc())

# helper: fetch latest spins from freeslotmania or from local json
import requests
def fetch_last_spins_from_api(limit=5):
    try:
        url = "https://freeslotmania.com/stats.json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        spins = []
        # структура может отличаться — пробуем варианты
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            for entry in data['data'][:limit]:
                # try spinResultSymbol, result numeric, or result string
                if 'spinResultSymbol' in entry:
                    spins.append({
                        'multiplier': entry.get('multiplier'),
                        'spinResultSymbol': entry.get('spinResultSymbol'),
                        'totalPayout': entry.get('totalPayout', 0)
                    })
                elif 'result' in entry:
                    # если числовой код — конвертируем по известной карте
                    result_map = {1:'one',2:'two',5:'five',10:'ten',400:'crazytime',300:'coinflip',200:'pachinko',100:'cashhunt'}
                    res = entry.get('result')
                    spins.append({
                        'multiplier': entry.get('multiplier') or entry.get('min_multiplier') or entry.get('left_multiplier'),
                        'spinResultSymbol': result_map.get(res, str(res)),
                        'totalPayout': entry.get('total_payout', entry.get('totalPayout',0))
                    })
        return spins
    except Exception:
        return None

def prepare_input_for_model(spins):
    """
    spins: list of dicts with keys multiplier (like '1X' or number), spinResultSymbol, totalPayout
    returns: numpy array shaped (1, n_features) or (1, timesteps*features) depending on model expectations
    We'll mimic the training: features = [multiplier_encoded, slotResultSymbol_encoded, totalPayout]
    and then scale.
    """
    if not label_encoders or not scaler:
        return None
    rows = []
    for s in spins:
        m = s.get('multiplier', '1X')
        # ensure format like '1X'
        if isinstance(m, (int,float)):
            m = f"{int(m)}X"
        m = str(m)
        # slotResultSymbol may be present; if not use spinResultSymbol
        slot = s.get('slotResultSymbol') or s.get('spinResultSymbol') or 'unknown'
        payout = s.get('totalPayout') or s.get('total_payout') or 0
        # encode
        try:
            me = label_encoders['multiplier'].transform([m])[0]
        except Exception:
            try:
                # fallback: try remove X
                me = label_encoders['multiplier'].transform([str(m).upper()])[0]
            except Exception:
                me = 0
        try:
            se = label_encoders['slotResultSymbol'].transform([slot])[0]
        except Exception:
            try:
                se = label_encoders['slotResultSymbol'].transform([slot.lower()])[0]
            except Exception:
                se = 0
        rows.append([me, se, float(payout)])
    if len(rows) == 0:
        return None
    # If model expects a sequence (LSTM), we try to shape to (1, timesteps, features)
    X = np.array(rows)
    # scale using scaler (fit on training features)
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        # if scaler expects different shape, try fit_transform fallback
        X_scaled = X
    # Many models expect fixed shape; if model.input_shape has 3 dims -> expand to 3D
    if model is not None:
        in_shape = model.input_shape
        if isinstance(in_shape, tuple) and len(in_shape) == 3:
            # shape (batch, timesteps, features). Ensure timesteps = rows.shape[0]
            X_scaled = np.expand_dims(X_scaled, axis=0)  # (1, timesteps, features)
        else:
            X_scaled = X_scaled.reshape(1, -1)  # (1, features)
    else:
        X_scaled = X_scaled.reshape(1, -1)
    return X_scaled

def decode_prediction(pred_vector):
    """Если модель возвращает softmax — вернём метку и confidence"""
    # If vector with probs
    if isinstance(pred_vector, (list, tuple, np.ndarray)):
        arr = np.array(pred_vector)
        idx = int(np.argmax(arr))
        conf = float(arr[idx])
        # inverse transform using label_encoders['spinResultSymbol']
        inv_map = {v:k for k,v in enumerate(label_encoders['spinResultSymbol'].classes_) } if label_encoders and 'spinResultSymbol' in label_encoders else None
        try:
            label = label_encoders['spinResultSymbol'].inverse_transform([idx])[0]
        except Exception:
            # fallback: use sector_order mapping if idx in range
            label = sector_order[idx % len(sector_order)]
        return label, round(conf, 3)
    else:
        return str(pred_vector), 0.0

# API: отдаём прогнозы (реальные, если модель доступна)
@app.route('/api/signals', methods=['GET'])
def api_signals():
    try:
        cnt = int(request.args.get('count', 5))
    except:
        cnt = 5
    preds = []
    # Получаем последние спины из API
    spins = fetch_last_spins_from_api(limit=10) or []
    # Подготовим данные для модели: берем последние N (train used 5)
    if spins and model is not None and label_encoders is not None and scaler is not None:
        X = prepare_input_for_model(spins[-5:])
        if X is not None:
            try:
                probs = model.predict(X)
                # Если модель возвращает 2D array with softmax (1, classes)
                if probs.ndim == 2:
                    label, conf = decode_prediction(probs[0])
                    ts = datetime.now(timezone.utc).isoformat()
                    preds.append({'prediction': label, 'confidence': conf, 'timestamp': ts, 'mode':'model'})
                    # fill rest with random variants or repeated pred
                    for _ in range(cnt-1):
                        preds.append({'prediction': label, 'confidence': conf, 'timestamp': ts, 'mode':'model'})
                    # push to history
                    return jsonify({'predictions': preds})
                # else handle other shapes
            except Exception as e:
                print("Model predict error:", e)
                # fallback below
                pass
    # fallback mode (random)
    CHOICES = ['one','two','five','ten','coinflip','cashhunt','pachinko','crazytime']
    for _ in range(cnt):
        label = random.choice(CHOICES)
        preds.append({'prediction': label, 'confidence': round(random.uniform(0.45,0.96),2), 'timestamp': datetime.now(timezone.utc).isoformat(), 'mode':'random'})
    return jsonify({'predictions': preds})

# API: отдаём текущий (последний) спин — может пригодиться для синхронизации колеса
@app.route('/api/last-spin', methods=['GET'])
def api_last_spin():
    spins = fetch_last_spins_from_api(limit=1)
    if spins and len(spins)>0:
        return jsonify({'last': spins[0]})
    else:
        return jsonify({'last': None})

# Раздача frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print("[INFO] Starting Flask app")
    app.run(host='0.0.0.0', port=5000)
