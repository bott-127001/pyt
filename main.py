import os
import sqlite3
import json
import threading
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

from flask import Flask, request, jsonify, redirect, render_template, url_for, session
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from apscheduler.schedulers.background import BackgroundScheduler
import requests

# --- Main Flask App Instance ---
flask_app = Flask(__name__)
CORS(flask_app) # Enable CORS for all routes
flask_app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_session_management_merged')

# --- Configurations ---
# Common
IST = timezone(timedelta(hours=5, minutes=30))

# ML Backend specific
ML_DB_FILE = os.environ.get('ML_DB_FILE', 'ml_data.db')

# Main App specific
EMPEROR_CLIENT_ID = os.environ.get('EMPEROR_CLIENT_ID', '6a4f686a-5933-4d96-a772-e5de90b8417c')
EMPEROR_CLIENT_SECRET = os.environ.get('EMPEROR_CLIENT_SECRET', 'x3tyejmpgj')
EMPEROR_DB_FILE = os.environ.get('EMPEROR_DB_FILE', 'emperor_tokens.db')

KING_CLIENT_ID = os.environ.get('KING_CLIENT_ID', '192ea5d0-d3ca-407d-8651-045bd923b3a2')
KING_CLIENT_SECRET = os.environ.get('KING_CLIENT_SECRET', 'lmg6tg1m6k')
KING_DB_FILE = os.environ.get('KING_DB_FILE', 'king_tokens.db')

REDIRECT_URI = os.environ.get('UPSTOX_REDIRECT_URI', 'https://www.google.co.in/')
AUTH_URL = 'https://api.upstox.com/v2/login/authorization/dialog'
TOKEN_URL = 'https://api.upstox.com/v2/login/authorization/token'

# --- Database Initialization Functions ---
def init_ml_db():
    """Initialize the SQLite database for ML data."""
    conn = sqlite3.connect(ML_DB_FILE)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            snapshot_timestamp TEXT NOT NULL,
            metrics_data TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_prompt TEXT,
            model_prediction TEXT,
            user_feedback TEXT,
            live_metrics_snapshot TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            state_data TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def init_db(db_file):
    """Initialize a specific SQLite database file for app data."""
    if not db_file:
        print(f"Warning: init_db called with no db_file specified.")
        return
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS option_chain_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            strike_price REAL NOT NULL,
            raw_data TEXT NOT NULL,
            fetch_timestamp TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshot_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS baseline_metrics (
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_date TEXT NOT NULL,
            UNIQUE(expiry_date, metric_name, metric_date)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS delta_percentage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- Initialize Databases on Startup ---
init_ml_db()
if EMPEROR_DB_FILE: init_db(EMPEROR_DB_FILE)
if KING_DB_FILE: init_db(KING_DB_FILE)

# --- ML Model Definition ---
class OptionChainMLModel:
    def __init__(self):
        self._load_state()
        self.rules = {}

    def _load_state(self):
        conn = sqlite3.connect(ML_DB_FILE)
        c = conn.cursor()
        c.execute('SELECT state_data FROM model_state ORDER BY timestamp DESC LIMIT 1')
        row = c.fetchone()
        conn.close()
        if row:
            try:
                self.rules = json.loads(row[0])
                print("ML model state loaded.")
            except json.JSONDecodeError:
                print("Error loading ML model state.")
                self.rules = {}
        else:
            print("No previous ML model state found. Starting fresh.")
            self.rules = {}

    def _save_state(self):
        conn = sqlite3.connect(ML_DB_FILE)
        c = conn.cursor()
        state_data = json.dumps(self.rules)
        timestamp = datetime.now(IST).isoformat()
        c.execute('INSERT INTO model_state (timestamp, state_data) VALUES (?, ?)', (timestamp, state_data))
        conn.commit()
        conn.close()
        print("ML model state saved.")

    def predict(self, live_metrics, user_prompt):
        print(f"ML Model received live metrics for prediction: {list(live_metrics.keys())}")
        print(f"User prompt: {user_prompt}")

        features_for_bias = {k: live_metrics.get(k) for k in [
            'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
            'CE_totalTradedVolume_delta_percent', 'PE_totalTradedVolume_delta_percent',
            'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price',
            'high_call_oi_strikes', 'high_put_oi_strikes'
        ]}
        features_for_strength = {k: live_metrics.get(k) for k in [
             'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
             'CE_totalTradedVolume_delta_percent', 'PE_totalTradedVolume_delta_percent',
             'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price',
             'high_call_oi_strikes', 'high_put_oi_strikes'
        ]}
        features_for_participants = {k: live_metrics.get(k) for k in [
            'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
            'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price'
        ]}

        current_price = features_for_bias.get('price')
        high_call_strikes = features_for_bias.get('high_call_oi_strikes') or []
        high_put_strikes = features_for_bias.get('high_put_oi_strikes', [])

        bias = "Neutral"
        strength = "Weak"
        participants = "Undetermined"
        confidence = 50.0
        analysis_notes = []

        prompt_lower = user_prompt.lower()
        request_bias = any(word in prompt_lower for word in ['bias', 'direction', 'outlook'])
        request_strength = any(word in prompt_lower for word in ['strength', 'momentum', 'speed'])
        request_participants = any(word in prompt_lower for word in ['participants', 'oi', 'volume', 'iv', 'delta', 'who is active'])
        request_levels = any(word in prompt_lower for word in ['support', 'resistance', 'levels', 'high oi'])

        ce_oi_delta_pct = features_for_bias.get('CE_openInterest_delta_percent', 0) or 0
        pe_oi_delta_pct = features_for_bias.get('PE_openInterest_delta_percent', 0) or 0

        if ce_oi_delta_pct > pe_oi_delta_pct * 1.2:
             bias = "Bullish"
             confidence = min(confidence + 15, 100)
             analysis_notes.append("CE OI delta % significantly higher than PE OI delta %.")
        elif pe_oi_delta_pct > ce_oi_delta_pct * 1.2:
             bias = "Bearish"
             confidence = min(confidence + 15, 100)
             analysis_notes.append("PE OI delta % significantly higher than CE OI delta %.")

        ce_vol_delta_pct = features_for_strength.get('CE_totalTradedVolume_delta_percent', 0) or 0
        pe_vol_delta_pct = features_for_strength.get('PE_totalTradedVolume_delta_percent', 0) or 0
        if ce_vol_delta_pct + pe_vol_delta_pct > 50:
             strength = "Strong"
             confidence = min(confidence + 10, 100)
             analysis_notes.append("Overall volume delta % is high.")

        ce_iv_delta_pct = features_for_participants.get('CE_IV_delta_percent', 0) or 0
        pe_iv_delta_pct = features_for_participants.get('PE_IV_delta_percent', 0) or 0

        if ce_oi_delta_pct > 0 and ce_iv_delta_pct > 0:
             participants = "New Longs (CE)"
             confidence = min(confidence + 10, 100)
             analysis_notes.append("CE OI and IV delta % both positive, suggesting new CE longs.")
        elif pe_oi_delta_pct > 0 and pe_iv_delta_pct > 0:
             participants = "New Longs (PE)"
             confidence = min(confidence + 10, 100)
             analysis_notes.append("PE OI and IV delta % both positive, suggesting new PE longs.")

        if request_participants and participants == "Undetermined":
             analysis_notes.append(f"Current CE OI Delta %: {ce_oi_delta_pct:.2f}%, PE OI Delta %: {pe_oi_delta_pct:.2f}%.")
             analysis_notes.append(f"Current CE Volume Delta %: {ce_vol_delta_pct:.2f}%, PE Volume Delta %: {pe_vol_delta_pct:.2f}%.")
             analysis_notes.append(f"Current CE IV Delta %: {ce_iv_delta_pct:.2f}%, PE IV Delta %: {pe_iv_delta_pct:.2f}%.")

        if current_price is not None and high_call_strikes:
            if high_call_strikes[0] is not None and high_call_strikes[0] > 0 and \
               abs(high_call_strikes[0] - current_price) < (high_call_strikes[0] * 0.002):
                analysis_notes.append(f"Price ({current_price}) is approaching strong Call OI resistance at {high_call_strikes[0]}.")
                if bias == "Bullish":
                    strength = "Weak" if strength == "Neutral" else strength
                    confidence = max(confidence - 5, 0)

        if current_price is not None and high_put_strikes:
            if high_put_strikes[0] is not None and high_put_strikes[0] > 0 and \
               abs(current_price - high_put_strikes[0]) < (high_put_strikes[0] * 0.002):
                analysis_notes.append(f"Price ({current_price}) is near strong Put OI support at {high_put_strikes[0]}.")
                if bias == "Bearish":
                    strength = "Weak" if strength == "Neutral" else strength
                    confidence = max(confidence - 5, 0)

        prediction_to_log = {
            "bias": bias, "strength": strength, "participants": participants,
            "confidence": round(confidence, 2), "analysis_notes": analysis_notes
        }
        response_to_user = {}
        if not (request_bias or request_strength or request_participants or request_levels):
            response_to_user = prediction_to_log.copy()
        else:
            if request_bias: response_to_user['bias'] = bias
            if request_strength: response_to_user['strength'] = strength
            if request_participants: response_to_user['participants'] = participants
            response_to_user['confidence'] = round(confidence, 2)
            relevant_notes = []
            if analysis_notes:
                for note in analysis_notes:
                    note_lower = note.lower()
                    is_note_relevant = False
                    if request_bias and any(kw in note_lower for kw in ["bias", "oi delta", "bullish", "bearish", "direction"]): is_note_relevant = True
                    if request_strength and any(kw in note_lower for kw in ["strength", "volume delta", "strong", "weak", "momentum"]): is_note_relevant = True
                    if request_participants and any(kw in note_lower for kw in ["participants", "oi", "iv", "delta", "longs", "shorts", "active"]): is_note_relevant = True
                    if request_levels and any(kw in note_lower for kw in ["support", "resistance", "level", "strike"]): is_note_relevant = True
                    if is_note_relevant: relevant_notes.append(note)
            response_to_user['analysis_notes'] = relevant_notes
            if request_levels:
                if 'analysis_notes' not in response_to_user: response_to_user['analysis_notes'] = []
                call_sr_summary_note = f"Key Call OI Resistance levels: {high_call_strikes}"
                put_sr_summary_note = f"Key Put OI Support levels: {high_put_strikes}"
                has_call_sr_details = any(str(s_strike) in note for s_strike in high_call_strikes for note in response_to_user.get('analysis_notes', [])) if high_call_strikes else False
                has_put_sr_details = any(str(s_strike) in note for s_strike in high_put_strikes for note in response_to_user.get('analysis_notes', [])) if high_put_strikes else False
                if high_call_strikes and not has_call_sr_details: response_to_user['analysis_notes'].append(call_sr_summary_note)
                if high_put_strikes and not has_put_sr_details: response_to_user['analysis_notes'].append(put_sr_summary_note)

        interaction_id = self._log_interaction(user_prompt, prediction_to_log, live_metrics)
        response_to_user['interaction_id'] = interaction_id
        return response_to_user

    def train_incremental(self, snapshot_metrics):
        print(f"ML Model received 15-min snapshot for training: {list(snapshot_metrics.keys())}")
        # ... Incremental training logic ...
        self._save_state()

    def process_feedback(self, user_feedback):
        print(f"ML Model received user feedback: {json.dumps(user_feedback, indent=2)}")
        if isinstance(user_feedback, dict) and user_feedback.get('rule_suggestion'):
            suggested_rule_text = user_feedback['rule_suggestion']
            if 'user_suggested_rules' not in self.rules: self.rules['user_suggested_rules'] = []
            self.rules['user_suggested_rules'].append({"raw_rule": suggested_rule_text, "added_at": datetime.now(IST).isoformat()})
            print(f"Stored user suggested rule: {suggested_rule_text}")
        if isinstance(user_feedback, dict) and user_feedback.get('correction'):
            print(f"User provided correction: {user_feedback['correction']}")

        interaction_id_to_update = user_feedback.get('interaction_id') if isinstance(user_feedback, dict) else None
        self._log_interaction(user_feedback=user_feedback, interaction_id_to_update=interaction_id_to_update)
        self._save_state()

    def _log_interaction(self, user_prompt=None, model_prediction=None, live_metrics_snapshot=None, user_feedback=None, interaction_id_to_update=None):
        conn = sqlite3.connect(ML_DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now(IST).isoformat()
        logged_interaction_id = None
        if interaction_id_to_update and user_feedback:
             c.execute('UPDATE user_interactions SET user_feedback = ? WHERE id = ?', (json.dumps(user_feedback), interaction_id_to_update))
             logged_interaction_id = interaction_id_to_update
             print(f"Updated interaction {interaction_id_to_update} with feedback.")
        else:
            c.execute('''
                INSERT INTO user_interactions (timestamp, user_prompt, model_prediction, user_feedback, live_metrics_snapshot)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, user_prompt, json.dumps(model_prediction) if model_prediction else None,
                  json.dumps(user_feedback) if user_feedback else None, json.dumps(live_metrics_snapshot) if live_metrics_snapshot else None))
            logged_interaction_id = c.lastrowid
            print(f"Logged new interaction with ID: {logged_interaction_id}")
        conn.commit()
        conn.close()
        return logged_interaction_id

# --- ML Model Instance ---
ml_model = OptionChainMLModel()

# --- Scheduler Setup ---
scheduler = BackgroundScheduler(timezone=IST)
scheduler_lock = threading.Lock()

# --- Helper Functions (from app.py) ---
def get_db_file(): # This is used by routes within request context
    user_role = session.get('user_role')
    if user_role == 'emperor': return EMPEROR_DB_FILE
    if user_role == 'king': return KING_DB_FILE
    return None

def store_token(access_token, refresh_token, expires_in_seconds, db_file):
    if not db_file:
        print("store_token: No DB file provided.")
        return
    expires_at = datetime.now(IST) + timedelta(seconds=expires_in_seconds)
    created_at = datetime.now(IST)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('DELETE FROM tokens')
    c.execute('INSERT INTO tokens (access_token, refresh_token, expires_at, created_at) VALUES (?, ?, ?, ?)',
              (access_token, refresh_token, expires_at.isoformat(), created_at.isoformat()))
    conn.commit()
    conn.close()

def get_token(db_file_override=None):
    db_file = db_file_override if db_file_override else get_db_file() # Use override if provided
    if not db_file:
        print("get_token: No DB file could be determined (session or override).")
        return None, None, None
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT access_token, refresh_token, expires_at FROM tokens ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    if row:
        access_token, refresh_token, expires_at_str = row
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now(IST) < expires_at:
            return access_token, refresh_token, expires_at
    return None, None, None

# --- Data Fetching and Calculation Logic (from app.py) ---
def fetch_option_chain_data(expiry_date, db_file, instrument_key='NSE_INDEX|Nifty 50'):
    access_token, _, _ = get_token(db_file_override=db_file) # Pass db_file to get_token
    if not access_token:
        print("No valid access token. Cannot fetch option chain data.")
        return
    if not db_file:
        print(f"No DB file determined for fetching option chain data for expiry {expiry_date}.")
        return

    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    url = f'https://api.upstox.com/v2/option/chain?instrument_key={instrument_key}&expiry_date={expiry_date}'
    
    try:
        print(f"Fetching option chain data from API for expiry {expiry_date}, instrument {instrument_key}, db: {db_file}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, str): data = json.loads(data)
    except Exception as e:
        print(f"Error fetching option chain data: {e}")
        return

    print(f"Fetched {len(data.get('data', []))} option chain items for expiry {expiry_date}")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('DELETE FROM option_chain_data WHERE expiry_date = ?', (expiry_date,))
    
    cutoff_date = (datetime.now(IST) - timedelta(days=7)).date().isoformat()
    c.execute('DELETE FROM option_chain_data WHERE expiry_date < ?', (cutoff_date,))
    c.execute('DELETE FROM metrics WHERE expiry_date < ?', (cutoff_date,))
    c.execute('DELETE FROM snapshot_metrics WHERE expiry_date < ?', (cutoff_date,))
    c.execute('DELETE FROM delta_percentage_history WHERE expiry_date < ?', (cutoff_date,))
    # Note: baseline_metrics are cleaned by metric_date in daily_cleanup_job

    fetch_timestamp = datetime.now(IST).isoformat()
    for item in data.get('data', []):
        strike_price = item.get('strike_price') or item.get('strikePrice')
        raw_json = json.dumps(item)
        c.execute('INSERT INTO option_chain_data (expiry_date, strike_price, raw_data, fetch_timestamp) VALUES (?, ?, ?, ?)',
                  (expiry_date, strike_price, raw_json, fetch_timestamp))
    conn.commit()

    spot_price = 0
    option_data_list = data.get('data', [])
    if option_data_list:
        spot_price_raw = option_data_list[0].get('underlying_spot_price')
        try:
            if spot_price_raw is not None: spot_price = float(spot_price_raw)
        except (ValueError, TypeError): print(f"Could not parse spot_price: {spot_price_raw}")

    c.execute('SELECT raw_data FROM option_chain_data WHERE expiry_date = ?', (expiry_date,))
    parsed_option_chain_strikes = [json.loads(row[0]) for row in c.fetchall()]

    atm_strike = None
    if parsed_option_chain_strikes:
        min_diff = float('inf')
        for strike_obj in parsed_option_chain_strikes:
            current_strike_price = strike_obj.get('strike_price')
            if current_strike_price is not None:
                diff = abs(current_strike_price - spot_price)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = current_strike_price
        if atm_strike is None: # Fallback
             atm_strike = parsed_option_chain_strikes[len(parsed_option_chain_strikes)//2].get('strike_price', 0)


    today_str = datetime.now(IST).date().isoformat()
    initial_totals_for_day = {}
    c.execute('SELECT metric_name, metric_value FROM baseline_metrics WHERE expiry_date = ? AND metric_date = ?', (expiry_date, today_str))
    is_baseline_set_for_today = False
    baseline_rows = c.fetchall()
    if baseline_rows:
        is_baseline_set_for_today = True
        for bl_metric_name, bl_metric_value in baseline_rows:
            initial_totals_for_day[bl_metric_name] = bl_metric_value
    
    filtered_options_for_summation = {'CE': [], 'PE': []}
    if atm_strike is not None: # Proceed only if ATM strike is determined
        for strike_obj in parsed_option_chain_strikes:
            strike_price_val = strike_obj.get('strike_price')
            if strike_price_val is None: continue
            ce_option_data = strike_obj.get('call_options')
            pe_option_data = strike_obj.get('put_options')
            if strike_price_val == atm_strike:
                if ce_option_data: filtered_options_for_summation['CE'].append(ce_option_data)
                if pe_option_data: filtered_options_for_summation['PE'].append(pe_option_data)
            elif strike_price_val < atm_strike and pe_option_data: filtered_options_for_summation['PE'].append(pe_option_data)
            elif strike_price_val > atm_strike and ce_option_data: filtered_options_for_summation['CE'].append(ce_option_data)

    fields_to_sum = ['bidQty', 'askQty', 'openInterest', 'totalTradedVolume', 'IV', 'delta']
    field_key_map = {
        'bidQty': ['market_data', 'bid_qty'], 'askQty': ['market_data', 'ask_qty'],
        'openInterest': ['market_data', 'oi'], 'totalTradedVolume': ['market_data', 'volume'],
        'IV': ['option_greeks', 'iv'], 'delta': ['option_greeks', 'delta']
    }
    current_totals = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}

    for side in ['CE', 'PE']:
        for option_data_item in filtered_options_for_summation[side]:
            for field_name in fields_to_sum:
                keys = field_key_map[field_name]
                value_container = option_data_item
                val = None
                try:
                    for key_part in keys:
                        value_container = value_container.get(key_part, {}) if isinstance(value_container, dict) else None
                        if value_container is None: break
                    val = value_container
                    if val is not None and not isinstance(val, (dict, str)): current_totals[side][field_name] += float(val)
                except (TypeError, ValueError): pass

    if not is_baseline_set_for_today and parsed_option_chain_strikes:
        for side in ['CE', 'PE']:
            for field, total_val in current_totals[side].items():
                baseline_metric_name = f"{side}_{field}_total_baseline"
                c.execute('INSERT OR IGNORE INTO baseline_metrics (expiry_date, metric_name, metric_value, metric_date) VALUES (?, ?, ?, ?)',
                          (expiry_date, baseline_metric_name, total_val, today_str))
                initial_totals_for_day[baseline_metric_name] = total_val
        conn.commit()

    differences = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}
    delta_percentages = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}
    for side in ['CE', 'PE']:
        for field in fields_to_sum:
            current_val = current_totals[side].get(field, 0.0)
            initial_val_for_diff = initial_totals_for_day.get(f'{side}_{field}_total_baseline', 0.0)
            diff = current_val - initial_val_for_diff
            differences[side][field] = diff
            delta_percentages[side][field] = (diff / initial_val_for_diff * 100.0) if initial_val_for_diff != 0 else 0.0 # Corrected delta % calculation

    ts_for_delta_history = datetime.now(IST).isoformat()
    for side_hist in ['CE', 'PE']:
        for field_hist in fields_to_sum:
            hist_metric_name = f"{side_hist}_{field_hist}_delta_percent"
            hist_metric_value = delta_percentages[side_hist].get(field_hist, 0.0)
            c.execute('INSERT INTO delta_percentage_history (expiry_date, metric_name, metric_value, metric_timestamp) VALUES (?, ?, ?, ?)',
                      (expiry_date, hist_metric_name, hist_metric_value, ts_for_delta_history))
    conn.commit()

    change_in_delta_percentages = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}
    c.execute('''SELECT metric_name, metric_value FROM delta_percentage_history
                 WHERE expiry_date = ? AND metric_timestamp < ? AND metric_name LIKE '%_delta_percent'
                 ORDER BY metric_timestamp DESC''', (expiry_date, ts_for_delta_history))
    latest_previous_delta_percentages = {row[0]: row[1] for row in c.fetchall()} # Simplified to take latest unique

    for side in ['CE', 'PE']:
        for field in fields_to_sum:
            current_delta_pct = delta_percentages[side].get(field, 0.0)
            prev_delta_pct_metric_name = f"{side}_{field}_delta_percent"
            # Get the most recent previous value, not just any
            # The query already orders by timestamp desc, so we need to pick the first one for each metric if multiple exist
            # For simplicity, if the exact previous timestamp isn't matched, this might use an older one.
            # A more robust way would be to fetch specifically the entry just before ts_for_delta_history for each metric.
            # The current `latest_previous_delta_percentages` will overwrite with older values if not careful.
            # Let's refine how `latest_previous_delta_percentages` is built.
            # We need the *single* latest previous value for each metric.
            # The current approach of dict comprehension will get the *oldest* if multiple rows for same metric_name exist.
            # Corrected logic for latest_previous_delta_percentages:
            
            # Re-fetch with GROUP BY to get the latest previous for each metric
            c.execute('''
                SELECT metric_name, metric_value
                FROM delta_percentage_history
                WHERE expiry_date = ? AND metric_timestamp < ? AND metric_name LIKE '%_delta_percent'
                AND metric_timestamp = (
                    SELECT MAX(metric_timestamp)
                    FROM delta_percentage_history AS sub
                    WHERE sub.expiry_date = delta_percentage_history.expiry_date
                      AND sub.metric_name = delta_percentage_history.metric_name
                      AND sub.metric_timestamp < ?
                )
            ''', (expiry_date, ts_for_delta_history, ts_for_delta_history))
            
            # Build a dictionary of the true latest previous values
            grouped_latest_previous_delta_percentages = {row[0]: row[1] for row in c.fetchall()}
            prev_delta_pct_val = grouped_latest_previous_delta_percentages.get(prev_delta_pct_metric_name, 0.0)
            change_in_delta_percentages[side][field] = current_delta_pct - prev_delta_pct_val


    c.execute('DELETE FROM metrics WHERE expiry_date = ?', (expiry_date,))
    metric_timestamp = datetime.now(IST).isoformat()
    for side in ['CE', 'PE']:
       for field in fields_to_sum:
            for m_type, m_val_dict, m_val_key in [
                ('total', current_totals[side], field),
                ('difference', differences[side], field),
                ('delta_percent', delta_percentages[side], field),
                ('chg_delta_percent', change_in_delta_percentages[side], field)
            ]:
                metric_name = f"{side}_{field}_{m_type}"
                metric_value = m_val_dict.get(m_val_key, 0.0)
                c.execute('INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp) VALUES (?, ?, ?, ?)',
                          (expiry_date, metric_name, metric_value, metric_timestamp))
    conn.commit()
    conn.close()
    print(f"Stored calculated metrics for expiry {expiry_date} at {metric_timestamp} for db: {db_file}")

# --- Scheduler Job Functions (from app.py, modified) ---
def scheduled_fetch_and_calculate(expiry_date, db_file_for_job, instrument_key='NSE_INDEX|Nifty 50'):
    # This function is called by the scheduler.
    # It needs to operate outside of a Flask request context, so session is not available here.
    # db_file_for_job is passed when the job is scheduled.
    with scheduler_lock:
        print(f"Scheduler: Running fetch_option_chain_data for {expiry_date} on DB: {db_file_for_job}")
        # We need a way to get a token without session for scheduled jobs.
        # For now, this will fail if get_token() strictly relies on session.
        # A potential solution: store a "system" or "scheduler" token, or use client credentials grant if available.
        # Assuming fetch_option_chain_data can somehow get a token or this job is for a user context where token is pre-set.
        # The current get_token() will return None if no session.
        # This implies scheduled_fetch_and_calculate might only work if a user is logged in AND the db_file corresponds to their role.
        # For simplicity, we keep the original logic. If it needs to run truly headless, token management for scheduler needs rework.
        fetch_option_chain_data(expiry_date, db_file_for_job, instrument_key)

def scheduled_15min_snapshot(expiry_date, db_file_for_job):
    with scheduler_lock:
        if not db_file_for_job:
            print(f"Scheduler (15min_snapshot): No DB file provided for job. Skipping for expiry {expiry_date}.")
            return
        conn = sqlite3.connect(db_file_for_job)
        c = conn.cursor()
        now = datetime.now(IST)
        metric_timestamp = now.isoformat()
        print(f"Scheduler (15min_snapshot): Capturing for {expiry_date}, DB: {db_file_for_job} at {metric_timestamp}")
        for side in ['CE', 'PE']:
            for field in ['openInterest', 'totalTradedVolume']:
                c.execute('SELECT metric_value FROM metrics WHERE expiry_date = ? AND metric_name = ? ORDER BY metric_timestamp DESC LIMIT 1',
                          (expiry_date, f"{side}_{field}_total"))
                row = c.fetchone()
                if row:
                    snapshot_metric_name = f"{side}_{field}_snapshot_{now.strftime('%H%M')}"
                    c.execute('INSERT INTO snapshot_metrics (expiry_date, metric_name, metric_value, metric_timestamp) VALUES (?, ?, ?, ?)',
                              (expiry_date, snapshot_metric_name, row[0], metric_timestamp))
        conn.commit()
        conn.close()

def calculate_15min_deltas(expiry_date, db_file_for_job):
    with scheduler_lock:
        if not db_file_for_job:
            print(f"Scheduler (15min_deltas): No DB file provided for job. Skipping for expiry {expiry_date}.")
            return
        conn = sqlite3.connect(db_file_for_job)
        c = conn.cursor()
        now = datetime.now(IST)
        metric_timestamp = now.isoformat()
        print(f"Scheduler (15min_deltas): Calculating for {expiry_date}, DB: {db_file_for_job} at {metric_timestamp}")
        for side in ['CE', 'PE']:
            for field in ['openInterest', 'totalTradedVolume']:
                c.execute('SELECT metric_value FROM metrics WHERE expiry_date = ? AND metric_name = ? ORDER BY metric_timestamp DESC LIMIT 1',
                          (expiry_date, f"{side}_{field}_total"))
                current_row = c.fetchone()
                if not current_row: continue
                
                snapshot_time = (now - timedelta(minutes=15)).strftime('%H%M')
                snapshot_metric_name = f"{side}_{field}_snapshot_{snapshot_time}"
                c.execute('SELECT metric_value FROM snapshot_metrics WHERE expiry_date = ? AND metric_name = ? ORDER BY metric_timestamp DESC LIMIT 1',
                          (expiry_date, snapshot_metric_name))
                snapshot_row = c.fetchone()
                if not snapshot_row: continue

                delta_value = current_row[0] - snapshot_row[0]
                delta_metric_name = f"{side}_{field}_15min_delta"
                c.execute('INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp) VALUES (?, ?, ?, ?)',
                          (expiry_date, delta_metric_name, delta_value, metric_timestamp))
        conn.commit()
        conn.close()

def send_metrics_to_ml_for_training(expiry_date): # Specifically uses EMPEROR_DB_FILE
    """Collects latest metrics from EMPEROR_DB and sends to ML model for training."""
    if not EMPEROR_DB_FILE:
        print("Scheduler (send_metrics_to_ml): EMPEROR_DB_FILE not configured. Skipping.")
        return

    with scheduler_lock:
        conn = sqlite3.connect(EMPEROR_DB_FILE)
        try:
            c = conn.cursor()
            c.execute('SELECT metric_name, metric_value FROM metrics WHERE expiry_date = ?', (expiry_date,))
            latest_metrics = {row[0]: row[1] for row in c.fetchall()}
        finally:
            conn.close()

        if not latest_metrics:
            print(f"Scheduler (send_metrics_to_ml): No latest metrics for {expiry_date} in {EMPEROR_DB_FILE}.")
            return

        print(f"Scheduler (send_metrics_to_ml): Sending snapshot for {expiry_date} from {EMPEROR_DB_FILE} to ML model.")
        # Store raw snapshot in ML DB
        snapshot_timestamp = datetime.now(IST).isoformat()
        conn_ml = sqlite3.connect(ML_DB_FILE)
        c_ml = conn_ml.cursor()
        c_ml.execute('INSERT INTO training_snapshots (expiry_date, snapshot_timestamp, metrics_data) VALUES (?, ?, ?)',
                     (expiry_date, snapshot_timestamp, json.dumps(latest_metrics)))
        conn_ml.commit()
        conn_ml.close()
        
        # Call ML model's training method directly
        ml_model.train_incremental(latest_metrics)
        print(f"Sent 15-min metrics snapshot for {expiry_date} to ML model for training.")


def daily_cleanup_job():
    with scheduler_lock:
        today_str = datetime.now(IST).date().isoformat()
        db_files_to_clean = [EMPEROR_DB_FILE, KING_DB_FILE]
        print(f"Starting daily cleanup for baselines older than {today_str}...")

        for db_file in db_files_to_clean:
            if not db_file:
                print(f"Skipping cleanup for unconfigured DB file.")
                continue
            
            print(f"Running daily cleanup for baselines in {db_file}...")
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                c = conn.cursor()
                # Clean baseline_metrics by metric_date (trading date)
                c.execute('DELETE FROM baseline_metrics WHERE metric_date < ?', (today_str,))
                # Also clean very old data from other tables by expiry_date
                cutoff_expiry_date = (datetime.now(IST) - timedelta(days=30)).date().isoformat() # e.g., 30 days
                tables_to_clean_by_expiry = ['option_chain_data', 'metrics', 'snapshot_metrics', 'delta_percentage_history']
                for table_name in tables_to_clean_by_expiry:
                     c.execute(f'DELETE FROM {table_name} WHERE expiry_date < ?', (cutoff_expiry_date,))
                conn.commit()
                print(f"Daily cleanup of baselines and old data in {db_file} completed.")
            except sqlite3.Error as e:
                print(f"SQLite error during daily cleanup for {db_file}: {e}")
            except Exception as e:
                print(f"Unexpected error during daily cleanup for {db_file}: {e}")
            finally:
                if conn:
                    conn.close()
        print(f"Daily cleanup process finished.")


def start_schedulers(expiry_date):
    with scheduler_lock:
        current_db_file = get_db_file() # Get DB file based on current user's session
        if not current_db_file:
            print("Cannot start schedulers: User context (DB file) not determined.")
            return

        stop_schedulers() # Stop any existing jobs first
        
        # This job will fetch data for the DB determined by the user who starts it
        scheduler.add_job(scheduled_fetch_and_calculate, 'interval', seconds=5, 
                          args=[expiry_date, current_db_file], id='fetch_calc_job', replace_existing=True)
        
        # These jobs also run in the context of the user who started them (due to get_db_file() inside them)
        # This might need adjustment if they are meant to be system-wide or for a specific DB always.
        # Pass current_db_file to these jobs as well.
        scheduler.add_job(scheduled_15min_snapshot, 'cron', minute='0,15,30,45', args=[expiry_date, current_db_file], 
                          id='snapshot_job', replace_existing=True)
        scheduler.add_job(calculate_15min_deltas, 'cron', minute='1,16,31,46', args=[expiry_date, current_db_file], 
                          id='delta_calc_job', replace_existing=True)

        # This job is specific to Emperor data for ML training
        if session.get('user_role') == 'emperor': # Only emperor should trigger ML training data collection
            scheduler.add_job(send_metrics_to_ml_for_training, 'interval', minutes=15, args=[expiry_date], 
                              id='ml_train_job', replace_existing=True)

        if not scheduler.get_job('daily_cleanup'):
            scheduler.add_job(daily_cleanup_job, 'cron', hour=0, minute=5, id='daily_cleanup', replace_existing=False) # Run at 00:05 IST

        if not scheduler.running:
            scheduler.start()
        print(f"Schedulers started for expiry {expiry_date}, DB context: {current_db_file}")


def stop_schedulers():
    with scheduler_lock:
        job_ids = ['fetch_calc_job', 'ml_train_job', 'snapshot_job', 'delta_calc_job']
        for job_id in job_ids:
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
        print("Schedulers (fetch, ml_train, snapshot, delta_calc) stopped.")
        # Don't shutdown scheduler if daily_cleanup job exists and should persist
        # scheduler.shutdown(wait=False) # Only if all jobs are meant to be temporary

# --- Flask Routes ---
@flask_app.route('/')
def index():
    session.pop('user_role', None)
    return render_template('login.html')

@flask_app.route('/login/<role>')
def login_with_role(role):
    if role not in ['emperor', 'king']:
        return jsonify({'error': 'Invalid role specified'}), 400
    session['user_role'] = role
    # Initialize DB for this role if not already done (idempotent)
    # init_db(get_db_file()) # Tables should already exist from startup init
    return redirect(url_for('login'))

@flask_app.route('/login')
def login():
    user_role = session.get('user_role')
    if not user_role: return redirect(url_for('index'))
    client_id = EMPEROR_CLIENT_ID if user_role == 'emperor' else KING_CLIENT_ID
    params = {'client_id': client_id, 'redirect_uri': REDIRECT_URI, 'response_type': 'code'}
    return redirect(f"{AUTH_URL}?{urlencode(params)}")

@flask_app.route('/manual_auth', methods=['POST'])
def manual_auth():
    data = request.get_json()
    if not data or 'code' not in data: return jsonify({'error': 'Authorization code is required'}), 400
    
    user_role = session.get('user_role')
    if not user_role: return jsonify({'error': 'User role not set. Please login again.'}), 401

    client_id = EMPEROR_CLIENT_ID if user_role == 'emperor' else KING_CLIENT_ID
    client_secret = EMPEROR_CLIENT_SECRET if user_role == 'emperor' else KING_CLIENT_SECRET
    db_file = get_db_file()

    post_data = {'code': data['code'], 'client_id': client_id, 'client_secret': client_secret,
                 'redirect_uri': REDIRECT_URI, 'grant_type': 'authorization_code'}
    try:
        response = requests.post(TOKEN_URL, data=post_data)
        response.raise_for_status()
        token_data = response.json()
        if isinstance(token_data, str): token_data = json.loads(token_data)
    except Exception as e:
        return jsonify({'error': f'Failed to get access token: {e}, Response: {response.text if "response" in locals() else "N/A"}'}), 500

    if not isinstance(token_data, dict) or 'access_token' not in token_data:
        return jsonify({'error': f'Access token not found: {token_data}'}), 500

    store_token(token_data['access_token'], token_data.get('refresh_token'),
                int(token_data.get('expires_in', 86400)), db_file)
    return jsonify({'message': 'Authentication successful'})

@flask_app.route('/dashboard')
def dashboard():
    user_role = session.get('user_role')
    if not user_role: return redirect(url_for('index'))
    # Example: Start schedulers when dashboard is accessed for a specific expiry
    # This would typically be triggered by a user action on the dashboard (e.g., selecting an expiry and clicking "Start Monitoring")
    # For now, let's assume an expiry_date is passed or defaulted for demonstration
    # default_expiry = (datetime.now(IST) + timedelta(days=1)).strftime('%Y-%m-%d') # Placeholder
    # start_schedulers(request.args.get('expiry_date', default_expiry))
    return render_template('dashboard.html', user_role=user_role)

@flask_app.route('/start_monitoring', methods=['POST'])
def start_monitoring_endpoint():
    user_role = session.get('user_role')
    if not user_role: return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    expiry_date = data.get('expiry_date')
    if not expiry_date: return jsonify({'error': 'expiry_date is required'}), 400
    
    start_schedulers(expiry_date)
    return jsonify({'message': f'Monitoring started for {expiry_date}'})

@flask_app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring_endpoint():
    user_role = session.get('user_role')
    if not user_role: return jsonify({'error': 'Not logged in'}), 401
    stop_schedulers()
    return jsonify({'message': 'Monitoring stopped'})


@flask_app.route('/token')
def token_info():
    access_token, refresh_token, expires_at = get_token()
    if access_token:
        return jsonify({'access_token': access_token, 'refresh_token': refresh_token, 
                        'expires_at': expires_at.isoformat() if expires_at else None})
    return jsonify({'error': 'No valid token found'}), 404

@flask_app.route('/fetch_data') # Manual fetch trigger
def fetch_data_endpoint():
    expiry_date = request.args.get('expiry_date')
    db_file = get_db_file()
    if not db_file: return redirect(url_for('index'))
    if not expiry_date: return jsonify({'error': 'expiry_date query parameter is required'}), 400
    
    fetch_option_chain_data(expiry_date, db_file)
    return jsonify({'message': f'Manual fetch triggered for expiry {expiry_date} on DB {db_file}'})

@flask_app.route('/option_chain')
def get_option_chain_data_for_frontend():
    expiry_date = request.args.get('expiry_date')
    if not expiry_date: return jsonify({'error': 'expiry_date query parameter is required'}), 400

    db_file = get_db_file()
    if not db_file: return redirect(url_for('index'))
    
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT raw_data FROM option_chain_data WHERE expiry_date = ? ORDER BY strike_price ASC', (expiry_date,))
    option_chain_list = [json.loads(row[0]) for row in c.fetchall()]

    latest_calculated_metrics = {}
    metric_fields = ['bidQty', 'askQty', 'openInterest', 'totalTradedVolume', 'IV', 'delta']
    metric_types = ['total', 'difference', 'delta_percent', 'chg_delta_percent', '15min_delta']
    for side in ['CE', 'PE']:
        latest_calculated_metrics[side] = {}
        for field in metric_fields:
            latest_calculated_metrics[side][field] = {}
            for m_type in metric_types:
                metric_name_to_query = f"{side}_{field}_{m_type}"
                c.execute('SELECT metric_value FROM metrics WHERE expiry_date = ? AND metric_name = ? ORDER BY metric_timestamp DESC LIMIT 1',
                          (expiry_date, metric_name_to_query))
                row = c.fetchone()
                latest_calculated_metrics[side][field][m_type] = row[0] if row else None
    conn.close()
    return jsonify({'option_chain': option_chain_list, 'calculated_metrics': latest_calculated_metrics})

# --- ML Endpoints (formerly in ml_backend.py) ---
@flask_app.route('/ml/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    if not data or 'live_metrics' not in data or 'user_prompt' not in data:
        return jsonify({'error': 'Invalid input. Requires live_metrics and user_prompt.'}), 400
    
    # Direct call to the ML model's method
    prediction_response = ml_model.predict(data['live_metrics'], data['user_prompt'])
    print(f"Sending prediction to frontend: {json.dumps(prediction_response)}")
    return jsonify(prediction_response)

@flask_app.route('/ml/train_snapshot', methods=['POST']) # This endpoint is for external calls if any, internal calls use direct method
def train_snapshot_endpoint():
    data = request.get_json()
    if not data or 'expiry_date' not in data or 'snapshot_timestamp' not in data or 'metrics_data' not in data:
        return jsonify({'error': 'Invalid input. Requires expiry_date, snapshot_timestamp, and metrics_data.'}), 400

    # Store raw snapshot
    conn = sqlite3.connect(ML_DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO training_snapshots (expiry_date, snapshot_timestamp, metrics_data) VALUES (?, ?, ?)',
              (data['expiry_date'], data['snapshot_timestamp'], json.dumps(data['metrics_data'])))
    conn.commit()
    conn.close()
    
    # Direct call to the ML model's method
    ml_model.train_incremental(data['metrics_data'])
    return jsonify({'status': 'success', 'message': 'Snapshot received and used for training.'})

@flask_app.route('/ml/feedback', methods=['POST'])
def feedback_endpoint():
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({'error': 'Invalid input. Requires feedback.'}), 400
    
    # Direct call to the ML model's method
    ml_model.process_feedback(data['feedback'])
    return jsonify({'status': 'success', 'message': 'Feedback received and processed.'})

# --- Chat Endpoints (modified from app.py to call ML model directly) ---
@flask_app.route('/ml-chatbot')
def ml_chatbot_page():
    user_role = session.get('user_role')
    if not user_role: return redirect(url_for('index'))
    return render_template('ml-chatbot.html') # Pass user_role if template needs it

@flask_app.route('/chat/prompt', methods=['POST'])
def handle_chat_prompt():
    data = request.get_json()
    if not data or 'expiry_date' not in data or 'user_prompt' not in data:
        return jsonify({'error': 'Invalid input. Requires expiry_date and user_prompt.'}), 400

    db_file = get_db_file()
    if not db_file: return jsonify({'error': 'User role not set. Cannot predict.'}), 401

    expiry_date = data['expiry_date']
    user_prompt = data['user_prompt']

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT metric_name, metric_value FROM metrics WHERE expiry_date = ?', (expiry_date,))
    live_metrics = {row[0]: row[1] for row in c.fetchall()}
    
    # Fetch spot price and high OI strikes
    c.execute("SELECT raw_data FROM option_chain_data WHERE expiry_date = ? ORDER BY strike_price ASC", (expiry_date,))
    parsed_strikes_for_oi = [json.loads(row[0]) for row in c.fetchall()]
    conn.close()

    if parsed_strikes_for_oi:
        spot_price_raw = parsed_strikes_for_oi[0].get('underlying_spot_price')
        try: live_metrics['price'] = float(spot_price_raw) if spot_price_raw is not None else None
        except (ValueError, TypeError): live_metrics['price'] = None
        
        call_oi_data = sorted([{'strike': s['strike_price'], 'oi': s.get('call_options', {}).get('market_data', {}).get('oi', 0)} 
                               for s in parsed_strikes_for_oi if s.get('call_options', {}).get('market_data', {}).get('oi') is not None],
                              key=lambda x: x['oi'], reverse=True)
        put_oi_data = sorted([{'strike': s['strike_price'], 'oi': s.get('put_options', {}).get('market_data', {}).get('oi', 0)}
                              for s in parsed_strikes_for_oi if s.get('put_options', {}).get('market_data', {}).get('oi') is not None],
                             key=lambda x: x['oi'], reverse=True)
        live_metrics['high_call_oi_strikes'] = [item['strike'] for item in call_oi_data[:3]]
        live_metrics['high_put_oi_strikes'] = [item['strike'] for item in put_oi_data[:3]]

    if not live_metrics:
         return jsonify({'error': f'No live metrics for expiry {expiry_date}. Cannot predict.'}), 404

    # Direct call to ML model
    try:
        predictions = ml_model.predict(live_metrics, user_prompt)
        return jsonify(predictions)
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return jsonify({'error': f'Error getting prediction from ML model: {e}'}), 500

@flask_app.route('/chat/feedback', methods=['POST'])
def handle_chat_feedback():
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({'error': 'Invalid input. Requires feedback data.'}), 400

    user_role = session.get('user_role')
    if user_role != 'emperor':
        print(f"Feedback from role '{user_role}' noted, but not used for model training.")
        return jsonify({'status': 'success', 'message': 'Feedback noted (not used for training).'})

    # Direct call to ML model
    try:
        ml_model.process_feedback(data['feedback'])
        return jsonify({'status': 'success', 'message': 'Feedback sent to ML model.'})
    except Exception as e:
        print(f"Error sending feedback to ML model: {e}")
        return jsonify({'error': f'Error sending feedback to ML model: {e}'}), 500

# --- ASGI Wrapper ---
# The variable 'app' is what Gunicorn/Uvicorn will look for.
app = WsgiToAsgi(flask_app)

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure daily cleanup job is scheduled if not already
    if not scheduler.get_job('daily_cleanup') and scheduler.running: # Add if scheduler is running but job is missing
         scheduler.add_job(daily_cleanup_job, 'cron', hour=0, minute=5, id='daily_cleanup', replace_existing=False, timezone=IST)
    elif not scheduler.get_job('daily_cleanup') and not scheduler.running: # Add if scheduler not running
         scheduler.add_job(daily_cleanup_job, 'cron', hour=0, minute=5, id='daily_cleanup', replace_existing=False, timezone=IST)
         # scheduler.start() # Start only if other jobs are also expected to run from startup without user interaction

    # For local development, you might want to start the scheduler if it has jobs.
    # However, `start_schedulers` is typically called after user login for specific expiries.
    # The `daily_cleanup_job` is an exception as it's system-wide.
    if not scheduler.running and scheduler.get_job('daily_cleanup'):
        print("Starting scheduler for daily cleanup job...")
        scheduler.start()
        
    print("Starting Flask development server on http://127.0.0.1:5000/")
    # Use flask_app for Flask's built-in dev server, not the ASGI wrapped 'app'
    flask_app.run(debug=True, port=5000)

