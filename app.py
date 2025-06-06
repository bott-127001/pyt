import os
import sqlite3
from datetime import datetime, timedelta, timezone
from flask import Flask, redirect, request, jsonify, render_template, url_for
from flask import session # Import session
from asgiref.wsgi import WsgiToAsgi
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import requests # Import requests for inter-service communication


# Original Flask app instance
flask_app = Flask(__name__)

# Configure a secret key for sessions
flask_app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_session_management') # Replace with a strong, random key

# Configuration - replace with your actual Upstox API credentials and redirect URI
EMPEROR_CLIENT_ID = os.environ.get('EMPEROR_CLIENT_ID', '6a4f686a-5933-4d96-a772-e5de90b8417c')
EMPEROR_CLIENT_SECRET = os.environ.get('EMPEROR_CLIENT_SECRET', 'x3tyejmpgj')
# Use environment variables for DB file paths, with defaults
# Example for Render: Set these to /data/emperor_tokens.db if using a disk mounted at /data
EMPEROR_DB_FILE = os.environ.get('EMPEROR_DB_FILE', 'emperor_tokens.db')

KING_CLIENT_ID = os.environ.get('KING_CLIENT_ID', '192ea5d0-d3ca-407d-8651-045bd923b3a2')
KING_CLIENT_SECRET = os.environ.get('KING_CLIENT_SECRET', 'lmg6tg1m6k')
KING_DB_FILE = os.environ.get('KING_DB_FILE', 'king_tokens.db')


REDIRECT_URI = os.environ.get('UPSTOX_REDIRECT_URI', 'https://www.google.co.in/')  # Must match Upstox app settings

# Upstox OAuth URLs
AUTH_URL = 'https://api.upstox.com/v2/login/authorization/dialog'
TOKEN_URL = 'https://api.upstox.com/v2/login/authorization/token'

# ML Backend URL (assuming it runs on localhost:5001)
ML_BACKEND_URL = os.environ.get('ML_BACKEND_URL', 'http://127.0.0.1:5001/ml')

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))

def get_db_file():
    """Determine the database file based on the user role in the session."""
    user_role = session.get('user_role')
    if user_role == 'emperor':
        return EMPEROR_DB_FILE
    elif user_role == 'king':
        return KING_DB_FILE
    return None # Should not happen if login flow is followed

def init_db(db_file):
    """Initialize a specific SQLite database file and create tables if not exists."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Enable WAL mode for better concurrency
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
    # Create option_chain_data table
    c.execute('''
        CREATE TABLE IF NOT EXISTS option_chain_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            strike_price REAL NOT NULL,
            raw_data TEXT NOT NULL,
            fetch_timestamp TEXT NOT NULL
        )
    ''')
     # Create metrics table to store ONLY THE LATEST calculated metrics per expiry_date
    c.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL
        )
    ''')
    # Create snapshot_metrics table for 15-minute snapshots
    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshot_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL, -- e.g., CE_openInterest_snapshot_0930
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL -- Timestamp of this specific calculation
        )
    ''')
    # Create baseline_metrics table for first snapshot of the day
    c.execute('''
        CREATE TABLE IF NOT EXISTS baseline_metrics (
            expiry_date TEXT NOT NULL, -- Date for which the option expires
            metric_name TEXT NOT NULL, -- e.g., CE_openInterest_total_baseline
            metric_value REAL NOT NULL, -- The actual baseline value
            metric_date TEXT NOT NULL, -- The trading date for which this baseline is valid
            UNIQUE(expiry_date, metric_name, metric_date) -- Ensure one baseline per metric per day
        )
    ''')
    # Create delta_percentage_history table to store historical delta percentages for chg_delta_percent calculation
    c.execute('''
        CREATE TABLE IF NOT EXISTS delta_percentage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            metric_name TEXT NOT NULL, -- e.g., CE_openInterest_delta_percent
            metric_value REAL NOT NULL,
            metric_timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
# Initialize both databases on startup (runs in production)
init_db(EMPEROR_DB_FILE)
init_db(KING_DB_FILE)

def store_token(access_token, refresh_token, expires_in_seconds, db_file):
    """Store the new access token in the database and delete old tokens."""
    expires_at = datetime.now(IST) + timedelta(seconds=expires_in_seconds)
    created_at = datetime.now(IST) # This was missing in the provided context, but should be here
    conn = sqlite3.connect(db_file) # Use the passed db_file argument
    c = conn.cursor()
    # Delete old tokens
    c.execute('DELETE FROM tokens')
    # Insert new token
    c.execute('''
        INSERT INTO tokens (access_token, refresh_token, expires_at, created_at)
        VALUES (?, ?, ?, ?)
    ''', (access_token, refresh_token, expires_at.isoformat(), created_at.isoformat()))
    conn.commit()
    conn.close()

# Scheduler for periodic fetching
scheduler = BackgroundScheduler()
scheduler_lock = threading.Lock()
fetch_job = None

def scheduled_fetch_and_calculate(expiry_date, instrument_key='NSE_INDEX|Nifty 50'):
    with scheduler_lock:
        fetch_option_chain_data(expiry_date, get_db_file(), instrument_key) # Pass db_file

def scheduled_15min_snapshot(expiry_date):
    with scheduler_lock:
        conn = sqlite3.connect(get_db_file()) # Use the correct db_file
        c = conn.cursor()
        # Capture snapshot of OI and Volume totals at current time
        now = datetime.now(IST) # This was missing in the provided context, but should be here
        metric_timestamp = now.isoformat()

        # Fetch latest totals for OI and Volume
        fields = ['openInterest', 'totalTradedVolume']
        sides = ['CE', 'PE']
        for side in sides:
            for field in fields:
                # Get latest metric value for total
                c.execute('''
                    SELECT metric_value FROM metrics
                    WHERE expiry_date = ? AND metric_name = ? AND metric_timestamp <= ?
                    ORDER BY metric_timestamp DESC LIMIT 1
                ''', (expiry_date, f"{side}_{field}_total", metric_timestamp))
                row = c.fetchone()
                if row:
                    value = row[0]
                    # Store snapshot with special metric name including timestamp
                    snapshot_metric_name = f"{side}_{field}_snapshot_{now.strftime('%H%M')}"
                    c.execute('''
                        INSERT INTO snapshot_metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (expiry_date, snapshot_metric_name, value, metric_timestamp))
        conn.commit()
        conn.close()

def calculate_15min_deltas(expiry_date):
    with scheduler_lock:
        conn = sqlite3.connect(get_db_file()) # Use the correct db_file
        c = conn.cursor()
        now = datetime.now(IST) # This was missing in the provided context, but should be here
        metric_timestamp = now.isoformat()
        fields = ['openInterest', 'totalTradedVolume']
        sides = ['CE', 'PE']

        # Calculate delta = current total - total at T (15 minutes ago)
        for side in sides:
            for field in fields:
                # Get current total
                c.execute('''
                    SELECT metric_value FROM metrics
                    WHERE expiry_date = ? AND metric_name = ?
                    ORDER BY metric_timestamp DESC LIMIT 1
                ''', (expiry_date, f"{side}_{field}_total"))
                current_row = c.fetchone()
                if not current_row:
                    continue
                current_value = current_row[0]

                # Get snapshot from 15 minutes ago (approximate)
                snapshot_time = (now - timedelta(minutes=15)).strftime('%H%M')
                snapshot_metric_name = f"{side}_{field}_snapshot_{snapshot_time}"
                c.execute('''
                    SELECT metric_value FROM snapshot_metrics
                    WHERE expiry_date = ? AND metric_name = ?
                    ORDER BY metric_timestamp DESC LIMIT 1
                ''', (expiry_date, snapshot_metric_name))
                snapshot_row = c.fetchone()
                if not snapshot_row:
                    continue
                snapshot_value = snapshot_row[0]

                delta_value = current_value - snapshot_value

                # Store 15min delta metric
                delta_metric_name = f"{side}_{field}_15min_delta" # This metric name seems unused in ML backend currently
                c.execute('''
                    INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (expiry_date, delta_metric_name, delta_value, metric_timestamp)) # Storing 15min delta difference
        conn.commit()
        conn.close()

def start_schedulers(expiry_date):
    with scheduler_lock:
        # Add daily cleanup jobs if they don't exist (ensures they are set up in deployed env)
        # The daily_cleanup_job function now iterates through known DB files internally
        if not scheduler.get_job('daily_cleanup'):
             scheduler.add_job(daily_cleanup_job, 'cron', hour=0, minute=0, timezone='Asia/Kolkata', id='daily_cleanup', replace_existing=False)
        # Ensure schedulers are stopped before starting new ones
        stop_schedulers()
        # Schedule 5-second fetch and calculation job
        # Pass the current DB file to the scheduled job
        scheduler.add_job(scheduled_fetch_and_calculate, 'interval', seconds=5, args=[expiry_date, get_db_file()], id='fetch_calc_job', replace_existing=True)
        # Schedule a new job to collect ALL metrics every 15 minutes and send to ML backend
        scheduler.add_job(send_metrics_to_ml_backend, 'interval', minutes=15, args=[expiry_date], id='ml_train_job', replace_existing=True)
        # Schedule 15-minute snapshot job (e.g., at 0, 15, 30, 45 minutes past the hour)
        scheduler.add_job(scheduled_15min_snapshot, 'cron', minute='0,15,30,45', args=[expiry_date], id='snapshot_job', replace_existing=True, timezone=IST)
        # Schedule 15-minute delta calculation job (e.g., 1 minute after snapshots, or adjust as needed)
        scheduler.add_job(calculate_15min_deltas, 'cron', minute='1,16,31,46', args=[expiry_date], id='delta_calc_job', replace_existing=True, timezone=IST)
        scheduler.start()

def stop_schedulers():
    with scheduler_lock:
        # Check if jobs exist before removing
        if scheduler.get_job('fetch_calc_job'):
            scheduler.remove_job('fetch_calc_job')
        # Only remove the ML training job if it exists (it might not for 'king')
        if scheduler.get_job('ml_train_job'):
             scheduler.remove_job('ml_train_job')
        # The old snapshot and delta jobs are no longer needed
        if scheduler.get_job('snapshot_job'):
            scheduler.remove_job('snapshot_job')
        if scheduler.get_job('delta_calc_job'):
            scheduler.remove_job('delta_calc_job')
        # Only shutdown if there are no other jobs (e.g., cleanup jobs if added later)
        if not scheduler.get_jobs():
             scheduler.shutdown(wait=False)

def daily_cleanup_job():
    """Scheduled job to run at midnight IST to clear daily baselines."""
    with scheduler_lock: # Ensure thread safety if other jobs are running
        today_str = datetime.now(IST).date().isoformat()

        # Iterate through all known DB files for cleanup
        db_files_to_clean = [EMPEROR_DB_FILE, KING_DB_FILE]

        for db_file in db_files_to_clean:
            conn = sqlite3.connect(db_file)
            c = conn.cursor()
            print(f"Running daily cleanup for baselines in {db_file} older than {today_str}...")
        print(f"Running daily cleanup for baselines older than {today_str}...")

        # Clear baseline_metrics for all expiry_dates that are for previous trading days.
        # This ensures that on the new day, a fresh baseline will be created.
        c.execute('DELETE FROM baseline_metrics WHERE metric_date < ?', (today_str,))
        # Optionally, you might want to clear other daily accumulated data if needed,
        # for example, if 'metrics' table contains daily aggregates that should reset.
        # For now, only resetting baselines as per primary requirement.
        conn.commit()
        conn.close()
        print(f"Daily cleanup of baselines in {db_file} completed.")



def send_metrics_to_ml_backend(expiry_date):
    """Collects all latest metrics and sends them to the ML backend for training."""
    # This job now ONLY collects metrics from the Emperor database.
    # The ML model should be trained on Emperor data only.
    conn = sqlite3.connect(EMPEROR_DB_FILE)  # Use EMPEROR_DB_FILE directly
    try:
        c = conn.cursor()
        # Fetch ALL latest metrics for the given expiry date
        c.execute('''
            SELECT metric_name, metric_value FROM metrics
            WHERE expiry_date = ?
        ''', (expiry_date,))
        latest_metrics = {row[0]: row[1] for row in c.fetchall()}
    finally:
        conn.close()

    if not latest_metrics:

        print(f"No latest metrics found for expiry {expiry_date} to send to ML backend.")
        return

    snapshot_timestamp = datetime.now(IST).isoformat()
    payload = {'expiry_date': expiry_date, 'snapshot_timestamp': snapshot_timestamp, 'metrics_data': latest_metrics}

    try:
        response = requests.post(f'{ML_BACKEND_URL}/train_snapshot', json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Sent 15-min metrics snapshot for {expiry_date} to ML backend. Response: {response.json()}")
    except requests.RequestException as e:
        print(f"Error sending 15-min metrics snapshot to ML backend: {e}")

def fetch_option_chain_data(expiry_date, db_file, instrument_key='NSE_INDEX|Nifty 50'):
    """Fetch option chain data from Upstox API for the given expiry date and store in DB."""
    import json
    # get_token needs to be updated to accept db_file or use the session
    access_token, _, _ = get_token()
    if not access_token:
        print("No valid access token found. Cannot fetch option chain data.")
        return

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    url = f'https://api.upstox.com/v2/option/chain?instrument_key={instrument_key}&expiry_date={expiry_date}'

    import json
    try:
        print(f"Fetching option chain data from API for expiry {expiry_date} and instrument {instrument_key}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        try:
            data = response.json()
            # Defensive: if data is string, parse again
            if isinstance(data, str):
                data = json.loads(data)
        except Exception:
            # Fallback parse raw text
            data = json.loads(response.text)
        print(f"API response data keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error fetching option chain data: {e}")
        return

    print(f"Fetched {len(data.get('data', []))} option chain items for expiry {expiry_date}")

    # Clear old data for this expiry_date before inserting new data
    conn = sqlite3.connect(db_file) # Use the provided db_file
    c = conn.cursor()
    c.execute('DELETE FROM option_chain_data WHERE expiry_date = ?', (expiry_date,))

    # Cleanup old expiry dates data older than 7 days
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now(IST) - timedelta(days=7)).date().isoformat()
    c.execute('DELETE FROM option_chain_data WHERE expiry_date < ?', (cutoff_date,))
    c.execute('DELETE FROM metrics WHERE expiry_date < ?', (cutoff_date,)) # 7-day cleanup for latest metrics
    c.execute('DELETE FROM snapshot_metrics WHERE expiry_date < ?', (cutoff_date,)) # 7-day cleanup for snapshots
    c.execute('DELETE FROM delta_percentage_history WHERE expiry_date < ?', (cutoff_date,)) # 7-day cleanup for delta history
    c.execute('DELETE FROM baseline_metrics WHERE metric_date < ?', (cutoff_date,)) # Clean baseline by trading date

    # Insert new data as raw JSON string per strike price
    fetch_timestamp = datetime.now(IST).isoformat()
    inserted_count = 0
    for item in data.get('data', []):
        strike_price = item.get('strike_price') or item.get('strikePrice')
        raw_json = json.dumps(item)
        c.execute('''
            INSERT INTO option_chain_data (
                expiry_date, strike_price, raw_data, fetch_timestamp
            ) VALUES (?, ?, ?, ?)
        ''', (
            expiry_date,
            strike_price,
            raw_json,
            fetch_timestamp
        ))
        inserted_count += 1
    conn.commit()

    # Get underlyingValue (spot price) from the first item in the 'data' array of the API response
    # as per Upstox documentation, 'underlying_spot_price' is within each element of the 'data' list.
    spot_price = None
    option_data_list = data.get('data', [])
    if option_data_list:
        # Assuming underlying_spot_price is consistent across all items for a single fetch
        spot_price = option_data_list[0].get('underlying_spot_price')
    try:
        if spot_price is None:
            print("Could not get 'underlying_spot_price' from option chain data. Defaulting to 0.")
            spot_price = 0
        else:
            spot_price = float(spot_price) # Ensure it's a float
    except (ValueError, TypeError) as e:
        print(f"Error processing 'underlying_spot_price': {e}. Defaulting to 0.")
        spot_price = 0

    # Load all raw option chain data (json strings) for expiry_date from DB
    c.execute('SELECT raw_data FROM option_chain_data WHERE expiry_date = ?', (expiry_date,)) # Use the provided db_file
    db_rows = c.fetchall()
    parsed_option_chain_strikes = []
    for row_item in db_rows:
        try:
            parsed_option_chain_strikes.append(json.loads(row_item[0]))
        except Exception as e:
            print(f"Error parsing strike data from DB: {e}")
            continue

    # Determine ATM strike (closest to spot_price)
    atm_strike = None
    min_diff = float('inf')
    for strike_obj in parsed_option_chain_strikes:
        current_strike_price = strike_obj.get('strike_price')
        if current_strike_price is not None:
            diff = abs(current_strike_price - spot_price)
            if diff < min_diff:
                min_diff = diff
                atm_strike = current_strike_price

    if atm_strike is None and parsed_option_chain_strikes: # Fallback if spot price is weird
        atm_strike = parsed_option_chain_strikes[len(parsed_option_chain_strikes)//2].get('strike_price', 0)

    # --- Baseline Metrics (Initial Totals for the Day) ---
    today_str = datetime.now(IST).date().isoformat()
    initial_totals_for_day = {}
    is_baseline_set_for_today = False

    c.execute('''
        SELECT metric_name, metric_value FROM baseline_metrics
        WHERE expiry_date = ? AND metric_date = ?
    ''', (expiry_date, today_str))
    baseline_rows = c.fetchall()

    if baseline_rows:
        is_baseline_set_for_today = True
        for bl_metric_name, bl_metric_value in baseline_rows:
            initial_totals_for_day[bl_metric_name] = bl_metric_value
    # --- End Baseline Metrics ---

    # --- ATM/OTM Filtering and preparing data for summation ---
    filtered_options_for_summation = {'CE': [], 'PE': []}
    for strike_obj in parsed_option_chain_strikes:
        strike_price_val = strike_obj.get('strike_price')
        if strike_price_val is None:
            continue
        ce_option_data = strike_obj.get('call_options')
        pe_option_data = strike_obj.get('put_options')

        if strike_price_val == atm_strike:
            if ce_option_data: filtered_options_for_summation['CE'].append(ce_option_data)
            if pe_option_data: filtered_options_for_summation['PE'].append(pe_option_data)
        elif strike_price_val < atm_strike:  # OTM for PE
            if pe_option_data: filtered_options_for_summation['PE'].append(pe_option_data)
        elif strike_price_val > atm_strike:  # OTM for CE
            if ce_option_data: filtered_options_for_summation['CE'].append(ce_option_data)
    # --- End ATM/OTM Filtering ---

    # --- Calculate Current Totals ---
    fields_to_sum = ['bidQty', 'askQty', 'openInterest', 'totalTradedVolume', 'IV', 'delta']
    field_key_map = { # Maps generic field name to path in Upstox option data object
        'bidQty': ['market_data', 'bid_qty'],
        'askQty': ['market_data', 'ask_qty'],
        'openInterest': ['market_data', 'oi'],
        'totalTradedVolume': ['market_data', 'volume'], # Corrected key
        'IV': ['option_greeks', 'iv'],
        'delta': ['option_greeks', 'delta']
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
                    val = value_container # This should be the final value
                    if val is not None and not isinstance(val, (dict, str)):
                        current_totals[side][field_name] += float(val)
                except (TypeError, ValueError) as e:
                    # print(f"Skipping value for {side} {field_name} due to error: {e}, value: {val}")
                    pass # Silently skip if a value is not summable
    # --- End Calculate Current Totals ---

    # --- Store Baseline if not set for today, and populate initial_totals_for_day ---
    if not is_baseline_set_for_today and parsed_option_chain_strikes: # Only set baseline if data exists
        for side in ['CE', 'PE']:
            for field, total_val in current_totals[side].items():
                baseline_metric_name = f"{side}_{field}_total_baseline"
                c.execute('''
                    INSERT OR IGNORE INTO baseline_metrics (expiry_date, metric_name, metric_value, metric_date)
                    VALUES (?, ?, ?, ?)
                ''', (expiry_date, baseline_metric_name, total_val, today_str))
                initial_totals_for_day[baseline_metric_name] = total_val # Populate for current run
        conn.commit()
        is_baseline_set_for_today = True
    # --- End Store Baseline ---

    # --- Calculate Differences and Delta Percentages using Baseline ---
    differences = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}
    delta_percentages = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}

    for side in ['CE', 'PE']:
        for field in fields_to_sum:
            current_val = current_totals[side].get(field, 0.0)
            initial_val_for_diff = initial_totals_for_day.get(f'{side}_{field}_total_baseline', 0.0)

            diff = current_val - initial_val_for_diff
            differences[side][field] = diff
            delta_percentages[side][field] = (diff / current_val * 100.0) if current_val != 0 else 0.0
    # --- End Calculate Differences and Delta ---
    # --- Store current delta percentages in history table BEFORE calculating change ---
    ts_for_delta_history = datetime.now(IST).isoformat()
    for side_hist in ['CE', 'PE']:
        for field_hist in fields_to_sum:
            hist_metric_name = f"{side_hist}_{field_hist}_delta_percent"
            hist_metric_value = delta_percentages[side_hist].get(field_hist, 0.0)
            c.execute('''
                INSERT INTO delta_percentage_history (expiry_date, metric_name, metric_value, metric_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (expiry_date, hist_metric_name, hist_metric_value, ts_for_delta_history))
    conn.commit() # Commit history before querying it for change calculation
    # --- Calculate Change in Delta Percentage ---
    # Fetches the delta_percentage from the *previous* 5-second interval
    change_in_delta_percentages = {'CE': {f: 0.0 for f in fields_to_sum}, 'PE': {f: 0.0 for f in fields_to_sum}}
    c.execute('''
    SELECT metric_name, metric_value
        FROM delta_percentage_history
        WHERE expiry_date = ? AND metric_timestamp < ? AND (
            metric_name LIKE '%_delta_percent'
        )
        ORDER BY metric_timestamp DESC
   ''', (expiry_date, ts_for_delta_history)) # Query for values before the ones just inserted

    # Group previous delta percentages by metric name to get the latest one for each
    latest_previous_delta_percentages = {}
    for prev_metric_name, prev_metric_value in c.fetchall():
        if prev_metric_name not in latest_previous_delta_percentages:
            latest_previous_delta_percentages[prev_metric_name] = prev_metric_value

    for side in ['CE', 'PE']:
        for field in fields_to_sum:
            current_delta_pct = delta_percentages[side].get(field, 0.0)
            prev_delta_pct_metric_name = f"{side}_{field}_delta_percent"
            prev_delta_pct_val = latest_previous_delta_percentages.get(prev_delta_pct_metric_name, 0.0)
            change_in_delta_percentages[side][field] = current_delta_pct - prev_delta_pct_val
    # --- End Calculate Change in Delta Percentage ---

     # --- Store all LATEST calculated metrics in 'metrics' table ---
    # Clear previous LATEST metrics for THIS specific expiry_date before inserting new ones
    c.execute('DELETE FROM metrics WHERE expiry_date = ?', (expiry_date,))
    metric_timestamp = datetime.now(IST).isoformat()
    for side in ['CE', 'PE']:
       for field in fields_to_sum:
            metric_name_total = f"{side}_{field}_total"
            metric_value_total = current_totals[side].get(field, 0.0)
            c.execute('''
                INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (expiry_date, metric_name_total, metric_value_total, metric_timestamp))
            # Store difference
            metric_name_diff = f"{side}_{field}_difference"
            metric_value_diff = differences[side].get(field, 0)
            c.execute('''
                INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (expiry_date, metric_name_diff, metric_value_diff, metric_timestamp))
            # Store delta percentage
            metric_name_delta_pct = f"{side}_{field}_delta_percent"
            metric_value_delta_pct = delta_percentages[side].get(field, 0.0)
            c.execute('''
                INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (expiry_date, metric_name_delta_pct, metric_value_delta_pct, metric_timestamp))
            # Store change in delta percentage
            metric_name_chg_delta_pct = f"{side}_{field}_chg_delta_percent"
            metric_value_chg_delta_pct = change_in_delta_percentages[side].get(field, 0.0)
            c.execute('''
                INSERT INTO metrics (expiry_date, metric_name, metric_value, metric_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (expiry_date, metric_name_chg_delta_pct, metric_value_chg_delta_pct, metric_timestamp))
    conn.commit()
    conn.close()
    print(f"Stored calculated metrics for expiry {expiry_date} at {metric_timestamp}")
@flask_app.route('/option_chain')
def get_option_chain_data_for_frontend(): # This endpoint needs to use the correct DB based on session
    """
    Get latest stored raw option chain data and calculated metrics for a given expiry date.
    This endpoint is for the frontend dashboard.
    """
    import json
    expiry_date = request.args.get('expiry_date')
    if not expiry_date:
        return jsonify({'error': 'expiry_date query parameter is required'}), 400

    db_file = get_db_file()
    if not db_file: return redirect(url_for('index')) # Redirect if not logged in
    conn = sqlite3.connect(db_file) # Use the correct DB file
    c = conn.cursor()
   # 1. Fetch raw option chain data (list of strike objects)
    c.execute('''
        SELECT raw_data FROM option_chain_data
        WHERE expiry_date = ?
        ORDER BY strike_price ASC
    ''', (expiry_date,))
    db_rows = c.fetchall()
    option_chain_list = []
    for row in db_rows:
        try:
            option_chain_list.append(json.loads(row[0]))
        except Exception:
            pass

    # 2. Fetch latest calculated metrics from the 'metrics' table
    latest_calculated_metrics = {}
    metric_fields = ['bidQty', 'askQty', 'openInterest', 'totalTradedVolume', 'IV', 'delta']
    metric_types = ['total', 'difference', 'delta_percent', 'chg_delta_percent', '15min_delta']

    for side in ['CE', 'PE']:
        latest_calculated_metrics[side] = {}
        for field in metric_fields:
            latest_calculated_metrics[side][field] = {}
            for m_type in metric_types:
                metric_name_to_query = f"{side}_{field}_{m_type}"
                c.execute('''
                    SELECT metric_value FROM metrics
                    WHERE expiry_date = ? AND metric_name = ?
                    ORDER BY metric_timestamp DESC LIMIT 1
                ''', (expiry_date, metric_name_to_query))
                metric_row = c.fetchone()
                latest_calculated_metrics[side][field][m_type] = metric_row[0] if metric_row else None
    conn.close()
    return jsonify({
        'option_chain': option_chain_list,
        'calculated_metrics': latest_calculated_metrics
    })

@flask_app.route('/fetch_data')
def fetch_data(): # This endpoint needs to use the correct DB based on session
    """Trigger immediate fetch of option chain data for given expiry date."""
    expiry_date = request.args.get('expiry_date')
    db_file = get_db_file()
    if not db_file: return redirect(url_for('index')) # Redirect if not logged in

    if not expiry_date:
        return jsonify({'error': 'expiry_date query parameter is required'}), 400
    # Call fetch function directly
    fetch_option_chain_data(expiry_date, db_file) # Pass db_file
    return jsonify({'message': f'Fetch triggered for expiry {expiry_date}'})

def get_token():
    """Retrieve the current access token from the database for the current user role, or None if not found or expired."""
    db_file = get_db_file()
    if not db_file:
        return None, None, None # No user role in session
    conn = sqlite3.connect(db_file) # Use the correct DB file
    c = conn.cursor() # Use the correct DB file
    c.execute('SELECT access_token, refresh_token, expires_at FROM tokens ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    if row:
        access_token, refresh_token, expires_at_str = row
        expires_at = datetime.fromisoformat(expires_at_str)
        now = datetime.now(IST)
        if now < expires_at:
            return access_token, refresh_token, expires_at
    return None, None, None

@flask_app.route('/')
def index(): # Login page
    """Render the login page with 'Login with Upstox' button."""
    # Clear session on landing on index page to force role selection
    session.pop('user_role', None)
    return render_template('login.html')

@flask_app.route('/login/<role>')
def login_with_role(role):
    """Set user role in session and redirect to Upstox OAuth authorization URL."""
    if role not in ['emperor', 'king']:
        return jsonify({'error': 'Invalid role specified'}), 400

    session['user_role'] = role
    # Redirect to the standard login route which will handle the OAuth flow
    return redirect(url_for('login'))

@flask_app.route('/login')
def login():
    """Redirect user to Upstox OAuth authorization URL."""
    user_role = session.get('user_role')
    if not user_role:
        return redirect(url_for('index')) # Force role selection if not set

    client_id = EMPEROR_CLIENT_ID if user_role == 'emperor' else KING_CLIENT_ID
    # client_secret is not needed for the initial auth URL redirect


    # Construct authorization URL with required parameters
    params = {
        'client_id': client_id, # Use the role-specific client_id
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
    }
    from urllib.parse import urlencode
    url = f"{AUTH_URL}?{urlencode(params)}"
    return redirect(url)

@flask_app.route('/manual_auth', methods=['POST'])
def manual_auth(): # This endpoint needs to use the correct credentials and DB based on session
    """Accept authorization code manually and exchange for access token."""
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({'error': 'Authorization code is required'}), 400
    code = data['code']

    user_role = session.get('user_role')
    if not user_role:
        return jsonify({'error': 'User role not set in session. Please login again.'}), 401

    client_id = EMPEROR_CLIENT_ID if user_role == 'emperor' else KING_CLIENT_ID
    client_secret = EMPEROR_CLIENT_SECRET if user_role == 'emperor' else KING_CLIENT_SECRET
    db_file = EMPEROR_DB_FILE if user_role == 'emperor' else KING_DB_FILE

    # Exchange authorization code for access token
    post_data = {
        'code': code,
        'client_id': client_id, # Use role-specific client_id
        'client_secret': client_secret, # Use role-specific client_secret
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    try:
        response = requests.post(TOKEN_URL, data=post_data)
        response.raise_for_status()
        # Try to parse JSON response safely
        try:
            token_data = response.json()
        except ValueError:
            # If response is not JSON, try to parse as string key=value pairs
            token_data = {}
            for pair in response.text.split('&'):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    token_data[k.strip()] = v.strip()
        # Defensive: if token_data is a string, convert to dict using json.loads
        if isinstance(token_data, str):
            import json
            try:
                token_data = json.loads(token_data)
            except Exception:
                token_data = {}
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to get access token: {e}'}), 500

    # Defensive: ensure token_data is a dict
    if not isinstance(token_data, dict):
        return jsonify({'error': 'Invalid token response format'}), 500

    access_token = token_data.get('access_token')
    refresh_token = token_data.get('refresh_token')
    expires_in = int(token_data.get('expires_in', 86400))  # Default 1 day if not provided

    if not access_token:
        return jsonify({'error': f'Access token not found in response: {token_data}'}), 500

    # Store token in database
    store_token(access_token, refresh_token, expires_in, db_file) # Pass the correct db_file

    return jsonify({'message': 'Authentication successful'})

@flask_app.route('/dashboard')
def dashboard():
    """Render the dashboard UI page."""
    user_role = session.get('user_role')
    if not user_role:
        return redirect(url_for('index')) # Redirect if not logged in

    return render_template('dashboard.html', user_role=user_role) # Pass user_role to the template

@flask_app.route('/token')
def token_info(): # This endpoint needs to use the correct DB based on session
    """Endpoint to get current token info (for testing)."""
    access_token, refresh_token, expires_at = get_token()
    if access_token:
        return jsonify({
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': expires_at.isoformat()
        })
    else:
        return jsonify({'error': 'No valid token found'}), 404

@flask_app.route('/ml-chatbot')
def ml_chatbot_page(): # Pass user role to template
    """Render the ML Chatbot UI page."""
    user_role = session.get('user_role')
    if not user_role:
        return redirect(url_for('index')) # Redirect if not logged in
    # Pass user_role to the template, which can then add it to the URL for JS
    # We just need to render the template
    return render_template('ml-chatbot.html')

@flask_app.route('/chat/prompt', methods=['POST'])
def handle_chat_prompt():
    """
    Receives user prompt from frontend, fetches latest metrics,
    sends to ML backend for prediction, and returns result.
    Expected JSON body: {'expiry_date': '...', 'user_prompt': '...'}
    """
    data = request.get_json()
    if not data or 'expiry_date' not in data or 'user_prompt' not in data:
        return jsonify({'error': 'Invalid input. Requires expiry_date and user_prompt.'}), 400

    db_file = get_db_file()
    if not db_file: return jsonify({'error': 'User role not set in session. Cannot predict.'}), 401

    expiry_date = data['expiry_date']
    user_prompt = data['user_prompt']

    # 1. Fetch latest live metrics from the correct database
    conn = sqlite3.connect(db_file) # Use the correct db_file
    c = conn.cursor()
    c.execute('''
        SELECT metric_name, metric_value FROM metrics
        WHERE expiry_date = ?
    ''', (expiry_date,))
    live_metrics = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    # Fetch raw option chain to extract spot price and high OI strikes
    conn = sqlite3.connect(db_file) # Use the correct DB file
    c = conn.cursor()
    # Fetch all raw_data entries for the given expiry_date, then parse them
    # Assuming each raw_data in option_chain_data is a JSON string representing ONE strike's full data
    c.execute("SELECT raw_data FROM option_chain_data WHERE expiry_date = ? ORDER BY strike_price ASC", (expiry_date,))
    raw_strike_json_strings = c.fetchall()
    conn.close()

    parsed_strikes_for_oi = []
    for row_tuple in raw_strike_json_strings:
        import json # Ensure json is imported
        try:
            parsed_strikes_for_oi.append(json.loads(row_tuple[0]))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for a strike in handle_chat_prompt: {row_tuple[0]}")
            continue

    if parsed_strikes_for_oi:
        live_metrics['price'] = parsed_strikes_for_oi[0].get('underlying_spot_price')
        call_oi_data = [{'strike': s['strike_price'], 'oi': s.get('call_options', {}).get('market_data', {}).get('oi', 0)} for s in parsed_strikes_for_oi if s.get('call_options', {}).get('market_data', {}).get('oi') is not None]
        put_oi_data = [{'strike': s['strike_price'], 'oi': s.get('put_options', {}).get('market_data', {}).get('oi', 0)} for s in parsed_strikes_for_oi if s.get('put_options', {}).get('market_data', {}).get('oi') is not None]
        
        call_oi_data.sort(key=lambda x: x['oi'], reverse=True)
        put_oi_data.sort(key=lambda x: x['oi'], reverse=True)
        live_metrics['high_call_oi_strikes'] = [item['strike'] for item in call_oi_data[:3]] # Top 3
        live_metrics['high_put_oi_strikes'] = [item['strike'] for item in put_oi_data[:3]]   # Top 3

    if not live_metrics:
         return jsonify({'error': f'No live metrics found for expiry {expiry_date}. Cannot predict.'}), 404

    # 2. Send live metrics and prompt to ML backend for prediction
    ml_payload = {
        'live_metrics': live_metrics,
        'user_prompt': user_prompt
    }

    try:
        response = requests.post(f'{ML_BACKEND_URL}/predict', json=ml_payload)
        response.raise_for_status() # Raise an exception for bad status codes
        predictions = response.json()
        return jsonify(predictions)
    except requests.RequestException as e:
        print(f"Error sending prediction request to ML backend: {e}")
        return jsonify({'error': f'Error getting prediction from ML backend: {e}'}), 500

@flask_app.route('/chat/feedback', methods=['POST'])
def handle_chat_feedback():
    """
    Receives user feedback from frontend and forwards it to the ML backend.
    Expected JSON body: {'feedback': {...}}
    The structure of the feedback dictionary will depend on how you design
    the feedback mechanism in your UI and how the ML backend expects it.
    This endpoint should only process feedback if the user is 'emperor'.
    """
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({'error': 'Invalid input. Requires feedback data.'}), 400

    user_role = session.get('user_role')
    if user_role != 'emperor':
        # If not emperor, accept the feedback but don't send it to the ML backend
        print(f"Feedback received from user role '{user_role}', but training/feedback is disabled for this role.")
        # Optionally log this attempt or just acknowledge
        return jsonify({'status': 'success', 'message': 'Feedback noted (but not used for model training).'})

    # If user is emperor, proceed to send feedback to ML backend
    feedback_data = data['feedback']
    ml_payload = {
        'feedback': feedback_data
    }
    try:
        response = requests.post(f'{ML_BACKEND_URL}/feedback', json=ml_payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return jsonify(response.json()) # Return ML backend's confirmation
    except requests.RequestException as e:
        print(f"Error sending feedback to ML backend: {e}")
        return jsonify({'error': f'Error sending feedback to ML backend: {e}'}), 500

# Wrap the Flask app with WsgiToAsgi for ASGI compatibility (e.g., Uvicorn)
app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
        # Initialize databases and add daily cleanup job are now done outside __main__
    # Start scheduler if it has jobs and isn't running (for local dev)
    # In production, the scheduler should be started as part of the application setup
    if not scheduler.running and scheduler.get_jobs():
        scheduler.start()
    flask_app.run(debug=True) # Use flask_app for Flask's built-in dev server
