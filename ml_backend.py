import os
import sqlite3
from flask import Flask, request, jsonify
from datetime import datetime, timezone, timedelta
from flask_cors import CORS # Import CORS
from asgiref.wsgi import WsgiToAsgi
import json

# Original Flask app instance
flask_ml_app = Flask(__name__)
CORS(flask_ml_app) # Enable CORS for all routes by default

# Configuration for ML Backend
# Use environment variable for DB file path, with a default
ML_DB_FILE = os.environ.get('ML_DB_FILE', 'ml_data.db')

# IST timezone offset (assuming same timezone as main app)
IST = timezone(timedelta(hours=5, minutes=30))

def init_ml_db():
    """Initialize the SQLite database for ML data."""
    conn = sqlite3.connect(ML_DB_FILE)
    c = conn.cursor()
    # Enable WAL mode for better concurrency
    c.execute('PRAGMA journal_mode=WAL;')
    # Table to store 15-minute metric snapshots for training
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expiry_date TEXT NOT NULL,
            snapshot_timestamp TEXT NOT NULL,
            metrics_data TEXT NOT NULL -- Store all metrics as JSON string
        )
    ''')
    # Table to store user interactions and feedback
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_prompt TEXT,
            model_prediction TEXT, -- Store prediction output as JSON string
            user_feedback TEXT, -- Store feedback as JSON string
            live_metrics_snapshot TEXT -- Store metrics used for prediction as JSON string
        )
    ''')
    # Table to store the current state of the ML model (rules, weights, etc.)
    # The structure of this table will depend heavily on the chosen ML approach
    # Placeholder: store a simple JSON representation of the model state
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            state_data TEXT NOT NULL -- Store model state as JSON string
        )
    ''')
    conn.commit()
    conn.close()

# Placeholder for the ML model logic

# Initialize the database tables before the model tries to access them
init_ml_db()
class OptionChainMLModel:
    def __init__(self):
        # Load model state from DB on initialization
        self._load_state()
        # Placeholder for learned rules or parameters
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
        # Optional: Keep only the latest state or version states
        # c.execute('DELETE FROM model_state')
        state_data = json.dumps(self.rules)
        timestamp = datetime.now(IST).isoformat()
        c.execute('INSERT INTO model_state (timestamp, state_data) VALUES (?, ?)', (timestamp, state_data))
        conn.commit()
        conn.close()
        print("ML model state saved.")

    def predict(self, live_metrics, user_prompt):
        """
        Generate predictions based on live metrics and user prompt.
        Uses specific subsets of metrics for each prediction type.
        """
        # This is where the core ML logic will go.
        # For now, it's a placeholder.
        print(f"ML Model received live metrics for prediction: {list(live_metrics.keys())}")
        print(f"User prompt: {user_prompt}")

        # Extract relevant features based on prediction type implied by prompt
        # (This logic needs refinement based on how prompts map to predictions)
        features_for_bias = {k: live_metrics.get(k) for k in [
            'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
            'CE_totalTradedVolume_delta_percent', 'PE_totalTradedVolume_delta_percent',
            'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price',
            'high_call_oi_strikes', 'high_put_oi_strikes' # New features
        ]}
        features_for_strength = {k: live_metrics.get(k) for k in [
             'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
             'CE_totalTradedVolume_delta_percent', 'PE_totalTradedVolume_delta_percent',
             'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price',
             # Strength might also be influenced by how close price is to S/R
             'high_call_oi_strikes', 'high_put_oi_strikes'
        ]}
        features_for_participants = {k: live_metrics.get(k) for k in [
            'CE_openInterest_delta_percent', 'PE_openInterest_delta_percent',
            'CE_IV_delta_percent', 'PE_IV_delta_percent', 'price'
        ]}

        # Apply learned rules or model logic based on features and prompt
        # Access features:
        current_price = features_for_bias.get('price')
        # Ensure high_call_strikes and high_put_strikes are lists, even if None or not found
        high_call_strikes = features_for_bias.get('high_call_oi_strikes') or []
        high_put_strikes = features_for_bias.get('high_put_oi_strikes', [])

        # Placeholder predictions
        bias = "Neutral"
        strength = "Weak"
        participants = "Undetermined"
        confidence = 50.0 # Placeholder confidence

        # Determine which parts of the prediction the user is interested in based on prompt
        prompt_lower = user_prompt.lower()
        request_bias = any(word in prompt_lower for word in ['bias', 'direction', 'outlook'])
        request_strength = any(word in prompt_lower for word in ['strength', 'momentum', 'speed'])
        request_participants = any(word in prompt_lower for word in ['participants', 'oi', 'volume', 'iv', 'delta', 'who is active'])
        request_levels = any(word in prompt_lower for word in ['support', 'resistance', 'levels', 'high oi'])
        analysis_notes = [] # To store reasons for the prediction

        # Example: Simple rule based on OI delta percent
        ce_oi_delta_pct = features_for_bias.get('CE_openInterest_delta_percent', 0) or 0
        pe_oi_delta_pct = features_for_bias.get('PE_openInterest_delta_percent', 0) or 0

        if ce_oi_delta_pct > pe_oi_delta_pct * 1.2:
             bias = "Bullish"
             confidence = min(confidence + 15, 100) # Increase confidence for strong signal
             analysis_notes.append("CE OI delta % significantly higher than PE OI delta %.")
        elif pe_oi_delta_pct > ce_oi_delta_pct * 1.2:
             bias = "Bearish"
             confidence = min(confidence + 15, 100) # Increase confidence for strong signal
             analysis_notes.append("PE OI delta % significantly higher than CE OI delta %.")

        # Example: Simple rule for strength based on Volume delta percent
        ce_vol_delta_pct = features_for_strength.get('CE_totalTradedVolume_delta_percent', 0) or 0
        pe_vol_delta_pct = features_for_strength.get('PE_totalTradedVolume_delta_percent', 0) or 0
        if ce_vol_delta_pct + pe_vol_delta_pct > 50: # Arbitrary threshold
             strength = "Strong"
             confidence = min(confidence + 10, 100) # Increase confidence for strong signal
             analysis_notes.append("Overall volume delta % is high.")

        # Example: Simple rule for participants based on OI delta percent and IV delta percent
        ce_iv_delta_pct = features_for_participants.get('CE_IV_delta_percent', 0) or 0
        pe_iv_delta_pct = features_for_participants.get('PE_IV_delta_percent', 0) or 0

        if ce_oi_delta_pct > 0 and ce_iv_delta_pct > 0:
             participants = "New Longs (CE)"
             confidence = min(confidence + 10, 100) # Increase confidence for strong signal
             analysis_notes.append("CE OI and IV delta % both positive, suggesting new CE longs.")
        elif pe_oi_delta_pct > 0 and pe_iv_delta_pct > 0:
             participants = "New Longs (PE)"
             confidence = min(confidence + 10, 100) # Increase confidence for strong signal
             analysis_notes.append("PE OI and IV delta % both positive, suggesting new PE longs.")
        # Add more complex rules for unwinding/covering based on negative delta percent and price movement

        # --- Enhanced Participants Response ---
        # If the user specifically asked about participants AND the simple rule didn't fire,
        # provide a summary of relevant metrics in the analysis notes or the participants field itself.
        if request_participants and participants == "Undetermined":
             # Option 1: Add summary to analysis notes (current frontend displays notes)
             analysis_notes.append(f"Current CE OI Delta %: {ce_oi_delta_pct:.2f}%, PE OI Delta %: {pe_oi_delta_pct:.2f}%.")
             analysis_notes.append(f"Current CE Volume Delta %: {ce_vol_delta_pct:.2f}%, PE Volume Delta %: {pe_vol_delta_pct:.2f}%.")
             analysis_notes.append(f"Current CE IV Delta %: {ce_iv_delta_pct:.2f}%, PE IV Delta %: {pe_iv_delta_pct:.2f}%.")
             # Option 2: Set the participants field to a summary string (requires frontend adjustment to display this)
             # participants = f"CE OI: {ce_oi_delta_pct:.2f}%, PE OI: {pe_oi_delta_pct:.2f}% | CE Vol: {ce_vol_delta_pct:.2f}%, PE Vol: {pe_vol_delta_pct:.2f}%"
             # For now, adding to notes is simpler with current frontend logic.
        # --- End Enhanced Participants Response ---

        # Example: Incorporate price and S/R
        if current_price is not None and high_call_strikes: # Check if price is available and strikes list is not empty
            # Check if price is near the strongest call OI (resistance)
            # Ensure the strike value is not None before calculation
            if high_call_strikes[0] is not None and high_call_strikes[0] > 0 and current_price is not None and \
               abs(high_call_strikes[0] - current_price) < (high_call_strikes[0] * 0.002): # Within 0.2% of resistance
                analysis_notes.append(f"Price ({current_price}) is approaching strong Call OI resistance at {high_call_strikes[0]}.")
                if bias == "Bullish": # If bullish but near resistance, might weaken
                    strength = "Weak" if strength == "Neutral" else strength # Don't override if already strong
                    confidence = max(confidence - 5, 0)

        if current_price is not None and high_put_strikes: # Check if price is available and strikes list is not empty
            # Check if price is near the strongest put OI (support)
            # Ensure the strike value is not None before calculation
            if high_put_strikes[0] is not None and high_put_strikes[0] > 0 and current_price is not None and \
               abs(current_price - high_put_strikes[0]) < (high_put_strikes[0] * 0.002): # Within 0.2% of support
                analysis_notes.append(f"Price ({current_price}) is near strong Put OI support at {high_put_strikes[0]}.")
                if bias == "Bearish": # If bearish but near support, might weaken
                    strength = "Weak" if strength == "Neutral" else strength
                    confidence = max(confidence - 5, 0)

        # This is the complete prediction that will be logged internally
        prediction_to_log = {
            "bias": bias,
            "strength": strength,
            "participants": participants,
            "confidence": round(confidence, 2),
            "analysis_notes": analysis_notes # All generated notes
        }

        # Now, construct the response to send to the user, filtered by their prompt
        response_to_user = {}

        if not (request_bias or request_strength or request_participants or request_levels):
            # No specific request, send the full prediction (or a copy of it)
            response_to_user = prediction_to_log.copy()
        else:
            # Specific request(s) made, build the filtered response
            if request_bias:
                response_to_user['bias'] = bias
            if request_strength:
                response_to_user['strength'] = strength
            if request_participants:
                response_to_user['participants'] = participants
            
            # Always include confidence in the filtered response
            response_to_user['confidence'] = round(confidence, 2)

            # Filter analysis_notes based on relevance to the request
            relevant_notes = []
            if analysis_notes: # Make sure analysis_notes is not None
                for note in analysis_notes:
                    note_lower = note.lower()
                    is_note_relevant = False
                    if request_bias and any(kw in note_lower for kw in ["bias", "oi delta", "bullish", "bearish", "direction"]):
                        is_note_relevant = True
                    if request_strength and any(kw in note_lower for kw in ["strength", "volume delta", "strong", "weak", "momentum"]):
                        is_note_relevant = True
                    if request_participants and any(kw in note_lower for kw in ["participants", "oi", "iv", "delta", "longs", "shorts", "active"]):
                        is_note_relevant = True
                    if request_levels and any(kw in note_lower for kw in ["support", "resistance", "level", "strike"]):
                        is_note_relevant = True
                    
                    if is_note_relevant:
                        relevant_notes.append(note)
            
            response_to_user['analysis_notes'] = relevant_notes

            # If 'request_levels' was specifically true, ensure high OI strikes are mentioned
            if request_levels:
                if 'analysis_notes' not in response_to_user: response_to_user['analysis_notes'] = []
                
                # Add key S/R level summaries if requested.
                # Check if similar info isn't already in a more specific rule-based note.
                call_sr_summary_note = f"Key Call OI Resistance levels: {high_call_strikes}"
                put_sr_summary_note = f"Key Put OI Support levels: {high_put_strikes}"

                # Simple check to avoid adding generic S/R list if specific strikes already mentioned
                has_call_sr_details = any(str(s_strike) in note for s_strike in high_call_strikes for note in response_to_user.get('analysis_notes', [])) if high_call_strikes else False
                has_put_sr_details = any(str(s_strike) in note for s_strike in high_put_strikes for note in response_to_user.get('analysis_notes', [])) if high_put_strikes else False

                if high_call_strikes and not has_call_sr_details:
                     response_to_user['analysis_notes'].append(call_sr_summary_note)
                if high_put_strikes and not has_put_sr_details:
                     response_to_user['analysis_notes'].append(put_sr_summary_note)

        # Log the full interaction data (prediction_to_log)
        interaction_id = self._log_interaction(user_prompt, prediction_to_log, live_metrics)
        
        # Add interaction_id to the (potentially filtered) response being sent to the user
        response_to_user['interaction_id'] = interaction_id

        return response_to_user

    def train_incremental(self, snapshot_metrics):
        """
        Update the model based on 15-minute snapshot data.
        This is where the incremental learning from market data happens.
        """
        print(f"ML Model received 15-min snapshot for training: {list(snapshot_metrics.keys())}")
        # ... Incremental training logic using snapshot_metrics ...
        # This could involve updating rule weights, adding new rules based on patterns, etc.

        # Example: Simple rule update based on snapshot (Illustrative)
        # If snapshot shows strong bullish movement and model predicted neutral, adjust rules
        # This part is highly dependent on the specific ML approach

        self._save_state() # Save state after training

    def process_feedback(self, user_feedback):
        """
        Update the model based on explicit user feedback/corrections.
        This is the primary learning mechanism from user interaction.
        """
        print(f"ML Model received user feedback: {json.dumps(user_feedback, indent=2)}")

        # Placeholder for actual rule learning from user_feedback['rule_suggestion']
        # This would involve parsing the rule_suggestion string into a structured rule
        # and adding it to self.rules. For now, we'll just log it.
        if isinstance(user_feedback, dict) and user_feedback.get('rule_suggestion'):
            suggested_rule_text = user_feedback['rule_suggestion']
            # Example: A very simple way to store it directly if no parsing is done yet
            # In a real system, you'd parse this into a more structured format.
            if 'user_suggested_rules' not in self.rules:
                self.rules['user_suggested_rules'] = []
            self.rules['user_suggested_rules'].append({
                "raw_rule": suggested_rule_text,
                "added_at": datetime.now(IST).isoformat()
            })
            print(f"Stored user suggested rule: {suggested_rule_text}")

        # Placeholder for learning from user_feedback['correction']
        # This is more complex as it might involve adjusting existing rules or parameters.
        # For now, the correction is logged via _log_interaction.
        if isinstance(user_feedback, dict) and user_feedback.get('correction'):
            correction_data = user_feedback['correction']
            print(f"User provided correction: {correction_data}")
            # Here you might try to find which of your existing rules led to the
            # wrong prediction and adjust its confidence or parameters, or flag it.
            # This is a non-trivial ML task.

        # The feedback (including correction, rule_suggestion, comment)
        # will be logged by the _log_interaction method.
        # We need to ensure _log_interaction can handle the new feedback structure.

        # Log feedback
        # If feedback contains an interaction_id, we update that interaction.
        # Otherwise, we log it as a new, potentially unlinked, feedback entry.
        interaction_id_to_update = None
        if isinstance(user_feedback, dict) and user_feedback.get('interaction_id'):
            interaction_id_to_update = user_feedback.get('interaction_id')

        # Pass the interaction_id if we are updating an existing record with feedback
        self._log_interaction(
            user_feedback=user_feedback,
            interaction_id_to_update=interaction_id_to_update # New parameter for clarity
        )
        self._save_state() # Save state after feedback

    def _log_interaction(self, user_prompt=None, model_prediction=None, live_metrics_snapshot=None, user_feedback=None, interaction_id_to_update=None):
        """Logs user prompts, model predictions, and feedback."""
        conn = sqlite3.connect(ML_DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now(IST).isoformat()

        logged_interaction_id = None

        if interaction_id_to_update and user_feedback:
             # Update an existing interaction with feedback
             c.execute('''
                 UPDATE user_interactions
                 SET user_feedback = ?
                 WHERE id = ?
             ''', (json.dumps(user_feedback), interaction_id_to_update))
             logged_interaction_id = interaction_id_to_update
             print(f"Updated interaction {interaction_id_to_update} with feedback.")

        else:
            # Insert a new interaction record
            c.execute('''
                INSERT INTO user_interactions (timestamp, user_prompt, model_prediction, user_feedback, live_metrics_snapshot)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                timestamp,
                user_prompt,
                json.dumps(model_prediction) if model_prediction else None,
                json.dumps(user_feedback) if user_feedback else None,
                json.dumps(live_metrics_snapshot) if live_metrics_snapshot else None
            ))
            logged_interaction_id = c.lastrowid # Get the ID of the newly inserted row
            print(f"Logged new interaction with ID: {logged_interaction_id}")

        conn.commit()
        conn.close()
        return logged_interaction_id # Return the ID of the logged/updated interaction

ml_model = OptionChainMLModel()

@flask_ml_app.route('/ml/predict', methods=['POST'])
def predict_endpoint():
    """
    Receives live metrics and user prompt, returns predictions.
    Expected JSON body: {'live_metrics': {...}, 'user_prompt': '...'}
    """
    data = request.get_json()
    if not data or 'live_metrics' not in data or 'user_prompt' not in data:
        return jsonify({'error': 'Invalid input. Requires live_metrics and user_prompt.'}), 400

    live_metrics = data['live_metrics']
    user_prompt = data['user_prompt']

    # Call predict once. It already includes the interaction_id in its response.
    prediction_response = ml_model.predict(live_metrics, user_prompt)
    print(f"Sending prediction to frontend: {json.dumps(prediction_response)}")
    return jsonify(prediction_response)

@flask_ml_app.route('/ml/train_snapshot', methods=['POST'])
def train_snapshot_endpoint():
    """
    Receives 15-minute snapshot metrics for incremental training.
    Expected JSON body: {'expiry_date': '...', 'snapshot_timestamp': '...', 'metrics_data': {...}}
    """
    data = request.get_json()
    if not data or 'expiry_date' not in data or 'snapshot_timestamp' not in data or 'metrics_data' not in data:
        return jsonify({'error': 'Invalid input. Requires expiry_date, snapshot_timestamp, and metrics_data.'}), 400

    # Store the raw snapshot data for potential later use or direct training
    conn = sqlite3.connect(ML_DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO training_snapshots (expiry_date, snapshot_timestamp, metrics_data)
        VALUES (?, ?, ?)
    ''', (data['expiry_date'], data['snapshot_timestamp'], json.dumps(data['metrics_data'])))
    conn.commit()
    conn.close()

    # Pass the snapshot data to the model for incremental training
    ml_model.train_incremental(data['metrics_data'])

    return jsonify({'status': 'success', 'message': 'Snapshot received and used for training.'})

@flask_ml_app.route('/ml/feedback', methods=['POST'])
def feedback_endpoint():
    """
    Receives user feedback/corrections for model learning.
    Expected JSON body: {'feedback': {...}} - format TBD based on UI
    """
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({'error': 'Invalid input. Requires feedback.'}), 400

    user_feedback = data['feedback']
    ml_model.process_feedback(user_feedback)

    return jsonify({'status': 'success', 'message': 'Feedback received and processed.'})

# Wrap the Flask app with WsgiToAsgi for ASGI compatibility (e.g., Uvicorn)
app = WsgiToAsgi(flask_ml_app)

if __name__ == '__main__':
    # Run on a different port than the main app
    # Note: When deploying with Uvicorn/Gunicorn on Render, this __main__ block
    # is typically not executed. The ASGI server loads the 'app' object directly.
    # The WsgiToAsgi wrapper ensures the 'app' object is ASGI-compatible.
    flask_ml_app.run(debug=True, port=5001)