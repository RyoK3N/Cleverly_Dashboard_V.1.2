# Updated chatbot route to use embeddings in main.py
import os
from flask import Flask, render_template, url_for, request, redirect, flash, session, jsonify
from datetime import datetime, timedelta

app = Flask(__name__, static_url_path='/static', static_folder='static')
from pathlib import Path
import logging
import psycopg2
from werkzeug.security import check_password_hash
from functools import wraps
import traceback
import openai
import sys

from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, login_required,
                         logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import subprocess

# Flask app configuration
app = Flask(__name__, template_folder="pages")
app.config.update(SECRET_KEY='your-secret-key',
                  SQLALCHEMY_DATABASE_URI='sqlite:///database.db',
                  SQLALCHEMY_TRACK_MODIFICATIONS=False)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='Dharmil').first():
        user = User(username='Dharmil',
                    password_hash=generate_password_hash('Clev@2025'))
        db.session.add(user)
        db.session.commit()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
from monday_extract_groups import fetch_data
import json
import os
from monday_extract_groups import process_data , process_data_COLD_EMAIL
import requests
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logging and Directories
# --------------------------------------------------------------------------- #
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "sales_dash.log"

# Create data directories
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
MONDAY_DATA_DIR = DATA_DIR / "downloads_monday"
MONDAY_DATA_DIR.mkdir(exist_ok=True)
CALENDLY_DATA_DIR = DATA_DIR / "Calendly"
CALENDLY_DATA_DIR.mkdir(exist_ok=True)


def configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


logger = configure_logger()

# --------------------------------------------------------------------------- #
# Flask
# --------------------------------------------------------------------------- #


def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html',
                                   message='Invalid username or password')

    return render_template('login.html')


@app.route("/dashboard")
@login_required
def dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template("dashboard.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/fetch-data", methods=["POST"])
def fetch_monday_data():
    session_dir = Path("./sessions/sessions")
    session_dir.mkdir(exist_ok=True)

    try:
        now = datetime.now()
        csv_files = [p for p in session_dir.glob("*.csv") if p.is_file()]

        # ── 1. Decide whether refresh is needed ────────────────────────────
        needs_refresh = False
        if len(csv_files) < 7:
            needs_refresh = True
            logger.info("Only %d CSV file(s) found – refreshing",
                        len(csv_files))
        else:
            ages = []
            for fp in csv_files:
                try:
                    ts = fp.stem.split("_", 1)[1]
                    ages.append(now -
                                datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S"))
                except Exception:
                    logger.warning("Unparseable timestamp: %s", fp.name)
                    ages.append(timedelta(days=365))
            if sum(ages, timedelta()) / len(ages) >= timedelta(hours=24):
                needs_refresh = True
                logger.info("Average age ≥ 24 h – refreshing")

        if not needs_refresh:
            return jsonify(success=True,
                           message="Data is already up-to-date"), 200

        # ── 2. Run pipeline -----------------------------------------------------------------
        logger.info("Launching data_pipeline2.py …")
        proc = subprocess.run(
            [sys.executable, "data_pipeline2.py"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            logger.error("Pipeline failed:\n%s", proc.stderr)
            return (
                jsonify(
                    success=False,
                    message="Pipeline execution failed",
                    stderr=proc.stderr,
                ),
                500,
            )

        logger.debug("Pipeline stdout:\n%s", proc.stdout)
        return (
            jsonify(success=True, message="Fresh data downloaded"),
            200,
        )

    except Exception as exc:
        logger.exception("Unexpected error during /fetch-data:")
        return jsonify(success=False, message=str(exc)), 500


# @app.route("/fetch-data", methods=["POST"])
# def fetch_monday_data():
#     """
#     POST /fetch-data
#     – If <7 session CSV files OR the average file age ≥24 h, download fresh data
#     – Save each group to CSV + save whole payload to JSON
#     """
#     session_dir = Path("./sessions")
#     session_dir.mkdir(exist_ok=True)

#     try:
#         now_time = datetime.now()

#         # Count only the CSV files we create
#         csv_files = [p for p in session_dir.glob("*.csv") if p.is_file()]

#         # -------------------- 1. FEWER THAN 7 FILES → FETCH ------------------
#         if len(csv_files) < 7:
#             logger.info(
#                 "Found %d CSV file(s) in sessions (need 7) → fetching fresh data.",
#                 len(csv_files),
#             )

#         # -------------------- 2. EXACTLY 7 FILES → AGE CHECK -----------------
#         else:
#             deltas = []
#             for fp in csv_files:
#                 try:
#                     # Filename pattern:  <group>_YYYY-MM-DD_HH-MM-SS.csv
#                     ts_part = fp.stem.split("_", 1)[1]
#                     file_dt = datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S")
#                     deltas.append(now_time - file_dt)
#                 except (IndexError, ValueError):
#                     logger.warning(
#                         "Unrecognised timestamp in %s – treating as stale.",
#                         fp.name)
#                     deltas.append(timedelta(days=365))

#             avg_age = sum(deltas, timedelta()) / len(deltas)
#             logger.debug("Average age of session data: %.1f hours",
#                          avg_age.total_seconds() / 3600)

#             if avg_age.total_seconds() < 86_400:  # 24 h
#                 return jsonify(success=True,
#                                message="Data is already up to date")

#             logger.info(
#                 "Session data average age ≥ 24 h → fetching fresh data.")

#         # -------------------- 3. FETCH & SAVE NEW DATA -----------------------
#         logger.info("Starting data fetch from Monday.com …")
#         data = fetch_data()  # expected dict-like {name: DataFrame}

#         if not isinstance(data, dict):
#             raise TypeError(
#                 f"fetch_data() must return a dict, got {type(data)}")

#         logger.info("Successfully fetched data with %d groups.", len(data))

#         timestamp = now_time.strftime("%Y-%m-%d_%H-%M-%S")

#         # Save each group as CSV
#         for idx, (name, df) in enumerate(data.items(), 1):
#             safe = "_".join(str(name).strip().lower().split()) or f"group{idx}"
#             out_csv = session_dir / f"{safe}_{timestamp}.csv"
#             df.to_csv(out_csv, index=False)
#             logger.info("Saved %s", out_csv.name)

#         # Save **entire** payload as JSON (DataFrames → list-of-dicts)
#         consolidated = {
#             k: v.to_dict(orient="records")
#             for k, v in data.items()
#         }
#         out_json = session_dir / f"data_{timestamp}.json"
#         out_json.write_text(json.dumps(consolidated, indent=2),
#                             encoding="utf-8")
#         logger.info("Saved %s", out_json.name)

#         return jsonify(
#             success=True,
#             message=f"Fetched & saved data for {len(data)} groups.",
#         )

#     except Exception as exc:
#         logger.error("Error fetching Monday data: %s", exc)
#         logger.debug("Traceback:\n%s", traceback.format_exc())
#         return jsonify(success=False,
#                        message=f"Failed to fetch data: {exc}"), 500


@app.route('/fetch-calendly-data', methods=['POST'])
def fetch_calendly_data():
    try:
        logger.info("Starting Calendly data fetch...")

        # Run the fetch_calendly_data.py script
        result = subprocess.run(['python', 'fetch_calendly_data.py'],
                                capture_output=True,
                                text=True,
                                cwd='.')

        if result.returncode == 0:
            logger.info("Calendly data fetch completed successfully")
            logger.info(f"Script output: {result.stdout}")
            return jsonify({
                "status": "success",
                "message": "Calendly data fetched successfully",
                "output": result.stdout
            })
        else:
            logger.error(
                f"Calendly fetch failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return jsonify({
                "status": "error",
                "message": f"Script failed: {result.stderr}"
            })

    except Exception as e:
        logger.error(f"Failed to run Calendly fetch script: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/")
def chatbot():
    return render_template("chatbot.html")


@app.route('/update_llm_context', methods=['POST'])
def update_llm_context():
    try:
        logger.info("Starting update_llm_context")
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Expected JSON request"
            })

        latest_file = max(Path('./sessions/sessions').glob('data_*.json'),
                          key=lambda f: f.stat().st_mtime)
        logger.info(f"Using data file: {latest_file}")

        with latest_file.open('r', encoding='utf-8') as file:
            json_data = json.load(file)
            data = {k: pd.DataFrame(v) for k, v in json_data.items()}
            logger.info(f"Loaded data groups: {list(data.keys())}")

        logger.info("Processing data for embeddings")
        processed_df = process_data(
            data,
            request.json.get('start_date', '1900-01-01').strip(),
            request.json.get('end_date', '2100-12-31').strip(),
            request.json.get('filter_column', 'Sales Call Date').strip())

        logger.info(f"Processed data shape: {processed_df.shape}")
        logger.info(f"Sample data:\n{processed_df.head()}")

        # Store the processed DataFrame for context
        app.current_data = processed_df.copy()

        # Create embeddings for each row
        embeddings_data = []
        for idx, row in processed_df.iterrows():
            text = ' '.join(f"{col}: {val}" for col, val in row.items())
            logger.debug(f"Creating embedding for row {idx}:\n{text}")
            embedding = create_embeddings(text)
            if embedding:
                embeddings_data.append({
                    'text': text,
                    'embedding': embedding,
                    'index': idx,
                    'row_data': row.to_dict()
                })

        app.embeddings_store = {'current': embeddings_data}
        logger.info(
            f"Updated embeddings store with {len(embeddings_data)} entries")
        return jsonify({"status": "success"})

    except Exception as e:
        logger.error(f"Error in update_llm_context: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_owners', methods=['GET'])
def get_owners():
    try:
        SESSION_DIR = Path("./sessions/sessions")
        latest_file = max(SESSION_DIR.glob("data_*.json"),
                          key=lambda f: f.stat().st_mtime)

        with latest_file.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
            data = {k: pd.DataFrame(v) for k, v in json_data.items()}

        all_owners = set()
        for df in data.values():
            if 'Owner' in df.columns:
                all_owners.update(df['Owner'].dropna().unique())

        return jsonify({"owners": sorted(list(all_owners))})
    except Exception as e:
        logger.error(f"Error getting owners: {e}")
        return jsonify({"owners": []})


@app.route('/apply_filters', methods=['POST'])
def apply_filters_route():
    try:
        logger.info("Starting apply_filters_route")
        # Identify the latest data file in the sessions folder
        SESSION_DIR = Path("./sessions/sessions")
        data_files = [
            f for f in SESSION_DIR.glob("data_*.json") if f.is_file()
        ]
        logger.info(f"Found {len(data_files)} data files")
        latest_file = max(data_files,
                          key=lambda f: f.stat().st_mtime,
                          default=None)

        if not latest_file:
            return jsonify({
                "status":
                "error",
                "message":
                "No data files found in the session directory."
            })

        logger.info(f"Loading data from {latest_file}")
        with latest_file.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
            # Convert JSON data into DataFrames
            data = {}
            for key, items in json_data.items():
                data[key] = pd.DataFrame(items)
                logger.info(
                    f"Created DataFrame for {key} with {len(items)} rows")

        st_date = request.json.get('start_date', '1900-01-01').strip()
        end_date = request.json.get('end_date', '2100-12-31').strip()
        filter_column = request.json.get('filter_column',
                                         ['Sales Call Date', 'Date Created'])
        if not filter_column:
            filter_column = 'Sales Call Date'  # Default value if none provided
        logger.info(
            f"Processing data with date range {st_date} to {end_date} on column {filter_column}"
        )
        selected_owners = request.json.get('selected_owners', [])
        processed_df = process_data(data, st_date, end_date, filter_column)

        if selected_owners:
            # Filter data by selected owners (excluding any existing Total row)
            filtered_df = processed_df[processed_df['Owner'].isin(selected_owners)]
            
            # Recalculate totals for filtered data
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            totals_dict = {'Owner': 'Total'}
            
            for col in numeric_cols:
                if col != 'Owner':
                    if '%' in col or 'Rate' in col:
                        # For percentage columns, calculate mean
                        totals_dict[col] = filtered_df[col].mean()
                    else:
                        # For other numeric columns, calculate sum
                        totals_dict[col] = filtered_df[col].sum()
            
            # Create totals row DataFrame
            totals_row = pd.DataFrame([totals_dict])
            
            # Combine filtered data with new totals row
            processed_df = pd.concat([filtered_df, totals_row], ignore_index=True)
        
        logger.info(
            f"Data processing complete. Processed DataFrame has {len(processed_df)} rows"
        )

        processed_df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        },
                             inplace=True)

        numeric_columns = processed_df.select_dtypes(
            include=[np.number]).columns
        for col in numeric_columns:
            processed_df[col] = processed_df[col].astype(float)

        table_html = processed_df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            border=0)
        return jsonify({"status": "success", "table": table_html})

    except Exception as e:
        logger.error("Error in apply_filters_route: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})


def create_embeddings(text):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(input=text,
                                            model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return None


@app.route('/data')
@login_required
def data():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template("data.html")


@app.route('/get_latest_data', methods=['GET'])
@login_required
def get_latest_data():
    try:
        SESSION_DIR = Path("./sessions/sessions")
        data_files = [
            f for f in SESSION_DIR.glob("data_*.json") if f.is_file()
        ]

        if not data_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No data files found. Please fetch data first."
            })

        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading data from: {latest_file}")

        with latest_file.open("r", encoding="utf-8") as file:
            json_data = json.load(file)

        logger.info(f"Found {len(json_data)} groups in data file")

        # Convert to a single flat table combining all groups
        all_data = []
        groups_info = {}

        for group_name, group_data in json_data.items():
            if group_data and isinstance(group_data, list):
                groups_info[group_name] = len(group_data)
                for row in group_data:
                    if isinstance(row, dict):
                        row_with_group = row.copy()
                        row_with_group['Data_Group'] = group_name
                        # Handle NaN values
                        for key, value in row_with_group.items():
                            if pd.isna(value) or value in [np.inf, -np.inf]:
                                row_with_group[key] = None
                        all_data.append(row_with_group)
            else:
                logger.warning(
                    f"Group {group_name} has no data or invalid format")
                groups_info[group_name] = 0

        logger.info(
            f"Processed {len(all_data)} total records across {len(groups_info)} groups"
        )
        logger.info(f"Groups info: {groups_info}")

        return jsonify({
            "status": "success",
            "data": all_data,
            "filename": latest_file.name,
            "total_records": len(all_data),
            "groups": list(json_data.keys()),
            "groups_info": groups_info
        })

    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Failed to load data: {str(e)}"
        })


@app.route('/fetch_monday_csv_data', methods=['POST'])
@login_required
def fetch_monday_csv_data():
    try:
        logger.info("Starting Monday.com CSV data fetch...")

        # Run the download_data_monday.py script with real-time output
        import subprocess
        import sys

        process = subprocess.Popen([sys.executable, 'download_data_monday.py'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True,
                                   bufsize=1)

        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                logger.info(f"Script output: {output.strip()}")

        return_code = process.poll()
        full_output = '\n'.join(output_lines)

        if return_code == 0:
            logger.info("Monday.com CSV data fetch completed successfully")
            return jsonify({
                "status": "success",
                "message": "Monday.com data fetched and saved successfully",
                "output": full_output
            })
        else:
            logger.error(
                f"Monday.com fetch failed with return code {return_code}")
            return jsonify({
                "status": "error",
                "message": f"Script failed with return code {return_code}",
                "output": full_output
            })

    except Exception as e:
        logger.error(f"Failed to run Monday.com fetch script: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_monday_csv_data', methods=['GET'])
@login_required
def get_monday_csv_data():
    try:
        DATA_DIR = Path("./data/downloads_monday")

        if not DATA_DIR.exists():
            return jsonify({
                "status":
                "error",
                "message":
                "No Monday.com data directory found. Please fetch data first."
            })

        # Find the latest Monday_Data CSV file
        csv_files = [
            f for f in DATA_DIR.glob("Monday_Data_*.csv") if f.is_file()
        ]

        if not csv_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No Monday.com CSV files found. Please fetch data first."
            })

        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading Monday.com data from: {latest_file}")

        # Load the CSV file
        df = pd.read_csv(latest_file)

        # Handle NaN values and convert to records
        df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        all_data = df_clean.to_dict('records')

        # Get unique groups if Group column exists
        groups = []
        if 'Group' in df.columns:
            groups = df['Group'].dropna().unique().tolist()

        logger.info(f"Loaded {len(all_data)} records from Monday.com CSV")
        logger.info(f"Found {len(groups)} groups: {groups}")

        return jsonify({
            "status": "success",
            "data": all_data,
            "filename": latest_file.name,
            "total_records": len(all_data),
            "groups": groups
        })

    except Exception as e:
        logger.error(f"Error loading Monday.com CSV data: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({
            "status":
            "error",
            "message":
            f"Failed to load Monday.com CSV data: {str(e)}"
        })


@app.route('/predictions')
@login_required
def predictions():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template("predictions.html")


@app.route('/generate_predictions', methods=['POST'])
@login_required
def generate_predictions():
    try:
        import subprocess
        import glob
        from pathlib import Path

        # Find the latest scheduled CSV file
        session_dir = Path("./sessions/sessions")
        scheduled_files = list(session_dir.glob("scheduled_*.csv"))

        if not scheduled_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No scheduled CSV files found in sessions directory."
            })

        # Get the latest scheduled file
        latest_scheduled = max(scheduled_files,
                               key=lambda f: f.stat().st_mtime)
        logger.info(f"Using latest scheduled file: {latest_scheduled}")

        # Find the latest model files
        models_dir = Path("./models")
        model_files = list(models_dir.glob("model_*.joblib"))
        meta_files = list(models_dir.glob("metadata_*.json"))

        if not model_files or not meta_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No trained model found. Please train a model first."
            })

        # Get the latest model and metadata files
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)

        # Prepare the prediction command
        predictions_output = "predictions.csv"
        cmd = [
            "python", "lead_prediction.py", "predict", "--model",
            str(latest_model), "--meta",
            str(latest_meta), "--input",
            str(latest_scheduled), "--output", predictions_output
        ]

        logger.info(f"Running prediction command: {' '.join(cmd)}")

        # Run the prediction
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Prediction failed: {result.stderr}")
            return jsonify({
                "status": "error",
                "message": f"Prediction failed: {result.stderr}"
            })

        # Read the predictions CSV
        predictions_df = pd.read_csv(predictions_output)

        # Handle NaN values before JSON serialization
        predictions_df = predictions_df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        })

        # Convert to JSON for frontend
        predictions_data = predictions_df.to_dict('records')

        # Get model info
        model_info = f"{latest_model.name}"

        logger.info(
            f"Generated {len(predictions_data)} predictions successfully")

        return jsonify({
            "status": "success",
            "data": predictions_data,
            "model_info": model_info,
            "input_file": latest_scheduled.name
        })

    except Exception as e:
        logger.error(f"Error in generate_predictions: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})


@app.route('/merge_calendly_csv_data', methods=['POST'])
@login_required
def merge_calendly_csv_data():
    try:
        import pandas as pd
        from pathlib import Path

        # Path to Calendly data
        calendly_path = Path('./calendly_enriched_events.csv')

        if not calendly_path.exists():
            return jsonify({
                "status":
                "error",
                "message":
                "Calendly data file not found. Please fetch Calendly data first."
            })

        # Find latest Monday.com CSV data
        DATA_DIR = Path("./data/downloads_monday")

        if not DATA_DIR.exists():
            return jsonify({
                "status":
                "error",
                "message":
                "No Monday.com data directory found. Please fetch Monday.com CSV data first."
            })

        # Find the latest Monday_Data CSV file
        csv_files = [
            f for f in DATA_DIR.glob("Monday_Data_*.csv") if f.is_file()
        ]

        if not csv_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No Monday.com CSV files found. Please fetch Monday.com CSV data first."
            })

        latest_monday_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using latest Monday.com CSV file: {latest_monday_file}")

        # Load Calendly data
        df_calendly = pd.read_csv(calendly_path)
        logger.info(f"Loaded {len(df_calendly)} Calendly records")

        # Load Monday.com CSV data
        df_monday = pd.read_csv(latest_monday_file)
        logger.info(f"Loaded {len(df_monday)} Monday.com records from CSV")

        # Check if Email column exists in Monday.com data
        if 'Email' not in df_monday.columns:
            return jsonify({
                "status":
                "error",
                "message":
                "Email column not found in Monday.com CSV data."
            })

        # Create a new column 'origin' in df_monday DataFrame
        df_monday['origin'] = 'LinkedIn'  # Default value

        # Check each email and update origin accordingly
        matched_emails = set()
        for email in df_calendly['invitee_email']:
            if pd.notna(email) and email in df_monday['Email'].values:
                df_monday.loc[df_monday['Email'] == email,
                              'origin'] = 'cold-email'
                matched_emails.add(email)

        # Count how many Monday.com records were updated
        records_updated = (df_monday['origin'] == 'cold-email').sum()

        logger.info(
            f"Found {len(matched_emails)} unique email matches between Calendly and Monday.com CSV data"
        )
        logger.info(
            f"Updated {records_updated} Monday.com records with cold-email origin"
        )
        logger.info(
            f"Origin value counts: {df_monday['origin'].value_counts().to_dict()}"
        )

        # Save the merged data back to the CSV file
        df_monday.to_csv(latest_monday_file, index=False)
        logger.info(
            f"Successfully updated Monday.com CSV file: {latest_monday_file}")

        return jsonify({
            "status":
            "success",
            "message":
            f"Successfully merged data. Found {len(matched_emails)} email matches. Origin column added to Monday.com CSV data.",
            "calendly_records":
            len(df_calendly),
            "monday_records":
            len(df_monday),
            "email_matches":
            len(matched_emails),
            "csv_file_used":
            latest_monday_file.name,
            "origin_counts":
            df_monday['origin'].value_counts().to_dict()
        })

    except Exception as e:
        logger.error(f"Error in merge_calendly_csv_data: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})


@app.route('/fetch_pipeline_data', methods=['POST'])
@login_required
def fetch_pipeline_data():
    try:
        logger.info("Starting data pipeline fetch...")

        # Run the data_pipeline.py script
        result = subprocess.run([sys.executable, 'data_pipeline.py'],
                                capture_output=True,
                                text=True,
                                cwd='.')

        if result.returncode != 0:
            logger.error(
                f"Data pipeline failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return jsonify({
                "status": "error",
                "message": f"Data pipeline failed: {result.stderr}"
            })

        logger.info("Data pipeline completed successfully")
        logger.info(f"Pipeline output: {result.stdout}")

        # Read the output CSV file created by data_pipeline.py
        output_file = Path('./combined_output.csv')

        if not output_file.exists():
            return jsonify({
                "status":
                "error",
                "message":
                "Output file not found. Pipeline may have failed."
            })

        # Load the CSV data
        df = pd.read_csv(output_file)

        # Handle NaN values and convert to records
        df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        all_data = df_clean.to_dict('records')

        logger.info(f"Loaded {len(all_data)} records from pipeline output")

        return jsonify({
            "status": "success",
            "data": all_data,
            "total_records": len(all_data),
            "message": "Data pipeline executed successfully"
        })

    except Exception as e:
        logger.error(f"Error in fetch_pipeline_data: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Failed to run data pipeline: {str(e)}"
        })


@app.route('/merge_calendly_data', methods=['POST'])
@login_required
def merge_calendly_data():
    try:
        import pandas as pd
        from pathlib import Path

        # Path to Calendly data
        calendly_path = Path('./calendly_enriched_events.csv')

        if not calendly_path.exists():
            return jsonify({
                "status":
                "error",
                "message":
                "Calendly data file not found. Please fetch Calendly data first."
            })

        # Find latest Monday.com data from sessions folder by timestamp
        session_dir = Path("./sessions/sessions")
        data_files = [
            f for f in session_dir.glob("data_*.json") if f.is_file()
        ]

        if not data_files:
            return jsonify({
                "status":
                "error",
                "message":
                "No Monday.com data files found in sessions directory."
            })

        # Get the latest timestamp by finding the most recent data file
        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)

        # Extract timestamp from filename (format: data_YYYY-MM-DD_HH-MM-SS.json)
        filename_parts = latest_file.stem.split('_')
        if len(filename_parts) >= 3:
            timestamp = f"{filename_parts[1]}_{filename_parts[2]}"
        else:
            return jsonify({
                "status":
                "error",
                "message":
                "Could not extract timestamp from data file."
            })

        logger.info(f"Using timestamp: {timestamp} for all group files")

        # Load Calendly data
        df_calendly = pd.read_csv(calendly_path)
        logger.info(f"Loaded {len(df_calendly)} Calendly records")

        # Load all 7 groups from the same timestamp and combine them
        group_names = [
            'scheduled', 'unqualified', 'won', 'cancelled', 'noshow',
            'proposal', 'lost'
        ]
        all_monday_data = []

        for group_name in group_names:
            group_file = session_dir / f"{group_name}_{timestamp}.csv"
            if group_file.exists():
                df_group = pd.read_csv(group_file)
                # Add group column to track which group each record belongs to
                df_group['group'] = group_name
                all_monday_data.append(df_group)
                logger.info(
                    f"Loaded {len(df_group)} records from {group_name} group")
            else:
                logger.warning(f"Group file not found: {group_file}")

        if not all_monday_data:
            return jsonify({
                "status":
                "error",
                "message":
                f"No group CSV files found for timestamp {timestamp}."
            })

        # Combine all groups into a single DataFrame
        df_monday = pd.concat(all_monday_data, ignore_index=True)
        logger.info(
            f"Combined total of {len(df_monday)} Monday.com records from all groups"
        )

        # Check if Email column exists in Monday.com data
        if 'Email' not in df_monday.columns:
            return jsonify({
                "status":
                "error",
                "message":
                "Email column not found in Monday.com data."
            })

        # Create a new column 'origin' in df_monday DataFrame
        df_monday['origin'] = 'LinkedIn'  # Default value

        # Check each email and update origin accordingly
        matched_emails = set()
        for email in df_calendly['invitee_email']:
            if pd.notna(email) and email in df_monday['Email'].values:
                df_monday.loc[df_monday['Email'] == email,
                              'origin'] = 'cold-email'
                matched_emails.add(email)

        # Count how many Monday.com records were updated
        records_updated = (df_monday['origin'] == 'cold-email').sum()

        logger.info(
            f"Found {len(matched_emails)} unique email matches between Calendly and Monday.com data"
        )
        logger.info(
            f"Updated {records_updated} Monday.com records with cold-email origin"
        )
        logger.info(
            f"Origin value counts: {df_monday['origin'].value_counts().to_dict()}"
        )

        # Save the merged data back by updating the JSON file
        # Load the original JSON structure
        with latest_file.open("r", encoding="utf-8") as file:
            json_data = json.load(file)

        # Update each group in the JSON with the merged data
        merged_json_data = {}
        for group_name in group_names:
            if group_name in json_data:
                # Get records for this group
                group_records = df_monday[df_monday['group'] ==
                                          group_name].copy()
                # Remove the group column before saving (keep original structure)
                group_records = group_records.drop('group', axis=1)
                merged_json_data[group_name] = group_records.to_dict('records')
            else:
                # Keep original data if group not found
                merged_json_data[group_name] = json_data.get(group_name, [])

        # Save merged data back to the JSON file
        with latest_file.open("w", encoding="utf-8") as file:
            json.dump(merged_json_data, file, indent=2)

        # Also save individual CSV files with the origin column
        for group_name in group_names:
            group_records = df_monday[df_monday['group'] == group_name].copy()
            if not group_records.empty:
                group_records = group_records.drop('group', axis=1)
                group_file = session_dir / f"{group_name}_{timestamp}.csv"
                group_records.to_csv(group_file, index=False)
                logger.info(f"Updated {group_name} CSV with origin column")

        logger.info("Successfully merged Calendly data with Monday.com data")

        return jsonify({
            "status":
            "success",
            "message":
            f"Successfully merged data. Found {len(matched_emails)} email matches. Origin column added to all groups.",
            "calendly_records":
            len(df_calendly),
            "monday_records":
            len(df_monday),
            "email_matches":
            len(matched_emails),
            "timestamp_used":
            timestamp,
            "origin_counts":
            df_monday['origin'].value_counts().to_dict()
        })

    except Exception as e:
        logger.error(f"Error in merge_calendly_data: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_ai():
    try:
        if request.method == 'POST':
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                data = request.get_json()
                user_input = data.get('user_input')
            else:
                user_input = request.form.get('user_input')

            if not user_input:
                return jsonify({
                    "status": "error",
                    "message": "No input provided"
                })

            logger.info("User input received for chatbot: %s", user_input)
            logger.info("Checking embeddings store status")

            if not hasattr(app, 'embeddings_store'):
                logger.error("Embeddings store not initialized")
                return jsonify({
                    "status":
                    "error",
                    "message":
                    "Embeddings store not initialized. Please apply filters first."
                })

            # Check if session directory exists and contains data files
            session_dir = Path('./sessions/sessions')
            data_files = [
                f for f in session_dir.glob('data_*.json') if f.is_file()
            ]
            if not data_files:
                return jsonify({
                    "status":
                    "error",
                    "message":
                    "No fetched data found. Please fetch data first."
                })

            # Retrieve the latest data file
            latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
            logger.info("Loading data from latest file: %s", latest_file.name)

            with latest_file.open("r", encoding="utf-8") as file:
                json_data = json.load(file)

            # Convert JSON data into DataFrames
            data = {k: pd.DataFrame(v) for k, v in json_data.items()}
            logger.info("Data loaded and converted to DataFrames.")

            # Assume some processing or filtering is applied using `process_data`
            st_date = request.json.get('start_date', '1900-01-01').strip()
            end_date = request.json.get('end_date', '2100-12-31').strip()
            column = request.json.get('filter_column',
                                      'Sales Call Date').strip()

            processed_df = process_data(data, st_date, end_date, column)

            table_html = processed_df.to_html(
                classes='table table-striped table-bordered',
                index=False,
                border=0)

            if not hasattr(app, 'embeddings_store'
                           ) or 'current' not in app.embeddings_store:
                return jsonify({
                    "status":
                    "error",
                    "message":
                    "No data available. Please apply filters first."
                })

            # Create embedding for the user's question
            question_embedding = create_embeddings(user_input)
            if not question_embedding:
                return jsonify({
                    "status": "error",
                    "message": "Failed to process your question"
                })

            # Find relevant context using cosine similarity
            import numpy as np

            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            similarities = []
            for item in app.embeddings_store['current']:
                similarity = cosine_similarity(question_embedding,
                                               item['embedding'])
                similarities.append(
                    (similarity, item['row_data'], item['index']))

            # Get top 3 most relevant chunks
            top_contexts = sorted(similarities, reverse=True)[:3]

            # Convert the filtered DataFrame to CSV string for context
            if hasattr(app, 'current_data'):
                csv_content = app.current_data.to_csv(index=False)
                logger.info(
                    f"Using current data for context:\n{csv_content[:500]}...")
            else:
                logger.warning("No current_data available")
                csv_content = ""

            context = "\n".join([
                f"Row {idx + 1}:" +
                "\n".join([f"{k}: {v}" for k, v in data.items()])
                for _, data, idx in top_contexts
            ])

            messages = [{
                "role":
                "system",
                "content":
                ("You are a helpful assistant analyzing sales data. Use this exact data to answer questions:\n\n"
                 + csv_content + "\n\n" + "Relevant context:\n" + context +
                 "\n\n" +
                 "Be very precise with numbers. Only use the exact values from the data provided. "
                 + "Do not make assumptions or combine metrics incorrectly.")
            }, {
                "role": "user",
                "content": user_input
            }]

            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="o4-mini",
                messages=messages,
            )
            answer = response.choices[0].message.content.strip()

            return jsonify({"status": "success", "response": answer})

    except Exception as e:
        logger.error("Error in chatbot_ai: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

    return render_template('chatbot.html')

@app.route('/apply_filters_cold_email', methods=['POST'])
def apply_filters_cold_email():
    try:
        logger.info("Starting apply_filters_cold_email")
        # Identify the latest data file in the sessions folder
        SESSION_DIR = Path("./sessions/sessions")
        data_files = [
            f for f in SESSION_DIR.glob("data_*.json") if f.is_file()
        ]
        logger.info(f"Found {len(data_files)} data files")
        latest_file = max(data_files,
                          key=lambda f: f.stat().st_mtime,
                          default=None)

        if not latest_file:
            return jsonify({
                "status":
                "error",
                "message":
                "No data files found in the session directory."
            })

        logger.info(f"Loading data from {latest_file}")
        with latest_file.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
            # Convert JSON data into DataFrames and filter by origin='cold-email'
            data = {}
            for key, items in json_data.items():
                df = pd.DataFrame(items)
                # Filter only cold-email origin data
                if 'origin' in df.columns:
                    df_filtered = df[df['origin'] == 'cold-email']
                    logger.info(f"Group {key}: {len(df)} total rows, {len(df_filtered)} cold-email rows")
                else:
                    # If no origin column, assume no cold-email data
                    df_filtered = pd.DataFrame(columns=df.columns)
                    logger.info(f"Group {key}: No origin column found, using empty DataFrame")
                
                data[key] = df_filtered

        st_date = request.json.get('start_date', '1900-01-01').strip()
        end_date = request.json.get('end_date', '2100-12-31').strip()
        filter_column = request.json.get('filter_column', 'Sales Call Date')
        if not filter_column:
            filter_column = 'Sales Call Date'  # Default value if none provided
        logger.info(
            f"Processing cold-email data with date range {st_date} to {end_date} on column {filter_column}"
        )
        selected_owners = request.json.get('selected_owners', [])
        
        # Process the filtered data (only cold-email origin)
        processed_df = process_data_COLD_EMAIL(data, st_date, end_date, filter_column)

        if selected_owners:
            # Filter data by selected owners (excluding any existing Total row)
            filtered_df = processed_df[processed_df['Owner'].isin(selected_owners)]
            
            # Recalculate totals for filtered data
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            totals_dict = {'Owner': 'Total'}
            
            for col in numeric_cols:
                if col != 'Owner':
                    if '%' in col or 'Rate' in col:
                        # For percentage columns, calculate mean
                        totals_dict[col] = filtered_df[col].mean()
                    else:
                        # For other numeric columns, calculate sum
                        totals_dict[col] = filtered_df[col].sum()
            
            # Create totals row DataFrame
            totals_row = pd.DataFrame([totals_dict])
            
            # Combine filtered data with new totals row
            processed_df = pd.concat([filtered_df, totals_row], ignore_index=True)
        
        logger.info(
            f"Cold-email data processing complete. Processed DataFrame has {len(processed_df)} rows"
        )

        processed_df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        },
                             inplace=True)

        numeric_columns = processed_df.select_dtypes(
            include=[np.number]).columns
        for col in numeric_columns:
            processed_df[col] = processed_df[col].astype(float)

        table_html = processed_df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            border=0)
        return jsonify({"status": "success", "table": table_html})

    except Exception as e:
        logger.error("Error in apply_filters_cold_email: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)