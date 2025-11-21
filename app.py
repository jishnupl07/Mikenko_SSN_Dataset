# app.py
# Streamlit app for Mikenko dataset collection (MySQL backend via mysql-connector-python)
# Replaces the Excel backend with direct MySQL queries (connector + cursor).
# Commits once per Save button press.

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import find_peaks, butter, filtfilt, detrend
import tempfile
import torch
import torch.nn as nn
import mysql.connector
from mysql.connector import Error

# ---------- EXCEL->MYSQL Utilities ----------
import pandas as pd
from datetime import datetime

# ---- DB CONFIG - EDIT THESE BEFORE RUNNING ----
DB_CONFIG = {
    "host": "localhost",       # e.g. "127.0.0.1" or "db.example.com"
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "ssn_db",
}
# -----------------------------------------------

# Config (video + folders)
VIDEO_ROOT = "SSN/Video_Dataset"
GRAPH_ROOT = "SSN/Graphs_Dataset"
os.makedirs(VIDEO_ROOT, exist_ok=True)
os.makedirs(GRAPH_ROOT, exist_ok=True)

# Column names used in code (also used in SQL)
COL_DIGITAL = "Digital_ID"
COL_NAME = "Name"
COL_GENDER = "Gender"
COL_SECTION = "Section"
COL_DEPT = "Department"
# DB friendly email column (no spaces)
COL_EMAIL = "SSN_Email_Id"
COL_DOB = "DOB"
COL_DOP = "DOP"
COL_MIKENKO_HR = "Mikenko_HR"
COL_HR = "HR"
COL_MIKENKO_BP_SYS = "Mikenko_BP_Sys"
COL_MIKENKO_BP_DIA = "Mikenko_BP_dia"
COL_BP_SYS = "BP_sys"
COL_BP_DIA = "BP_dia"
COL_SPO2 = "SPO2"
COL_MODEL_NO = "Model_No"

# ---------- MySQL helper functions ----------

def get_connection():
    """Return a new mysql.connector connection using DB_CONFIG."""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG.get("port", 3306),
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            autocommit=False
        )
        return conn
    except Error as e:
        st.error(f"Database connection error: {e}")
        return None

def fetch_student_by_digital(digital_id):
    """
    Return a dict with student row (column -> value) for the given Digital_ID,
    or None if not found or on error.
    Tries integer match first if digital_id looks like an int.
    """
    if not digital_id:
        return None
    did = str(digital_id).strip()
    conn = get_connection()
    if conn is None:
        return None
    try:
        cur = conn.cursor(dictionary=True)
        # try integer match first (if digital_id looks like an int)
        try:
            did_int = int(did)
            query = f"SELECT * FROM SSN_Students WHERE {COL_DIGITAL} = %s LIMIT 1"
            cur.execute(query, (did_int,))
        except Exception:
            # fallback to string match
            query = f"SELECT * FROM SSN_Students WHERE {COL_DIGITAL} = %s LIMIT 1"
            cur.execute(query, (did,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row
    except Error as e:
        st.error(f"Error fetching student: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return None

def update_student_vitals(digital_id, values_dict):
    """
    Upsert vitals for student with Digital_ID = digital_id.
    Uses INSERT ... ON DUPLICATE KEY UPDATE to insert when missing or update when present.
    Returns (ok:bool, err_msg:str|None)
    NOTE: Digital_ID must be defined UNIQUE or have a UNIQUE index for ON DUPLICATE KEY UPDATE to work.
    """
    if not digital_id:
        return False, "Digital ID missing"
    if not values_dict:
        return False, "No values to update"

    did = str(digital_id).strip()
    # Build insert columns (include Digital_ID) and values
    insert_cols = [COL_DIGITAL] + list(values_dict.keys())
    placeholders = ["%s"] * len(insert_cols)
    insert_vals = [did] + [values_dict[k] for k in values_dict.keys()]

    # Build ON DUPLICATE KEY UPDATE clause: col=VALUES(col)
    update_parts = []
    for col in values_dict.keys():
        # Use VALUES(col) to keep the inserted value on duplicate key
        update_parts.append(f"`{col}` = VALUES(`{col}`)")
    update_clause = ", ".join(update_parts)

    insert_sql = f"INSERT INTO SSN_Students ({', '.join('`'+c+'`' for c in insert_cols)}) VALUES ({', '.join(placeholders)}) ON DUPLICATE KEY UPDATE {update_clause}"

    conn = get_connection()
    if conn is None:
        return False, "DB connection failed"
    try:
        cur = conn.cursor()
        cur.execute(insert_sql, insert_vals)
        conn.commit()
        cur.close()
        conn.close()
        return True, None
    except Error as e:
        # fallback: try simple UPDATE in case ON DUPLICATE KEY isn't available due to schema
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            # Build UPDATE statement
            set_parts = []
            params = []
            for col, val in values_dict.items():
                set_parts.append(f"`{col}` = %s")
                params.append(val)
            params.append(did)
            set_clause = ", ".join(set_parts)
            update_sql = f"UPDATE SSN_Students SET {set_clause} WHERE {COL_DIGITAL} = %s"
            cur.execute(update_sql, params)
            if cur.rowcount == 0:
                conn.rollback()
                cur.close()
                conn.close()
                return False, f"No student with Digital_ID {did} found."
            conn.commit()
            cur.close()
            conn.close()
            return True, None
        except Error as e2:
            try:
                conn.close()
            except Exception:
                pass
            return False, f"{e} ; fallback update error: {e2}"

# ---------- BP/HR logic (unchanged) ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bp_model_lstm.pth")

def bp_extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    red_intensities = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2]
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
    cap.release()
    return red_intensities

def bp_preprocess_signal(intensities, fps):
    detrended_signal = detrend(intensities)
    nyquist = 0.5 * fps
    low = 0.5 / nyquist
    high = 4 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)
    return filtered_signal

def bp_calculate_bpm(peaks, frame_count, fps):
    num_beats = len(peaks)
    duration_in_seconds = frame_count / fps if fps > 0 else frame_count / 30.0
    bpm = (num_beats / duration_in_seconds) * 60 if duration_in_seconds>0 else 0
    return bpm

def extract_bpm(video_path):
    intensities = bp_extract_red_intensity(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    cap.release()
    filtered_signal = bp_preprocess_signal(intensities, fps)
    try:
        distance = max(1, int(fps * 0.5))
        peaks, _ = find_peaks(filtered_signal, distance=distance, prominence=0.01)
    except Exception:
        peaks = np.array([])
    return bp_calculate_bpm(peaks, len(intensities), fps)

def load_video_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (64, 64))
        frames.append(frame_resized[:, :, 2] / 255.0)
        count += 1
    cap.release()
    if len(frames) < num_frames:
        padding = np.zeros((num_frames - len(frames), 64, 64))
        if len(frames) > 0:
            frames = np.concatenate((np.stack(frames, axis=0), padding), axis=0)
        else:
            frames = padding
    frames = np.stack(frames, axis=0) if isinstance(frames, list) else frames
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

class BPRegressionModel(nn.Module):
    def __init__(self):
        super(BPRegressionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, bpm):
        batch_size, channels, frames, height, width = x.size()
        cnn_out = self.cnn(x).view(batch_size, 25, -1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_last_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_last_out, bpm.view(-1, 1)), dim=1)
        output = self.fc(combined)
        return output

@st.cache_resource
def load_bp_model():
    model = BPRegressionModel()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device('cpu'),
            weights_only=False
        )
    )
    model.eval()
    return model

def predict_bp(model, video_path):
    bpm = extract_bpm(video_path)
    video_data = load_video_frames(video_path)
    with torch.no_grad():
        prediction = model(video_data.unsqueeze(1), torch.tensor([bpm]).float())
    systolic, diastolic = prediction.squeeze().tolist()
    return systolic, diastolic, bpm

# BPM helpers
def bpm_extract_red_intensity(video_path, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    red_intensities = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2]
        red_intensities.append(float(np.mean(red_channel)))
        if progress_callback is not None and frame_count > 0:
            progress_callback((i + 1) / frame_count, f"Analyzing frame {i+1}/{frame_count}")
    cap.release()
    return red_intensities, frame_count, fps

def bpm_normalize_intensities(intensities):
    arr = np.array(intensities)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def bpm_find_peaks_in_intensity(intensities, distance=15, prominence=0.05):
    peaks, properties = find_peaks(intensities, distance=distance, prominence=prominence)
    return peaks, properties

def bpm_calculate_bpm_from_peaks(peaks, duration_in_seconds):
    if duration_in_seconds <= 0:
        return 0.0
    num_beats = len(peaks)
    return (num_beats / duration_in_seconds) * 60.0

def bpm_create_bpm_plot(intensities, peaks, video_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(intensities, label='Normalized Red Intensity')
    if len(peaks) > 0:
        ax.plot(peaks, np.array(intensities)[peaks], "x", label=f'Detected Peaks ({len(peaks)})')
    ax.set_title(f'Red Intensity and Peaks for {video_name}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Normalized Average Red Intensity')
    ax.legend()
    plt.tight_layout()
    return fig

def bpm_signal_quality_check(intensities, peaks, properties, fps):
    if len(peaks) == 0:
        return False, 0.0
    signal_power = np.mean(np.array(intensities)[peaks] ** 2)
    noise_power = np.var(intensities)
    snr = float(signal_power / (noise_power + 1e-10))
    if fps == 0:
        fps = 30
    min_peaks = (len(intensities) / fps) * (30 / 60)
    if len(peaks) < max(1, int(min_peaks)):
        return False, snr
    if snr < 2:
        return False, snr
    return True, snr

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="Mikenko - Health Monitor (MySQL)")

# initialize session_state slots
if "tmp_video_path" not in st.session_state:
    st.session_state["tmp_video_path"] = None
if "tmp_video_name" not in st.session_state:
    st.session_state["tmp_video_name"] = None
if "hr_result" not in st.session_state:
    st.session_state["hr_result"] = None
if "bp_result" not in st.session_state:
    st.session_state["bp_result"] = None

# Sidebar UI
st.sidebar.header("Student Data (MySQL)")
if st.sidebar.button("Clear data"):
    # Clear internal keys
    keys_to_clear_none = [
        "tmp_video_path", "tmp_video_name",
        "hr_result", "bp_result",
        "excel_did",
    ]
    for k in keys_to_clear_none:
        st.session_state[k] = None

    # Reset text widget keys to empty string so inputs remain editable
    text_widget_keys = [
        "mikenko_hr", "hr_actual",
        "mikenko_bp_sys", "mikenko_bp_dia",
        "bp_sys", "bp_dia",
        "spo2", "model_no", "dob_input"
    ]
    for k in text_widget_keys:
        st.session_state[k] = ""

    # Keep DOP if you want; otherwise uncomment the next line
    # st.session_state["dop"] = datetime.today().strftime("%Y-%m-%d")

    st.success("Cleared session data (DOP preserved).")

DIGITAL_COL_INFO = f"Using DB column `{COL_DIGITAL}` for lookup."
st.sidebar.info(DIGITAL_COL_INFO)

search_id = st.sidebar.text_input("Digital ID", value="", key="excel_did")
if st.sidebar.button("Search student"):
    if not search_id.strip():
        st.sidebar.warning("Enter a Digital ID before searching.")
    else:
        row = fetch_student_by_digital(search_id)
        if row is None:
            st.sidebar.error(f"Student with Digital ID '{search_id}' not found or DB error.")
        else:
            st.sidebar.success(f"Found student: {row.get(COL_NAME,'(no name)')}")

# Try to load student if search_id exists
student_row = None
if search_id and search_id.strip():
    student_row = fetch_student_by_digital(search_id.strip())

# Prefill session_state from DB row (before creating widgets)
if student_row is not None:
    def safe_str(v):
        if v is None:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except Exception:
            pass
        return str(v)

    # map DB values to our widget defaults (only set if not already set by the user)
    if "mikenko_hr" not in st.session_state or not st.session_state.get("mikenko_hr"):
        st.session_state["mikenko_hr"] = safe_str(student_row.get(COL_MIKENKO_HR, ""))
    if "hr_actual" not in st.session_state or not st.session_state.get("hr_actual"):
        st.session_state["hr_actual"] = safe_str(student_row.get(COL_HR, ""))
    if "mikenko_bp_sys" not in st.session_state or not st.session_state.get("mikenko_bp_sys"):
        st.session_state["mikenko_bp_sys"] = safe_str(student_row.get(COL_MIKENKO_BP_SYS, ""))
    if "mikenko_bp_dia" not in st.session_state or not st.session_state.get("mikenko_bp_dia"):
        st.session_state["mikenko_bp_dia"] = safe_str(student_row.get(COL_MIKENKO_BP_DIA, ""))
    if "bp_sys" not in st.session_state or not st.session_state.get("bp_sys"):
        st.session_state["bp_sys"] = safe_str(student_row.get(COL_BP_SYS, ""))
    if "bp_dia" not in st.session_state or not st.session_state.get("bp_dia"):
        st.session_state["bp_dia"] = safe_str(student_row.get(COL_BP_DIA, ""))
    if "spo2" not in st.session_state or not st.session_state.get("spo2"):
        st.session_state["spo2"] = safe_str(student_row.get(COL_SPO2, ""))
    if "model_no" not in st.session_state or not st.session_state.get("model_no"):
        st.session_state["model_no"] = safe_str(student_row.get(COL_MODEL_NO, ""))
    if "dob_input" not in st.session_state or not st.session_state.get("dob_input"):
        st.session_state["dob_input"] = safe_str(student_row.get(COL_DOB, ""))

    # DOP & DOB prefill
    raw_dop = student_row.get(COL_DOP, "")
    try:
        if raw_dop:
            st.session_state["dop"] = pd.to_datetime(raw_dop).strftime("%Y-%m-%d")
        else:
            if "dop" not in st.session_state or not st.session_state.get("dop"):
                st.session_state["dop"] = datetime.today().strftime("%Y-%m-%d")
    except Exception:
        if "dop" not in st.session_state or not st.session_state.get("dop"):
            st.session_state["dop"] = datetime.today().strftime("%Y-%m-%d")

    #raw_dob = student_row.get(COL_DOB, "")


# Show student basic details (if loaded)
if student_row is not None:
    st.sidebar.markdown("**Student details (from DB)**")
    st.sidebar.write(f"Name: {student_row.get(COL_NAME,'')}")
    st.sidebar.write(f"Gender: {student_row.get(COL_GENDER,'')}")
    st.sidebar.write(f"Section: {student_row.get(COL_SECTION,'')}")
    st.sidebar.write(f"Department: {student_row.get(COL_DEPT,'')}")
    st.sidebar.write(f"Email: {student_row.get(COL_EMAIL,'')}")
else:
    st.sidebar.info("No student loaded. Use Digital ID search.")

st.sidebar.markdown("---")
st.sidebar.markdown("**DOB (yyyy-mm-dd)**")
# dob_input keyed and backed by st.session_state["dob_input"]
dob_input = st.sidebar.text_input("", value=st.session_state.get("dob_input",""), key="dob_input", help="Enter DOB in dd-mm-yyyy format. Will be saved as dd-mm-yyyy string in DB.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Vitals to save**")
dop_input = st.sidebar.text_input("DOP (YYYY-MM-DD)", value=st.session_state.get("dop", datetime.today().strftime("%Y-%m-%d")), key="dop")
mikenko_hr_input = st.sidebar.text_input("Mikenko_HR", value=str(st.session_state.get("mikenko_hr","")), key="mikenko_hr")
hr_input = st.sidebar.text_input("HR (actual)", value=str(st.session_state.get("hr_actual","")), key="hr_actual")

mikenko_bp_sys_input = st.sidebar.text_input("Mikenko_BP_Sys (systolic)", value=str(st.session_state.get("mikenko_bp_sys","")), key="mikenko_bp_sys")
mikenko_bp_dia_input = st.sidebar.text_input("Mikenko_BP_dia (diastolic)", value=str(st.session_state.get("mikenko_bp_dia","")), key="mikenko_bp_dia")

bp_sys_input = st.sidebar.text_input("BP_sys (actual systolic)", value=str(st.session_state.get("bp_sys","")), key="bp_sys")
bp_dia_input = st.sidebar.text_input("BP_dia (actual diastolic)", value=str(st.session_state.get("bp_dia","")), key="bp_dia")

spo2_input = st.sidebar.text_input("SPO2 (%)", value=str(st.session_state.get("spo2","")), key="spo2")
model_no_input = st.sidebar.text_input("Model No", value=str(st.session_state.get("model_no","")), key="model_no")

# Save uploaded video to DB student's dataset folder
def save_uploaded_video_to_student(digital_id, uploaded_file):
    """
    Save uploaded_file into dataset/<digital_id>.<ext> (no subfolders).
    Overwrites existing file with same name.
    uploaded_file must have .name and .getbuffer()
    """
    if not uploaded_file:
        return None, "No file provided"
    os.makedirs(VIDEO_ROOT, exist_ok=True)
    _, ext = os.path.splitext(uploaded_file.name)
    dest_path = os.path.join(VIDEO_ROOT, f"{digital_id}{ext}")
    try:
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return dest_path, None
    except Exception as e:
        return None, str(e)

if st.sidebar.button("Save uploaded video to selected student (if any)"):
    if not search_id or not search_id.strip():
        st.sidebar.warning("Search and select a student first before saving uploaded video.")
    else:
        tmp_name = st.session_state.get("tmp_video_name")
        tmp_path = st.session_state.get("tmp_video_path")
        if tmp_path and tmp_name:
            try:
                with open(tmp_path, "rb") as f:
                    class FakeUp:
                        def __init__(self, name, data):
                            self.name = name
                            self._data = data
                        def getbuffer(self):
                            return self._data
                    fake = FakeUp(tmp_name, f.read())
                saved_path, err = save_uploaded_video_to_student(search_id.strip(), fake)
                if err:
                    st.sidebar.error(f"Could not save video: {err}")
                else:
                    st.sidebar.success(f"Saved uploaded video to {saved_path}")
            except Exception as e:
                st.sidebar.error(f"Error saving uploaded file to student: {e}")
        else:
            st.sidebar.info("No uploaded video found in this session.")

# Save vitals to MySQL on button press (commits once)
if st.sidebar.button("Save vitals to DB"):
    if not search_id or not search_id.strip():
        st.sidebar.warning("Search and select a student first before saving.")
    else:
        # Validate DOP
        try:
            if dop_input:
                datetime.strptime(dop_input, "%Y-%m-%d")
        except Exception:
            st.sidebar.error("DOP must be in YYYY-MM-DD format.")
        else:
            # Validate DOB in dd-mm-yyyy and convert to YYYY-MM-DD for DATE column
            if dob_input:
                try:
                    parsed_dob = datetime.strptime(dob_input.strip(), "%Y-%m-%d")
                    dob_to_save = parsed_dob.strftime("%Y-%m-%d")
                except Exception:
                    st.sidebar.error("DOB must be in yyyy-mm-dd format .")
                    dob_to_save = None
            else:
                dob_to_save = ""
            if dob_to_save is not None:
                # prepare values dict
                vals = {
                    COL_DOP: dop_input or None,
                    COL_MIKENKO_HR: mikenko_hr_input or None,
                    COL_HR: hr_input or None,
                    COL_MIKENKO_BP_SYS: mikenko_bp_sys_input or None,
                    COL_MIKENKO_BP_DIA: mikenko_bp_dia_input or None,
                    COL_BP_SYS: bp_sys_input or None,
                    COL_BP_DIA: bp_dia_input or None,
                    COL_SPO2: spo2_input or None,
                    COL_MODEL_NO: model_no_input or None,
                    COL_DOB: dob_to_save or None
                }
                ok, err = update_student_vitals(search_id.strip(), vals)
                if not ok:
                    st.sidebar.error(f"Error saving to DB: {err}")
                else:
                    st.sidebar.success("Vitals & DOB & SPO2 saved to DB.")
                    # re-fetch to reflect new values in UI / session_state
                    updated = fetch_student_by_digital(search_id.strip())
                    if updated:
                        # refresh session_state with latest DB values
                        st.session_state["mikenko_hr"] = str(updated.get(COL_MIKENKO_HR, "")) or ""
                        st.session_state["hr_actual"] = str(updated.get(COL_HR, "")) or ""
                        st.session_state["mikenko_bp_sys"] = str(updated.get(COL_MIKENKO_BP_SYS, "")) or ""
                        st.session_state["mikenko_bp_dia"] = str(updated.get(COL_MIKENKO_BP_DIA, "")) or ""
                        st.session_state["bp_sys"] = str(updated.get(COL_BP_SYS, "")) or ""
                        st.session_state["bp_dia"] = str(updated.get(COL_BP_DIA, "")) or ""
                        st.session_state["spo2"] = str(updated.get(COL_SPO2, "")) or ""
                        st.session_state["model_no"] = str(updated.get(COL_MODEL_NO, "")) or ""
                        # DOP and DOB formatting
                        try:
                            if updated.get(COL_DOP):
                                st.session_state["dop"] = pd.to_datetime(updated.get(COL_DOP)).strftime("%Y-%m-%d")
                        except Exception:
                            pass
                        try:
                            if updated.get(COL_DOB):
                                st.session_state["dob_input"] = pd.to_datetime(updated.get(COL_DOB)).strftime("%Y-%m-%d")
                            else:
                                st.session_state["dob_input"] = ""
                        except Exception:
                            st.session_state["dob_input"] = str(updated.get(COL_DOB,"")) or ""

# Main area: upload + compute HR/BP (unchanged)
st.title("Health Monitor — HR & BP (MySQL-backed)")

uploaded_file = st.file_uploader("Upload one video (used for HR and/or BP).", type=["mp4","mov","avi","mkv"], accept_multiple_files=False, key="main_upload")

if uploaded_file is not None:
    if st.session_state.get("tmp_video_name") != uploaded_file.name or st.session_state.get("tmp_video_path") is None:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tmpf.write(uploaded_file.getvalue())
        tmpf.close()
        st.session_state["tmp_video_path"] = tmpf.name
        st.session_state["tmp_video_name"] = uploaded_file.name
        st.success(f"Saved uploaded video to temporary path: {tmpf.name}")
    else:
        st.info(f"Using existing uploaded video: {st.session_state.get('tmp_video_name')}")

    # Auto-save to dataset/<DigitalID>.<ext> if a student is selected
    if search_id and student_row is not None:
        try:
            with open(st.session_state["tmp_video_path"], "rb") as f:
                class FakeUp:
                    def __init__(self, name, data):
                        self.name = name
                        self._data = data
                    def getbuffer(self):
                        return self._data
                fake = FakeUp(st.session_state["tmp_video_name"], f.read())
            saved_path, err = save_uploaded_video_to_student(search_id.strip(), fake)
            if err:
                st.warning(f"Uploaded video saved to temp but could not be copied to dataset folder: {err}")
            else:
                st.success(f"Uploaded video automatically saved as: {os.path.basename(saved_path)}")
        except Exception as e:
            st.warning(f"Could not auto-save uploaded video to dataset folder: {e}")
    else:
        st.info("No student selected — uploaded video is in a temporary file. Select a student to automatically save it, or use the sidebar button.")

col_buttons = st.columns([1,1])
hr_btn = col_buttons[0].button("Calculate HR (BPM)")
bp_btn = col_buttons[1].button("Calculate BP")

hr_metrics_ph = st.empty()
hr_plot_ph = st.empty()
bp_metrics_ph = st.empty()

# Re-render persisted HR/BP results if present
if st.session_state.get("hr_result"):
    hrr = st.session_state["hr_result"]
    if hrr.get("snr") is not None:
        hr_metrics_ph.success(f"Calculated BPM: {hrr['bpm']:.2f} (SNR={hrr['snr']:.2f})")
    else:
        hr_metrics_ph.success(f"Calculated BPM: {hrr['bpm']:.2f}")
    try:
        normalized = np.array(hrr["normalized"])
        peaks = np.array(hrr["peaks"], dtype=int)
        fig = bpm_create_bpm_plot(normalized, peaks, hrr.get("video_name", "video"))
        hr_plot_ph.pyplot(fig)
    except Exception:
        hr_plot_ph.info("HR plot could not be reconstructed.")

if st.session_state.get("bp_result"):
    bpr = st.session_state["bp_result"]
    bp_metrics_ph.success("BP (persisted):")
    bp_metrics_ph.write({
        "Predicted Systolic": f"{bpr['systolic']:.1f}",
        "Predicted Diastolic": f"{bpr['diastolic']:.1f}",
        "Internal BPM used by model": f"{bpr['internal_bpm']:.2f}"
    })

# HR processing
if hr_btn:
    tmp_path = st.session_state.get("tmp_video_path")
    if tmp_path is None:
        st.warning("Please upload a video first.")
    else:
        st.info("Starting HR (BPM) processing...")
        progress = st.progress(0, text="Starting BPM analysis...")
        try:
            intensities, frame_count, fps = bpm_extract_red_intensity(tmp_path, progress_callback=lambda p, t: progress.progress(min(1.0, p), text=t))
            progress.empty()
            if intensities is None or len(intensities) == 0:
                st.error("Could not extract signal from video.")
            else:
                normalized = bpm_normalize_intensities(intensities)
                distance = max(1, int(fps * 0.5))
                peaks, props = bpm_find_peaks_in_intensity(normalized, distance=distance, prominence=0.05)
                is_valid, snr = bpm_signal_quality_check(normalized, peaks, props, fps)
                duration = frame_count / fps if fps > 0 else 0.0
                bpm_val = bpm_calculate_bpm_from_peaks(peaks, duration)
                if not is_valid:
                    hr_metrics_ph.warning(f"Low signal quality (SNR={snr:.2f}). BPM={bpm_val:.2f} may be unreliable.")
                else:
                    hr_metrics_ph.success(f"Calculated BPM: {bpm_val:.2f} (SNR={snr:.2f})")
                fig = bpm_create_bpm_plot(normalized, peaks, st.session_state.get("tmp_video_name") or "video")
                hr_plot_ph.pyplot(fig)

                # Save graph file (Graphs/<DigitalID>.png)
                try:
                    digital_id_src = search_id.strip() if search_id and search_id.strip() else os.path.splitext(st.session_state.get("tmp_video_name",""))[0] or "unknown"
                    digital_id_safe = re.sub(r'[^A-Za-z0-9._-]', '_', digital_id_src)
                    graph_path = os.path.join(GRAPH_ROOT, f"{digital_id_safe}.png")
                    fig.savefig(graph_path, dpi=150, bbox_inches='tight')
                    st.success(f"Saved HR graph to `{graph_path}`")
                except Exception as e:
                    st.warning(f"Could not save HR graph: {e}")

                # persist HR results into session
                st.session_state["hr_result"] = {
                    "bpm": float(bpm_val),
                    "snr": float(snr) if snr is not None else None,
                    "normalized": normalized.tolist(),
                    "peaks": peaks.tolist(),
                    "video_name": st.session_state.get("tmp_video_name") or "video",
                    "fps": float(fps)
                }
                # populate HR field in sidebar
                st.session_state["hr_actual"] = f"{bpm_val:.2f}"
        except Exception as e:
            progress.empty()
            st.error(f"BPM processing error: {e}")

# BP processing
if bp_btn:
    tmp_path = st.session_state.get("tmp_video_path")
    if tmp_path is None:
        st.warning("Please upload a video first.")
    else:
        try:
            bp_model = load_bp_model()
            model_load_error = None
        except Exception as e:
            bp_model = None
            model_load_error = str(e)

        if model_load_error is not None:
            st.error(f"BP model load error: {model_load_error}")
        else:
            st.info("Calculating Blood Pressure (this may take a moment)...")
            try:
                with st.spinner("Calculating Blood Pressure..."):
                    systolic, diastolic, internal_bpm = predict_bp(bp_model, tmp_path)
                bp_metrics_ph.success("BP processing complete.")
                bp_metrics_ph.write({
                    "Predicted Systolic": f"{systolic:.1f}",
                    "Predicted Diastolic": f"{diastolic:.1f}",
                    "Internal BPM used by model": f"{internal_bpm:.2f}"
                })
                st.session_state["bp_result"] = {
                    "systolic": float(systolic),
                    "diastolic": float(diastolic),
                    "internal_bpm": float(internal_bpm),
                    "video_name": st.session_state.get("tmp_video_name") or "video"
                }
                # populate sidebar fields for easy save
                st.session_state["mikenko_bp_sys"] = f"{int(round(systolic))}"
                st.session_state["mikenko_bp_dia"] = f"{int(round(diastolic))}"
                st.session_state["mikenko_hr"] = f"{internal_bpm:.2f}"
            except Exception as e:
                st.error(f"BP processing error: {e}")

st.markdown("---")
st.write("Tip: Search student by Digital ID, then upload a single video and run HR and/or BP. Use 'Save vitals to DB' to commit changes.")
