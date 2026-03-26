import streamlit as st
import os
import random
import sqlite3

# --- 1. Page Configuration ---
st.set_page_config(page_title="Model Evaluation (A/B Test)", layout="wide")

# --- 2. Configuration & Validation ---
EVAL_DIR = os.environ.get('CAESAR_HUMAN_EVAL_DIR', '.')

QUERY_DIR = os.path.join(EVAL_DIR, "query")
SOURCE_A_DIR = os.path.join(EVAL_DIR, "source_a")
SOURCE_B_DIR = os.path.join(EVAL_DIR, "source_b")
DB_FILE = os.path.join(EVAL_DIR, "results_ab.db")

for directory in [QUERY_DIR, SOURCE_A_DIR, SOURCE_B_DIR]:
    if not os.path.isdir(directory):
        st.error(f"CRITICAL ERROR: Required subdirectory is missing: {directory}")
        st.stop()

# --- 3. Initialize Session State ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "current_pair" not in st.session_state:
    st.session_state.current_pair = 0

# --- 4. Database Functions ---
def init_db():
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comparative_evals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_file TEXT,
                output_1_file TEXT,
                output_2_file TEXT,
                user_name TEXT,
                winner_selection TEXT,
                winning_file TEXT
            )
        ''')
        conn.commit()

def save_result(query_name, out1_file, out2_file, user_name, selection, winning_file):
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO comparative_evals
            (query_file, output_1_file, output_2_file, user_name, winner_selection, winning_file)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_name, out1_file, out2_file, user_name, selection, winning_file))
        conn.commit()

def reset_user_ratings(user_name):
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM comparative_evals WHERE user_name = ?', (user_name,))
        conn.commit()

def delete_all_records():
    """Admin only: Wipes the entire database table."""
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM comparative_evals')
        conn.commit()

def get_finished_user_count(total_required):
    """Counts how many unique users have completed all evaluations."""
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        # Counts users who have entries for every unique query_file
        cursor.execute('''
            SELECT COUNT(*) FROM (
                SELECT user_name
                FROM comparative_evals
                GROUP BY user_name
                HAVING COUNT(DISTINCT query_file) >= ?
            )
        ''', (total_required,))
        return cursor.fetchone()[0]

# --- 5. File Loading ---
def load_files():
    files_q = sorted([f for f in os.listdir(QUERY_DIR) if f.endswith('.txt')])
    files_a = sorted([f for f in os.listdir(SOURCE_A_DIR) if f.endswith('.txt')])
    files_b = sorted([f for f in os.listdir(SOURCE_B_DIR) if f.endswith('.txt')])

    if not (len(files_q) == len(files_a) == len(files_b)):
        st.error(f"File count mismatch! Queries: {len(files_q)}, Source A: {len(files_a)}, Source B: {len(files_b)}")
        st.stop()

    return files_q, files_a, files_b

# Initialize DB
init_db()

# --- 6. User Sidebar Controls ---
with st.sidebar:
    # Calculate finished users
    try:
        f_q, _, _ = load_files()
        total_queries = len(f_q)
        finished_count = get_finished_user_count(total_queries)
        st.metric("Users Finished", finished_count)
    except:
        pass
    if st.session_state.user_name:
        st.header("User Controls")

        # Standard User Reset
        # st.write(f"Reset ratings for **{st.session_state.user_name}**.")
        if st.button(f"🚨 Reset Ratings for **{st.session_state.user_name}**"):
            reset_user_ratings(st.session_state.user_name)
            st.session_state.user_name = ""
            st.session_state.current_pair = 0
            st.rerun()

        # Admin Global Reset
        if st.session_state.user_name == "Admin":
            st.markdown("---")
            st.error("🛠️ Admin Panel")
            st.write("Wipe the entire database (all users).")
            if st.button("🔥 DELETE ALL RECORDS"):
                delete_all_records()
                st.session_state.user_name = ""
                st.session_state.current_pair = 0
                st.rerun()
    else:
        st.info("Log in to see user controls.")

# --- 7. Main UI: Login Screen ---
if not st.session_state.user_name:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Query Answer Evaluation")
        name_input = st.text_input("Enter your name:")
        if st.button("Start"):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a valid name.")

# --- 8. Main UI: Evaluation Screen ---
else:
    st.title(f"Evaluator Name: {st.session_state.user_name}")

    files_q, files_a, files_b = load_files()
    total_evals = len(files_a)

    progress_val = st.session_state.current_pair / total_evals if total_evals > 0 else 0
    current_eval = min(st.session_state.current_pair + 1, total_evals)
    st.progress(progress_val)
    st.write(f"**Task {current_eval} of {total_evals}**")

    if st.session_state.current_pair >= total_evals:
        st.success("You have completed all evaluations. Thank you!")
        if st.button("Log Out"):
            st.session_state.user_name = ""
            st.session_state.current_pair = 0
            st.rerun()
    else:
        # --- The NUS Rubric ---
        with st.expander("📚 **How to judge Creativity (The NUS Framework)**", expanded=True):
            st.markdown("""
            **Creativity is defined as a combination of three elements:**
            * 🌟 **New (Novelty & Rarity):** Is the answer a genuinely new invention or a fresh synthesis that avoids tropes or cliches?
            * 🛠️ **Useful (Viability & Alignment):** Is the answer actionable, logically sound, and aligned with the query's constraints?
            * 🤯 **Surprising (Subversion & Trajectory):** Did the answer take a clever, non-obvious, or unpredictable lateral leap?
            """)

        idx = st.session_state.current_pair
        query_name = files_q[idx]
        file_a_name = files_a[idx]
        file_b_name = files_b[idx]

        with open(os.path.join(QUERY_DIR, query_name), 'r', encoding='utf-8') as f:
            text_q = f.read()
        with open(os.path.join(SOURCE_A_DIR, file_a_name), 'r', encoding='utf-8') as f:
            text_a = f.read()
        with open(os.path.join(SOURCE_B_DIR, file_b_name), 'r', encoding='utf-8') as f:
            text_b = f.read()

        st.markdown("### Query")
        st.info(text_q)
        st.markdown("---")

        # Randomize layout
        random.seed(f"{st.session_state.user_name}_{idx}")
        swap = random.choice([True, False])

        col1, col2 = st.columns(2)

        left_text, right_text = (text_b, text_a) if swap else (text_a, text_b)
        left_name, right_name = (file_b_name, file_a_name) if swap else (file_a_name, file_b_name)

        with col1:
            st.subheader("Answer 1")
            st.markdown(left_text)

        with col2:
            st.subheader("Answer 2")
            st.markdown(right_text)

        st.markdown("---")

        # --- Comparative Scoring ---
        st.markdown("### Which answer is more creative?")

        choice = st.radio(
            "Based on the New, Useful, and Surprising (NUS) metrics:",
            ["Answer 1", "Answer 2"],
            index=None,
            horizontal=True,
            key=f"choice_{idx}"
        )

        if st.button("Submit Choice & Continue", type="primary"):
            if choice is None:
                st.error("⚠️ Please select a winner before continuing.")
            else:
                winning_file = left_name if choice == "Answer 1" else right_name
                save_result(query_name, left_name, right_name, st.session_state.user_name, choice, winning_file)
                st.session_state.current_pair += 1
                st.rerun()