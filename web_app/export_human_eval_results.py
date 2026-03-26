import sqlite3
import pandas as pd
import os

# Grab the environment variable (or default to current directory if not set)
EVAL_DIR = os.environ.get('CAESAR_HUMAN_EVAL_DIR', '.')
DB_FILE = os.path.join(EVAL_DIR, "results_ab.db")
EXPORT_CSV_NAME = os.path.join(EVAL_DIR, "evaluation_winners.csv")

def export_to_csv():
    # 1. Check if the database exists
    if not os.path.exists(DB_FILE):
        print(f"❌ Error: Could not find '{DB_FILE}'. Have any evaluations been submitted yet?")
        return

    # 2. Connect to the database and load into a Pandas DataFrame
    print(f"📥 Connecting to database at {DB_FILE}...")
    with sqlite3.connect(DB_FILE) as conn:
        # Extract exactly the columns requested
        query = """
            SELECT
                query_file,
                user_name,
                winning_file
            FROM comparative_evals
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("⚠️ The database is empty. No evaluations to export.")
        return

    # 3. Export to CSV
    df.to_csv(EXPORT_CSV_NAME, index=False)
    print(f"✅ Success! Exported {len(df)} evaluation records to {EXPORT_CSV_NAME}")

    # 4. Show a quick summary in the terminal
    print("\n📊 Current Win Counts by Specific File:")
    print(df['winning_file'].value_counts().to_string())

    # 5. Extract the base model name (caesar vs gemini) for a high-level summary
    df['model_name'] = df['winning_file'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    print("\n🏆 Overall Win Counts by Model:")
    print(df['model_name'].value_counts().to_string())
    print()

if __name__ == "__main__":
    export_to_csv()