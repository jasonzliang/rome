import sqlite3
import pandas as pd
import os
import argparse

def export_to_csv(eval_dir, include_all=False):
    # Construct paths based on the provided directory
    db_file = os.path.join(eval_dir, "results_ab.db")
    export_csv_name = os.path.join(eval_dir, "evaluation_winners.csv")

    # 1. Check if the database exists
    if not os.path.exists(db_file):
        print(f"❌ Error: Could not find '{db_file}'.")
        print(f"   Checked directory: {os.path.abspath(eval_dir)}")
        return

    # 2. Determine which query to use
    if include_all:
        print("📝 Mode: Including ALL evaluation records.")
        query = "SELECT query_file, user_name, winning_file FROM comparative_evals"
    else:
        print("📝 Mode: Filtering for MOST RECENT evaluation per user/query.")
        query = """
            SELECT
                query_file,
                user_name,
                winning_file
            FROM comparative_evals
            WHERE id IN (
                SELECT MAX(id)
                FROM comparative_evals
                GROUP BY user_name, query_file
            )
        """

    # 3. Connect to the database and load into a Pandas DataFrame
    print(f"📥 Connecting to database at {db_file}...")
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("⚠️ The database is empty. No evaluations to export.")
        return

    # 4. Export to CSV
    df.to_csv(export_csv_name, index=False)
    print(f"✅ Success! Exported {len(df)} records to {export_csv_name}")

    # 5. Show summaries
    print("\n📊 Win Counts by Specific File:")
    print(df['winning_file'].value_counts().to_string())

    df['model_name'] = df['winning_file'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    print("\n🏆 Overall Win Counts by Model:")
    print(df['model_name'].value_counts().to_string())
    print()

if __name__ == "__main__":
    # Get the default directory from environment variable or current dir
    default_dir = os.environ.get('CAESAR_HUMAN_EVAL_DIR', '.')

    parser = argparse.ArgumentParser(description="Export evaluation results from SQLite to CSV.")

    # Path Override
    parser.add_argument(
        "--eval-dir", "-d",
        type=str,
        default=default_dir,
        help=f"Directory containing the DB and where to save CSV (default: {default_dir})"
    )

    # Inclusion Toggle
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all historical evaluations instead of just the most recent one."
    )

    args = parser.parse_args()

    export_to_csv(eval_dir=args.eval_dir, include_all=args.include_all)