#!/usr/bin/env python3
"""
Script to extract scoring results from judge JSON files and generate a CSV report.
Automatically discovers query categories from directory structure.
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def load_agent_mapping(mapping_file):
    """Load agent name mapping from file (code_name: real_name format)."""
    if not mapping_file or not Path(mapping_file).exists():
        print(f"⚠️  Agent mapping file not found: {mapping_file}")
        return {}

    mapping = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    code_name, real_name = line.split(':', 1)
                    code_name, real_name = code_name.strip(), real_name.strip()
                    mapping[code_name] = real_name
        print(f"✅ Loaded {len(mapping)} agent name mappings")
    except Exception as e:
        print(f"⚠️  Error loading agent mapping: {e}")

    return mapping

def discover_categories(base_dir):
    """Recursively discover categories from directories containing judge files."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_path}")
        return {}

    categories = defaultdict(list)
    for judge_file in base_path.rglob("judge_*.json"):
        category_name = judge_file.parent.name
        if judge_file.parent not in categories[category_name]:
            categories[category_name].append(judge_file.parent)

    return dict(categories)

def extract_judge_name(filename):
    """Remove numeric suffixes from judge filenames (e.g., judge_claude_1.json -> judge_claude)."""
    stem = filename.stem if hasattr(filename, 'stem') else Path(filename).stem
    parts = stem.rsplit('_', 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem

def process_judge_file(judge_file, category_name, csv_data, stats):
    """Process a single judge JSON file and extract reviews."""
    judge_name = extract_judge_name(judge_file)

    try:
        with open(judge_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f).get('reviews', [])

        if not reviews:
            print(f"  ⚠️  No reviews in {judge_file.name}")
            return 0

        for review in reviews:
            agent_name = review.get('agent_name', '')
            if not agent_name:
                print(f"  ⚠️  Missing agent_name in {judge_file.name}")
                continue
            agent_name = Path(agent_name).stem

            scores = review.get('scores', {})
            row = {
                'Query Category': category_name,
                'Agent Name': agent_name,
                'Judge Name': judge_name,
                'New Score': scores.get('New', 0),
                'Useful Score': scores.get('Useful', 0),
                'Surprising Score': scores.get('Surprising', 0),
            }
            row['Total Score'] = row['New Score'] + row['Useful Score'] + row['Surprising Score']

            csv_data.append(row)
            stats[category_name]['total'] += 1
            stats[category_name][judge_name] += 1

        return 1

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error in {judge_file.name}: {e}")
    except Exception as e:
        print(f"  ❌ Error processing {judge_file.name}: {e}")

    return 0

def process_directory(base_dir, categories):
    """Process all judge JSON files and extract scores."""
    csv_data = []
    stats = defaultdict(lambda: defaultdict(int))
    total_judge_files = 0

    for category_name in sorted(categories.keys()):
        category_paths = categories[category_name]
        print(f"Processing {category_name}: found in {len(category_paths)} location(s)")

        judge_files = [f for path in category_paths for f in sorted(path.glob("judge_*.json"))]
        total_judge_files += len(judge_files)

        for judge_file in judge_files:
            process_judge_file(judge_file, category_name, csv_data, stats)

        print(f"  → Processed {len(judge_files)} judge files")

    return csv_data, stats, total_judge_files

def write_csv(csv_data, output_file):
    """Write data to CSV file."""
    fieldnames = ['Query Category', 'Agent Name', 'Judge Name',
                  'New Score', 'Useful Score', 'Surprising Score', 'Total Score']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

def calculate_stats(csv_data, total_judge_files):
    """Calculate statistics for verification."""
    if not csv_data or total_judge_files == 0:
        return None

    return {
        'total_rows': len(csv_data),
        'num_agents': len(set(row['Agent Name'] for row in csv_data)),
        'num_judges': len(set(row['Judge Name'] for row in csv_data)),
        'agents': sorted(set(row['Agent Name'] for row in csv_data)),
        'judges': sorted(set(row['Judge Name'] for row in csv_data)),
    }

def print_category_breakdown(stats, categories):
    """Print per-category breakdown with judge details."""
    print(f"\n📁 Per-category breakdown:")
    for category_name in sorted(categories.keys()):
        print(f"  {category_name}: {stats[category_name]['total']} rows")
        judge_counts = {k: v for k, v in stats[category_name].items() if k != 'total'}
        for judge, count in sorted(judge_counts.items()):
            print(f"    - {judge}: {count} reviews")

def print_verification(csv_data, stats, categories, total_judge_files):
    """Print verification statistics."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    calc_stats = calculate_stats(csv_data, total_judge_files)
    if not calc_stats:
        print("\n❌ No data to verify!")
        return

    print(f"\n📊 Total rows generated: {calc_stats['total_rows']}")
    print_category_breakdown(stats, categories)

    print(f"\n🤖 Unique agents:")
    for agent in calc_stats['agents']:
        count = sum(1 for row in csv_data if row['Agent Name'] == agent)
        print(f"  {agent}: {count} total reviews")

    print(f"\n⚖️  Unique judges:")
    for judge in calc_stats['judges']:
        count = sum(1 for row in csv_data if row['Judge Name'] == judge)
        print(f"  {judge}: {count} total reviews")

    num_categories = len(categories)
    avg_files_per_cat = total_judge_files / num_categories
    avg_runs_per_judge = total_judge_files / (num_categories * calc_stats['num_judges'])
    expected_total = num_categories * calc_stats['num_agents'] * avg_files_per_cat

    print(f"\n✅ Discovered: {num_categories} categories, {calc_stats['num_agents']} agents, {calc_stats['num_judges']} judge types")
    print(f"✅ Total judge files processed: {total_judge_files}")
    print(f"✅ Average judge runs per category: {avg_files_per_cat:.1f}")
    print(f"✅ Average runs per unique judge: {avg_runs_per_judge:.1f}")
    print(f"✅ Expected rows: ~{int(expected_total)} ({num_categories} categories × {calc_stats['num_agents']} agents × {avg_files_per_cat:.1f} judge files per category)")
    print(f"✅ Actual rows: {calc_stats['total_rows']}")

    if calc_stats['total_rows'] >= expected_total * 0.95:
        print("✅ All scores appear to be accounted for!")
    else:
        print(f"⚠️  Potential missing data: ~{int(expected_total - calc_stats['total_rows'])} rows difference")

def analyze_csv(csv_file, agent_mapping, analysis_file):
    """Perform comprehensive analysis on the CSV with all required tables and judge bias detection."""
    print(f"\n📊 Starting analysis of {csv_file}...")

    # Read CSV
    csv_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['New Score'] = int(row['New Score'])
            row['Useful Score'] = int(row['Useful Score'])
            row['Surprising Score'] = int(row['Surprising Score'])
            row['Total Score'] = int(row['Total Score'])
            # Apply agent name mapping
            row['Display Name'] = agent_mapping.get(row['Agent Name'], row['Agent Name'])
            csv_data.append(row)

    if not csv_data:
        print("❌ No data found in CSV")
        return

    output_lines = []

    def add_section(title, content):
        sep = "=" * 120
        output_lines.extend([f"\n{sep}", title, sep, content])
        print(f"\n{sep}\n{title}\n{sep}\n{content}")

    def format_table(headers, rows, sort_by_col=None, reverse=True):
        """Format data as aligned table with optional sorting."""
        if sort_by_col is not None and rows:
            def get_sort_key(row):
                val = row[sort_by_col]
                try:
                    # Try to convert string numbers (e.g., "4.50") to float
                    return float(val)
                except (ValueError, TypeError):
                    # Handle "N/A" or text by treating it as 0
                    return 0

            rows = sorted(rows, key=get_sort_key, reverse=reverse)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format rows
        lines = []
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        for row in rows:
            lines.append(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))

        return "\n".join(lines)

    # Extract unique values
    categories = sorted(set(row['Query Category'] for row in csv_data))
    agents = sorted(set(row['Agent Name'] for row in csv_data))
    judges = sorted(set(row['Judge Name'] for row in csv_data))

    # TABLE 1: Raw Metrics for All Agents on Each Query (Compact)
    query_tables = []
    for category in categories:
        cat_data = [row for row in csv_data if row['Query Category'] == category]
        agent_scores = defaultdict(lambda: {'new': [], 'useful': [], 'surprising': [], 'total': []})

        for row in cat_data:
            agent_scores[row['Display Name']]['new'].append(row['New Score'])
            agent_scores[row['Display Name']]['useful'].append(row['Useful Score'])
            agent_scores[row['Display Name']]['surprising'].append(row['Surprising Score'])
            agent_scores[row['Display Name']]['total'].append(row['Total Score'])

        headers = ["Agent", "New", "Useful", "Surprising", "Total", "Avg Total"]
        rows = []
        for agent in sorted(agent_scores.keys()):
            scores = agent_scores[agent]
            avg_total = sum(scores['total']) / len(scores['total'])
            rows.append([
                agent,
                f"{sum(scores['new'])/len(scores['new']):.2f}",
                f"{sum(scores['useful'])/len(scores['useful']):.2f}",
                f"{sum(scores['surprising'])/len(scores['surprising']):.2f}",
                f"{sum(scores['total'])/len(scores['total']):.2f}",
                f"{avg_total:.2f}"
            ])

        query_tables.append(f"\nQUERY: {category}\n{format_table(headers, rows, sort_by_col=5, reverse=True)}")

    add_section("TABLE 1: RAW METRICS FOR ALL AGENTS ON EACH QUERY", "\n".join(query_tables))

    # TABLE 2: Average Score for Each Metric by Agent
    agent_metric_stats = defaultdict(lambda: {'new': [], 'useful': [], 'surprising': [], 'total': []})
    for row in csv_data:
        agent_metric_stats[row['Display Name']]['new'].append(row['New Score'])
        agent_metric_stats[row['Display Name']]['useful'].append(row['Useful Score'])
        agent_metric_stats[row['Display Name']]['surprising'].append(row['Surprising Score'])
        agent_metric_stats[row['Display Name']]['total'].append(row['Total Score'])

    headers = ["Agent", "Avg New", "Avg Useful", "Avg Surprising", "Avg Total", "Reviews"]
    rows = []
    for agent in sorted(agent_metric_stats.keys()):
        scores = agent_metric_stats[agent]
        rows.append([
            agent,
            f"{sum(scores['new'])/len(scores['new']):.2f}",
            f"{sum(scores['useful'])/len(scores['useful']):.2f}",
            f"{sum(scores['surprising'])/len(scores['surprising']):.2f}",
            f"{sum(scores['total'])/len(scores['total']):.2f}",
            sum(scores['total'])/len(scores['total']),  # For sorting
        ])

    # Replace last column with review count after sorting
    rows_sorted = sorted(rows, key=lambda x: x[5], reverse=True)
    for i, row in enumerate(rows_sorted):
        agent_name = row[0]
        row[5] = len(agent_metric_stats[agent_name]['total'])

    add_section("TABLE 2: AVERAGE SCORE FOR EACH METRIC BY AGENT (Sorted by Avg Total)",
                format_table(headers, rows_sorted))

    # TABLE 3: Average Score for Each Agent from Each Judge
    agent_judge_stats = defaultdict(lambda: defaultdict(lambda: {'scores': [], 'total': 0}))
    for row in csv_data:
        agent_judge_stats[row['Display Name']][row['Judge Name']]['scores'].append(row['Total Score'])

    headers = ["Agent"] + judges + ["Overall Avg"]
    rows = []
    for agent in sorted(agent_judge_stats.keys()):
        row = [agent]
        judge_avgs = []
        for judge in judges:
            if agent_judge_stats[agent][judge]['scores']:
                avg = sum(agent_judge_stats[agent][judge]['scores']) / len(agent_judge_stats[agent][judge]['scores'])
                row.append(f"{avg:.2f}")
                judge_avgs.append(avg)
            else:
                row.append("N/A")

        overall_avg = sum(judge_avgs) / len(judge_avgs) if judge_avgs else 0
        row.append(f"{overall_avg:.2f}")
        rows.append(row)

    add_section("TABLE 3: AVERAGE SCORE FOR EACH AGENT FROM EACH JUDGE (Sorted by Overall Avg)",
                format_table(headers, rows, sort_by_col=len(headers)-1, reverse=True))

    # TABLE 4: Judge Bias Analysis
    def extract_judge_base_name(judge_name):
        """Extract base name from judge (e.g., 'judge_claude' -> 'claude')."""
        return judge_name.replace('judge_', '').lower()

    def agent_similar_to_judge(agent_name, judge_base):
        """
        Check if agent name matches judge using abbreviations found in the mapping file.
        """
        agent = agent_name.lower()

        # 1. Define mappings based on your agent_names.txt suffixes
        mappings = {
            'claude': ['claude', 'cld', 'cls', 'sonnet'], # Matches answer_cow_cld, answer_dog_cls
            'gemini': ['gemini', 'ged', 'ges'],           # Matches answer_fox_ged, answer_lion_ges
            'gpt':    ['gpt', 'gps', 'gpd']               # Matches answer_pig_gps, answer_owl_gpd
        }

        # 2. Get the list of search terms for this judge (default to just the judge name)
        search_terms = mappings.get(judge_base, [judge_base])

        # 3. Check if ANY of the terms exist in the agent ID
        return any(term in agent for term in search_terms)

    bias_analysis = []

    for judge in judges:
        judge_base = extract_judge_base_name(judge)

        # Find agents with similar names to this judge
        similar_agents = [row['Agent Name'] for row in csv_data if agent_similar_to_judge(row['Agent Name'], judge_base)]
        similar_agents = list(set(similar_agents))  # Unique

        if not similar_agents:
            bias_analysis.append(f"\n{judge.upper()}:")
            bias_analysis.append(f"  No agents with similar names found (looking for '{judge_base}' in agent names)")
            continue

        # Calculate average score THIS judge gave to similar-named agents
        same_judge_scores = [row['Total Score'] for row in csv_data
                            if row['Judge Name'] == judge and row['Agent Name'] in similar_agents]

        # Calculate average score OTHER judges gave to the same similar-named agents
        other_judges_scores = [row['Total Score'] for row in csv_data
                              if row['Judge Name'] != judge and row['Agent Name'] in similar_agents]

        if same_judge_scores and other_judges_scores:
            same_avg = sum(same_judge_scores) / len(same_judge_scores)
            other_avg = sum(other_judges_scores) / len(other_judges_scores)
            bias = same_avg - other_avg

            bias_analysis.append(f"\n{judge.upper()}:")
            bias_analysis.append(f"  Similar-named agents: {', '.join(similar_agents)}")
            bias_analysis.append(f"  Avg score from {judge}: {same_avg:.2f} (n={len(same_judge_scores)})")
            bias_analysis.append(f"  Avg score from other judges: {other_avg:.2f} (n={len(other_judges_scores)})")
            bias_analysis.append(f"  Bias score: {bias:+.2f} ({'POSITIVE' if bias > 0 else 'NEGATIVE' if bias < 0 else 'NEUTRAL'})")

            if abs(bias) > 1.0:
                bias_analysis.append(f"  ⚠️  SIGNIFICANT BIAS DETECTED (>{1.0:.1f} points)")
        else:
            bias_analysis.append(f"\n{judge.upper()}:")
            bias_analysis.append(f"  Insufficient data for bias calculation")

    add_section("TABLE 4: JUDGE BIAS ANALYSIS", "\n".join(bias_analysis))

    # TABLE 5: Overall Statistics Summary
    overall_stats = f"""
Total Rows: {len(csv_data)}
Categories: {len(categories)}
Agents: {len(agents)}
Judges: {len(judges)}

OVERALL AVERAGES:
  Average Total Score:      {sum(row['Total Score'] for row in csv_data) / len(csv_data):.2f}
  Average New Score:        {sum(row['New Score'] for row in csv_data) / len(csv_data):.2f}
  Average Useful Score:     {sum(row['Useful Score'] for row in csv_data) / len(csv_data):.2f}
  Average Surprising Score: {sum(row['Surprising Score'] for row in csv_data) / len(csv_data):.2f}
"""
    add_section("TABLE 5: OVERALL STATISTICS", overall_stats)

    # Write to file
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n✅ Analysis complete: {Path(analysis_file).resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract scoring results from judge JSON files and generate CSV report.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       # Use current directory, output to ./judge_csv.txt
  %(prog)s -i /path/to/data                      # Output to /path/to/data/judge_csv.txt
  %(prog)s -i data -o csv.txt                    # Custom output file location
  %(prog)s -i data -a analysis.txt               # Generate CSV and analyze it
  %(prog)s -i data -a analysis.txt -m names.txt  # Analyze with custom agent mapping
        """
    )

    parser.add_argument('-i', '--input-dir', default='.',
        help='Base directory containing category folders (default: current directory)')
    parser.add_argument('-o', '--output', default=None,
        help='Output CSV filename (default: judge_csv.txt in input directory)')
    parser.add_argument('-a', '--analyze', default='judge_analysis.txt', metavar='FILE',
        help='Output analysis filename (default: judge_analysis.txt in input directory)')
    parser.add_argument('-m', '--agent-mapping',
        default='config/llm_as_judge/prompts/agent_names.txt',
        help='Agent name mapping file (default: config/llm_as_judge/prompts/agent_names.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Show detailed progress information')

    args = parser.parse_args()

    output_file = args.output if args.output else str(Path(args.input_dir) / "judge_csv.txt")

    print("🚀 Starting score extraction...")
    print(f"📂 Base directory: {Path(args.input_dir).resolve()}")
    print(f"📝 Output file: {Path(output_file).resolve()}\n")

    # Auto-discover categories
    print("🔍 Discovering categories...")
    categories = discover_categories(args.input_dir)

    if not categories:
        print("❌ No categories found! Make sure the directory contains subdirectories with judge_*.json files.")
        return

    print(f"✅ Found {len(categories)} categories:")
    for cat_name in sorted(categories.keys()):
        num_locs = len(categories[cat_name])
        loc_str = f" (in {num_locs} location{'s' if num_locs > 1 else ''})"
        print(f"   • {cat_name}{loc_str}")
    print()

    # Process all files
    csv_data, stats, total_judge_files = process_directory(args.input_dir, categories)

    if not csv_data:
        print("\n❌ No data extracted! Check if directories and files exist.")
        return

    # Write CSV
    write_csv(csv_data, output_file)
    print(f"\n✅ CSV generated successfully: {Path(output_file).resolve()}")

    # Print verification
    print_verification(csv_data, stats, categories, total_judge_files)

    # Perform analysis if requested
    if args.analyze:
        agent_mapping = load_agent_mapping(args.agent_mapping)
        analysis_file = Path(args.input_dir) / Path(args.analyze)
        analyze_csv(output_file, agent_mapping, analysis_file)

if __name__ == "__main__":
    main()