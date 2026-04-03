"""Transfer configuration functions for prepare_artifact.py"""
import glob
import json
import os

import networkx as nx

from prepare_artifact import find_latest_synthesis

# Outdated, do not use
def setup_transfer_dict_11_17():
    full = {
        'caesar_sources': '11_17_*/*12130155/*merged-3*',
        'other_sources': '12_4_answers/*/',
    }
    eli5 = {
        'caesar_sources': '11_17_*/*12130155/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5/*/',
    }
    eli5_600 = {
        'caesar_sources': '11_17_*/*12130327/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5_600t/*/',
    }
    # Returns (transfer_list, config_overrides)
    return [full, eli5, eli5_600], {}


def setup_transfer_dict_12_13():
    full = {
        'caesar_sources': '12_13_*/*12160*/*merged-3*',
        'other_sources': '12_4_answers/*/',
    }
    eli5 = {
        'caesar_sources': '12_13_*/*12160*/*merged-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
    }
    eli5_450 = {
        'caesar_sources': '12_13_*/*12160*/*merged-eli5-3.450w.1*',
        'other_sources': '12_4_answers_eli5_450w/*/',
    }

    overrides = {"CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [full, eli5, eli5_450], overrides


# Outdated, do not use
def setup_transfer_dict_12_13_v2():
    full = {
        'caesar_sources': '12_13_*/*12161*/*merged-3*',
        'other_sources': '12_4_answers/*/',
    }
    eli5 = {
        'caesar_sources': '12_13_*/*12161*/*merged-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
    }
    eli5_450 = {
        'caesar_sources': '12_13_*/*12161*/*merged-eli5-3.450w.1*',
        'other_sources': '12_4_answers_eli5_450w/*/',
    }

    overrides = {"CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [full, eli5, eli5_450], overrides


def setup_transfer_dict_12_13_syn_ablation():
    syn1 = {
        'caesar_sources': '12_13_*/*12160*/*synthesis-1*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3 = {
        'caesar_sources': '12_13_*/*12160*/*synthesis-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged = {
        'caesar_sources': '12_13_*/*12160*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }

    syn1_eli5 = {
        'caesar_sources': '12_13_*/*12160*/*synth-eli5-1.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3_eli5 = {
        'caesar_sources': '12_13_*/*12160*/*synth-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged_eli5 = {
        'caesar_sources': '12_13_*/*12160*/*merged-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }

    syn1_eli5_450w = {
        'caesar_sources': '12_13_*/*12160*/*synth-eli5-1.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3_eli5_450w = {
        'caesar_sources': '12_13_*/*12160*/*synth-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged_eli5_450w = {
        'caesar_sources': '12_13_*/*12160*/*merged-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }
    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [syn1, syn3, merged, syn1_eli5, syn3_eli5, merged_eli5, syn1_eli5_450w, syn3_eli5_450w, merged_eli5_450w], overrides


def setup_transfer_dict_12_13_iter_ablation():
    iter250 = {
        'caesar_sources': '12_13_*/*01072*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_250.txt'
    }
    iter500 = {
        'caesar_sources': '12_13_*/*01071*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_500.txt'
    }
    iter1000 = {
        'caesar_sources': '12_13_*/*01110646*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_1000.txt'
    }

    iter250_eli5 = {
        'caesar_sources': '12_13_*/*01072*/*merged-eli5-3.0*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_250.txt'
    }
    iter500_eli5 = {
        'caesar_sources': '12_13_*/*01071*/*merged-eli5-3.0*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_500.txt'
    }
    iter1000_eli5 = {
        'caesar_sources': '12_13_*/*01110646*/*merged-eli5-3.0*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_1000.txt'
    }

    iter250_eli5_450w = {
        'caesar_sources': '12_13_*/*01072*/*merged-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_250.txt'
    }
    iter500_eli5_450w = {
        'caesar_sources': '12_13_*/*01071*/*merged-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_500.txt'
    }
    iter1000_eli5_450w = {
        'caesar_sources': '12_13_*/*01110646*/*merged-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_1000.txt'
    }
    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [iter250, iter500, iter1000, iter250_eli5, iter500_eli5, iter1000_eli5, iter250_eli5_450w, iter500_eli5_450w, iter1000_eli5_450w], overrides


def setup_transfer_dict_3_28_graph_ablation():
    categories = ['constrained_creativity', 'counterfactual_reasoning',
        'crossdomain_synthesis', 'meta_creativity', 'openended_creativity']
    variants = [
        ('3_28_{cat}',    'answer_cat_3_28.txt'),
        # ('exp_03_2026/3_29_{cat}',    'answer_cat_3_29.txt'),
        ('4_2_{cat}_qe',  'answer_cat_4_2_qe.txt'),
        ('3_30_{cat}_qe', 'answer_cat_3_30_qe.txt'),
        # ('3_28_{cat}_v2', 'answer_cat_3_28_v2.txt'),
        # ('3_29_{cat}_v2', 'answer_cat_3_29_v2.txt'),
    ]
    file_patterns = [
        ('12_4_answers/*/',           '*merged-3*'),
        ('12_4_answers_eli5/*/',      '*merged-eli5-3.[01]*'),
        ('12_4_answers_eli5_450w/*/', '*merged-eli5-3.450w*'),
    ]

    # Baselines use hardcoded synthesis versions
    baseline_patterns = [
        # ('answer_cat_12_13.txt', [
        #     ('12_4_answers/*/',           '12_13_*/*12160*/*merged-3*'),
        #     ('12_4_answers_eli5/*/',      '12_13_*/*12160*/*merged-eli5-3.1*'),
        #     ('12_4_answers_eli5_450w/*/', '12_13_*/*12160*/*merged-eli5-3.450w.1*'),
        # ]),
        # ('answer_cat_250.txt', [
        #     ('12_4_answers/*/',           '12_13_*/*01072*/*merged-3*'),
        #     ('12_4_answers_eli5/*/',      '12_13_*/*01072*/*merged-eli5-3.0*'),
        #     ('12_4_answers_eli5_450w/*/', '12_13_*/*01072*/*merged-eli5-3.450w*'),
        # ]),
        # ('answer_cat_1000.txt', [
        #     ('12_4_answers/*/',           '12_13_*/*01110646*/*merged-3*'),
        #     ('12_4_answers_eli5/*/',      '12_13_*/*01110646*/*merged-eli5-3.0*'),
        #     ('12_4_answers_eli5_450w/*/', '12_13_*/*01110646*/*merged-eli5-3.450w*'),
        # ]),
    ]

    transfer_list = []
    for filename, patterns in baseline_patterns:
        for other_sources, source_pattern in patterns:
            transfer_list.append({
                'caesar_sources': source_pattern,
                'other_sources': other_sources,
                'caesar_filename': filename})

    # Other variants use latest synthesis
    for exp_template, filename in variants:
        for cat in categories:
            exp = exp_template.format(cat=cat)
            for other_sources, merged_pattern in file_patterns:
                transfer_list.append({
                    'caesar_sources': find_latest_synthesis(exp, merged_pattern),
                    'other_sources': other_sources,
                    'caesar_filename': filename})

    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return transfer_list, overrides


def setup_transfer_dict_12_13_insights(base_dir=None, output_dir=None):
    """Parse 12_13 checkpoint graphs, find top/bottom pages by neighbor count, write to files.

    Top 10: pages with most neighbors (visit_count > 1) — high-connectivity hubs.
    Bottom 10: pages with fewest neighbors (visit_count = 1) — leaf/dead-end pages.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')
    if output_dir is None:
        output_dir = os.path.join(base_dir, '12_13_insights')
    os.makedirs(output_dir, exist_ok=True)

    experiment_dirs = sorted(glob.glob(os.path.join(base_dir, '12_13_*')))
    transfer_list = []

    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        checkpoint_path = os.path.join(exp_dir, '__rome__', 'agent_CaesarExplorer.checkpoint.json')
        if not os.path.exists(checkpoint_path):
            continue

        with open(checkpoint_path) as f:
            data = json.load(f)

        graph = nx.node_link_graph(data['graph'], edges='edges')

        # Build (url, neighbor_count, visit_count, insights) tuples
        node_stats = []
        for node in graph.nodes:
            visit_count = graph.nodes[node].get('visit_count', 1)
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node)) - {node}
            neighbor_count = len(neighbors)
            insights = graph.nodes[node].get('insights', '')
            node_stats.append((node, neighbor_count, visit_count, insights))

        # Top 10: most neighbors, visit_count > 1
        hub_pages = sorted(
            [s for s in node_stats if s[2] > 1 and s[3]],
            key=lambda x: x[1], reverse=True)[:10]

        # Bottom 10: fewest neighbors, visit_count = 1
        leaf_pages = sorted(
            [s for s in node_stats if s[2] == 1 and s[3]],
            key=lambda x: x[1])[:10]

        # Write files
        for label, pages, suffix in [('top10', hub_pages, 'hubs'), ('bottom10', leaf_pages, 'leaves')]:
            filename = f'{exp_name}_{suffix}.txt'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                # f.write(f"# {label.upper()} pages for {exp_name}\n")
                # f.write(f"# Sorted by neighbor count ({'desc' if label == 'top10' else 'asc'})\n\n")
                for url, n_neighbors, visit_count, insights in pages:
                    # f.write(f"URL: {url}\n")
                    # f.write(f"Neighbors: {n_neighbors} | Visits: {visit_count}\n")
                    f.write(f"Insights: {insights if insights else '(none)'}\n")
                    f.write("-" * 80 + "\n\n")

            transfer_list.append({
                'caesar_sources': f'12_13_insights/{filename}',
                'other_sources': '12_4_answers/*/',
                'caesar_filename': f'answer_cat_{suffix}.txt',
            })

    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return transfer_list, overrides


TRANSFER_CONFIGS = {
    '11_17': setup_transfer_dict_11_17,
    '12_13': setup_transfer_dict_12_13,
    '12_13_v2': setup_transfer_dict_12_13_v2,
    '12_13_syn': setup_transfer_dict_12_13_syn_ablation,
    '12_13_iter': setup_transfer_dict_12_13_iter_ablation,
    '3_28_graph': setup_transfer_dict_3_28_graph_ablation,
    '12_13_insights': setup_transfer_dict_12_13_insights,
}
