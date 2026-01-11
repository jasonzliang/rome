"""Prepare Caesar artifacts for LLM as Judge"""
import glob
import os
import shutil
import sys
import time

# Default Configuration Dictionary
DEFAULT_CONFIG = {
    # Base directory filled with answers from Caesar agent
    "CAESAR_AGENT_BASE_DIR": os.path.abspath("result"),
    # Filename for Caesar agent answer in answer directory
    "CAESAR_AGENT_FILENAME": "answer_cat_cam.txt",
    # Base directory filled with answers from other agents
    "OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/other_agent_answers"),
    # Clean up the output directory before answer file is copied
    "CLEAR_OUTPUT_DIR": True,

    # Base directory to output all agent answers
    "ALL_AGENT_BASE_DIR": os.path.abspath("query_result/all_agent_answers"),
    # Empty files to create in the category level output directory
    "CATEGORY_NEW_FILES": ['judge_claude.txt', 'judge_gemini.txt', 'judge_gpt.txt'],
    # Empty files to create in the meta-category level output directory
    "META_CATEGORY_NEW_FILES": ['judge_summary.txt', 'judge_csv.txt']
}

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


def setup_transfer_dict_11_17_v2():
    full = {
        'caesar_sources': '11_17_*/*1217*/*merged-3*',
        'other_sources': '12_4_answers/*/',
    }
    eli5 = {
        'caesar_sources': '11_17_*/*1217*/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5/*/',
    }
    eli5_600 = {
        'caesar_sources': '11_17_*/*1217*/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5_600t/*/',
    }
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
    return [full, eli5, eli5_450], {}


def setup_transfer_dict_12_13_syn_ablation():
    syn1 = {
        'caesar_sources': '12_13_*/*12161*/*synthesis-1*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3 = {
        'caesar_sources': '12_13_*/*12161*/*synthesis-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged = {
        'caesar_sources': '12_13_*/*12161*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }

    syn1_eli5 = {
        'caesar_sources': '12_13_*/*12161*/*synth-eli5-1.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3_eli5 = {
        'caesar_sources': '12_13_*/*12161*/*synth-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged_eli5 = {
        'caesar_sources': '12_13_*/*12161*/*merged-eli5-3.1*',
        'other_sources': '12_4_answers_eli5/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }

    syn1_eli5_450w = {
        'caesar_sources': '12_13_*/*12161*/*synth-eli5-1.450w.1*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_syn1.txt'
    }
    syn3_eli5_450w = {
        'caesar_sources': '12_13_*/*12161*/*synth-eli5-3.450w.1*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_syn3.txt'
    }
    merged_eli5_450w = {
        'caesar_sources': '12_13_*/*12161*/*merged-eli5-3.450w*',
        'other_sources': '12_4_answers_eli5_450w/*/',
        'caesar_filename': 'answer_cat_merge.txt'
    }
    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [syn1, syn3, merged, syn1_eli5, syn3_eli5, merged_eli5, syn1_eli5_450w, syn3_eli5_450w, merged_eli5_450w], overrides


def setup_transfer_dict_12_13_iter_ablation():
    iter250 = {
        'caesar_sources': '12_13_*/*12161*/*synthesis-1*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_t250.txt'
    }
    iter500 = {
        'caesar_sources': '12_13_*/*12161*/*synthesis-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_t500.txt'
    }
    iter1000 = {
        'caesar_sources': '12_13_*/*12161*/*merged-3*',
        'other_sources': '12_4_answers/*/',
        'caesar_filename': 'answer_cat_t1000.txt'
    }
    overrides = {"OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, "CATEGORY_NEW_FILES": [], "META_CATEGORY_NEW_FILES": []}
    return [iter250, iter500, iter1000], overrides


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
    return [full, eli5, eli5_450], {}


def prepare_artifact(transfer_func):
    """Copy Caesar artifacts to query answer directory for LLM judging"""

    # Unpack transfer list and any configuration overrides
    transfer_dicts, config_overrides = transfer_func()

    # Create the active config by merging defaults with overrides
    config = DEFAULT_CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)

    print(f"Active Output Dir: {config['ALL_AGENT_BASE_DIR']}")

    for td in transfer_dicts:
        # Use config dictionary instead of globals
        other_source_dirs = glob.glob(os.path.join(config['OTHER_AGENT_BASE_DIR'],
            td['other_sources']))
        for caesar_file in glob.glob(os.path.join(config['CAESAR_AGENT_BASE_DIR'],
            td['caesar_sources'])):
            matched = None
            for other_source_dir in other_source_dirs:
                category_name = os.path.normpath(other_source_dir).split(os.sep)[-1]
                if category_name in caesar_file:
                    matched = other_source_dir
                    break

            print(f"Caesar answer: {caesar_file}")
            if not matched:
                print(f"Cannot find matching other source dir to caesar answer")
                continue
            else:
                print(f"Matching other agent answer directory: {matched}")
                other_source_dir = matched

            os.makedirs(config['ALL_AGENT_BASE_DIR'], exist_ok=True)
            output_dir = os.path.join(config['ALL_AGENT_BASE_DIR'],
                os.path.join(*os.path.normpath(other_source_dir).split(os.sep)[-2:]))
            meta_output_dir = os.path.dirname(os.path.normpath(output_dir))

            if config['CLEAR_OUTPUT_DIR'] and os.path.exists(output_dir):
                shutil.rmtree(output_dir) # Replaced os.system rm -rf

            shutil.copytree(other_source_dir, output_dir, dirs_exist_ok=True)

            for new_file in config['CATEGORY_NEW_FILES']:
                # Replaced os.system touch
                with open(os.path.join(output_dir, new_file), 'a'):
                    os.utime(os.path.join(output_dir, new_file), None)

            for new_file in config['META_CATEGORY_NEW_FILES']:
                # Replaced os.system touch
                with open(os.path.join(meta_output_dir, new_file), 'a'):
                    os.utime(os.path.join(meta_output_dir, new_file), None)

            # Use local filename override from td, otherwise fallback to config default
            caesar_filename = td.get("caesar_filename") or config['CAESAR_AGENT_FILENAME']

            # Using shutil.copy2 to preserve metadata instead of os.system cp
            src = caesar_file
            dst = os.path.join(output_dir, caesar_filename)
            print(f'cp {src} {dst}\n')
            shutil.copy2(src, dst)


if __name__ == '__main__':
    prepare_artifact(setup_transfer_dict_12_13_syn_ablation)
