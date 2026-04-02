"""Prepare Caesar artifacts for LLM as Judge"""
import glob
import os
import re
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    # Strip ABSTRACT section from start and SOURCES section from end of artifact
    "STRIP_ABSTRACT_AND_SOURCES": True,

    # Base directory to output all agent answers
    "ALL_AGENT_BASE_DIR": os.path.abspath("query_result/all_agent_answers"),
    # Empty files to create in the category level output directory
    "CATEGORY_NEW_FILES": ['judge_claude.txt', 'judge_gemini.txt', 'judge_gpt.txt'],
    # Empty files to create in the meta-category level output directory
    "META_CATEGORY_NEW_FILES": ['judge_summary.txt', 'judge_csv.txt']
}


def strip_abstract_and_sources(text):
    """Remove ABSTRACT section from start and SOURCES section from end."""
    # Strip abstract: everything before "ARTIFACT:" line
    artifact_match = re.search(r'^ARTIFACT:\s*\n', text, re.MULTILINE)
    if artifact_match:
        text = text[artifact_match.end():]
    # Strip sources: everything from "SOURCES:" line onward
    sources_match = re.search(r'\n\s*SOURCES:\s*\n', text)
    if sources_match:
        text = text[:sources_match.start()]
    return text.strip() + '\n'


def find_latest_synthesis(experiment_glob, file_pattern="*merged-3*",
                          base_dir=None):
    """Find file_pattern in the highest-numbered synthesis folder.

    Args:
        experiment_glob: Glob for experiment dirs (e.g. '03_28_*')
        file_pattern: Filename glob inside the synthesis folder
        base_dir: Override for CAESAR_AGENT_BASE_DIR

    Returns:
        Relative glob string usable as caesar_sources value.
    """
    base = base_dir or DEFAULT_CONFIG["CAESAR_AGENT_BASE_DIR"]
    # Find all synthesis dirs matching the experiment glob
    synth_dirs = sorted(glob.glob(os.path.join(base, experiment_glob,
        "*.synthesis.*")))
    if not synth_dirs:
        raise FileNotFoundError(
            f"No synthesis dirs found for {experiment_glob!r} in {base}")
    # Pick the highest-numbered synthesis dir that contains matching files
    def _synth_key(p):
        m = re.search(r'\.synthesis\.(\d+)', p)
        return int(m.group(1)) if m else 0
    synth_dirs.sort(key=_synth_key, reverse=True)

    for candidate in synth_dirs:
        rel = os.path.relpath(candidate, base)
        result = os.path.join(rel, file_pattern)
        matches = glob.glob(os.path.join(base, result))
        if matches:
            break
    else:
        raise FileNotFoundError(
            f"No synthesis dir with files matching {file_pattern!r} "
            f"for {experiment_glob!r} in {base}")
    print(f"Auto-resolved: {result}  ({len(matches)} file(s))")
    return result


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

            if config['STRIP_ABSTRACT_AND_SOURCES']:
                with open(dst, 'r') as f:
                    original = f.read()
                stripped = strip_abstract_and_sources(original)
                if stripped != original:
                    with open(dst, 'w') as f:
                        f.write(stripped)
                    print(f'  Stripped abstract/sources from {caesar_filename}')


if __name__ == '__main__':
    from transfer_configs import TRANSFER_CONFIGS
    prepare_artifact(TRANSFER_CONFIGS['12_13_insights'])
