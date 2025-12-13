"""Artifact Preparation - Copy Caesar artifacts to query results for judging and evaluation"""
import glob
import os
import random
import shutil
import sys
import time

# Base directory filled with answers from Caesar agent
CAESAR_AGENT_BASE_DIR = os.path.abspath("result")
# Filename for Caesar agent answer in answer directory
CAESAR_AGENT_FILENAME = "answer_cat_cam.txt"
# Base directory filled with answers from other agents
OTHER_AGENT_BASE_DIR = os.path.abspath("query_result/other_agent_answers")
# Base directory to output all agent answers
ALL_AGENT_BASE_DIR = os.path.abspath("query_result/all_agent_answers")


def setup_transfer_dict():
    full = {
        'caesar_sources': '11_17_*/*12130327/*merged-3*',
        'other_sources': '12_4_answers/*/',
    }

    eli5 = {
        'caesar_sources': '11_17_*/*12130155/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5/*/',
    }

    eli5_600 = {
        'caesar_sources': '11_17_*/*12130327/*merged-eli5-3*',
        'other_sources': '12_4_answers_eli5_short/*/',
    }
    return [full, eli5, eli5_600]


def prepare_artifact():
    transfer_dicts = setup_transfer_dict()
    for td in transfer_dicts:
        other_source_dirs = glob.glob(os.path.join(OTHER_AGENT_BASE_DIR, td['other_sources']))
        for caesar_file in glob.glob(os.path.join(CAESAR_AGENT_BASE_DIR, td['caesar_sources'])):
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

            os.makedirs(ALL_AGENT_BASE_DIR, exist_ok=True)
            output_dir = os.path.join(ALL_AGENT_BASE_DIR,
                os.path.join(*os.path.normpath(other_source_dir).split(os.sep)[-2:]))
            # print(output_dir); exit()
            if not os.path.exists(output_dir):
                shutil.copytree(other_source_dir, output_dir)
            cmd = f"cp {caesar_file} {os.path.join(output_dir, CAESAR_AGENT_FILENAME)}"
            print(f'{cmd}\n'); os.system(cmd)

if __name__ == '__main__':
    prepare_artifact()
