# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the dataset to parquet format
"""

import os
from datasets import load_dataset, concatenate_datasets
from functools import partial

from verl.utils.hdfs_io import copy, makedirs
import argparse


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx
        }
    }
    return data


def build_aime2024_dataset():

    def process_aime2024(example):
        return example["Problem"], str(example["Answer"])

    data_source = 'Maxwell-Jia/AIME_2024'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(example_map_fn,
                     process_fn=process_aime2024,
                     data_source=data_source,
                     ability="English",
                     split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_dimond_dataset():
    GPQA_QUERY_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"

    def process_gpqa_diamond(example):
        import random
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(A=choices[0],
                                                  B=choices[1],
                                                  C=choices[2],
                                                  D=choices[3],
                                                  Question=example["Question"])
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = 'Idavidrein/gpqa'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(example_map_fn,
                     process_fn=process_gpqa_diamond,
                     data_source=data_source,
                     ability="Math",
                     split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


TASK2DATA = {
    "aime2024": build_aime2024_dataset,
    "gpqa_diamond": build_gpqa_dimond_dataset,
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--tasks', default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(',') if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS.keys():
                raise NotImplementedError(f"{task} has not been supported.")

    datasets = []
    for task in args.tasks:
        datasets.append(TASK2DATA[task]())
    test_dataset = concatenate_datasets(datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
