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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import math, math_verify, gpqa, livecodebench
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source in ['Maxwell-Jia/AIME_2024', "opencompass/LiveMathBench"]:
        return math_verify.compute_score
    elif data_source == 'Idavidrein/gpqa':
        return gpqa.compute_score
    elif data_source in ['livecodebench/code_generation_lite', 'livecodebench/code_generation']:
        return livecodebench.compute_score
    else:
        raise NotImplementedError


def process_item(data_source, response_lst, reward_data):
    reward_fn = select_reward_fn(data_source)
    ground_truth = reward_data['ground_truth']
    score_lst = [reward_fn(r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    num_process_evaluate = 16

    total = len(dataset)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    args = [(data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]
    with tqdm(total=total) as pbar:

        with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
            futures = {executor.submit(process_item, *arg): i for i, arg in enumerate(args)}

            for future in as_completed(futures):
                data_source, score = future.result()
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

    print(metric_dict)


if __name__ == '__main__':
    main()
