from typing import Dict, List

import pandas as pd
import numpy as np

from collections import defaultdict

def uniform_sampling(ratings: pd.DataFrame,
                     item_num: int,
                     sample_size: int,
                     seed: int = 0) -> Dict[int, Dict[str, List[int]]]:

    np.random.seed(seed)

    result_dataset = defaultdict(get_defaultdict)
    user_history_lists = ratings.groupby('user')['item'].agg(list).to_dict()
    for user, items in user_history_lists.items():
        items = set(items)
        for item in items:
            negative_items = set(np.random.randint(0, item_num, size=sample_size))
            negative_items = list(negative_items.difference(items))
            positive_items = [item] * len(negative_items)
            result_dataset[user]['positive'] += positive_items
            result_dataset[user]['negative'] += negative_items

    return result_dataset

def get_defaultdict():
    return defaultdict(list)