import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch.utils.data as data
import torch.optim as optim
import torch

from typing import Dict, Union, Tuple, List, Iterable
from hyperopt import fmin, tpe, Trials, STATUS_OK
from collections import defaultdict
from functools import partial

from .implicit_utils import train_implicit_model, implicit_predict
from .metrics import precision_at_k, mean_average_precision_at_k, normalized_discounted_cumulative_gain_at_k
from .metrics import mean_diversity_at_k, mean_novelty_at_k, coverage_at_k
from .sampling_utils import uniform_sampling
from .data_utils import PairwiseRankingData
from .models.pairwise import BPRMatrixFactorization

RELEVANCE_METRIC_MAP = {
    'precision': precision_at_k,
    'map': mean_average_precision_at_k,
    'ndcg': normalized_discounted_cumulative_gain_at_k,
}


def get_score_by_metric(recommend_lists: Dict[int, List[int]],
                        relevant_lists: Dict[int, List[int]],
                        metric: str,
                        k: int,
                        distancies: sp.lil.lil_matrix = None,
                        item_long_tails: Dict[int, float] = None,
                        n_items: int = None) -> float:
    if metric == 'diversity':
        score = mean_diversity_at_k(recommend_lists, distancies, k)
    elif metric == 'novelty':
        score = mean_novelty_at_k(recommend_lists, item_long_tails, k)
    elif metric == 'coverage':
        score = coverage_at_k(recommend_lists, n_items, k)
    else:
        score = RELEVANCE_METRIC_MAP[metric](relevant_lists, recommend_lists, k)

    return score


def evaluate_implicit(
        evaluate_set: Tuple[pd.DataFrame, pd.DataFrame],
        params: Dict[str, Union[int, float, str]],
        shape: Tuple[int, int],
        metrics: List[str],
        k_list: List[int],
        distancies: sp.lil.lil_matrix = None,
        item_long_tails: Dict[int, float] = None,
        n_items: int = None) -> pd.DataFrame:

    train_df = evaluate_set[0]
    test_df = evaluate_set[1]

    model, train_matrix = train_implicit_model(train_df, params, shape)

    test_users = test_df['user'].unique().tolist()
    relevant_lists = test_df.groupby('user')['item'].agg(list).to_dict()

    scores = defaultdict(list)
    for metric in metrics:
        for k in k_list:
            recommend_lists = implicit_predict(test_users, train_matrix, model, recommend_len=k)
            score = get_score_by_metric(recommend_lists,
                                        relevant_lists,
                                        metric,
                                        k,
                                        distancies,
                                        item_long_tails,
                                        n_items)

            scores[metric].append(score)

    scores = pd.DataFrame(scores)
    scores['k'] = k_list

    return scores


def cv_implicit(folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                shape: Tuple[int, int],
                params: Dict[str, Union[str, int, float]],
                metric: str,
                k: int = 10,
                distancies: sp.lil.lil_matrix = None,
                item_long_tails: Dict[int, float] = None,
                n_items: int = None):

    scores = list()
    for fold_name, sets in folds.items():
        if fold_name == 'train':
            continue

        train_df = sets[0]
        test_df = sets[1]
        model, train_matrix = train_implicit_model(train_df, params, shape)

        test_users = test_df['user'].unique().tolist()
        relevant_lists = test_df.groupby('user')['item'].agg(list).to_dict()
        recommend_lists = implicit_predict(test_users, train_matrix, model, recommend_len=k)

        score = get_score_by_metric(recommend_lists,
                                    relevant_lists,
                                    metric,
                                    k,
                                    distancies,
                                    item_long_tails,
                                    n_items)

        scores.append(score)

    return np.mean(scores)


def optimize_implicit(folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                      shape: Tuple[int, int],
                      param_space: Dict[str, Iterable],
                      max_evals: int,
                      metric: str,
                      k: int = 10,
                      distancies: sp.lil.lil_matrix = None,
                      item_long_tails: Dict[int, float] = None,
                      n_items: int = None) -> Tuple[Dict[str, Union[int, float]], pd.DataFrame]:

    def objective(params: Dict[str, Union[str, int, float]],
                  folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                  shape: Tuple[int, int],
                  metric: str,
                  k: int = 10,
                  distancies: sp.lil.lil_matrix = None,
                  item_long_tails: Dict[int, float] = None,
                  n_items: int = None):

        loss = -cv_implicit(folds, shape, params, metric, k, distancies, item_long_tails, n_items)

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=partial(objective,
                           folds=folds,
                           shape=shape,
                           metric=metric,
                           k=k,
                           distancies=distancies,
                           item_long_tails=item_long_tails,
                           n_items=n_items),
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(1),
                show_progressbar=True)

    results = pd.DataFrame([{**x, **x['params']} for x in trials.results])
    results.drop(labels=['status', 'params'], axis=1, inplace=True)
    results.sort_values(by=['loss'], ascending=False, inplace=True)

    return best, results


def evaluate_pairwise(
        evaluate_set: Tuple[pd.DataFrame, pd.DataFrame],
        params: Dict[str, Union[int, float, str]],
        shape: Tuple[int, int],
        metrics: List[str],
        k_list: List[int],
        distancies: sp.lil.lil_matrix = None,
        item_long_tails: Dict[int, float] = None,
        n_items: int = None) -> pd.DataFrame:

    train_df = evaluate_set[0]
    test_df = evaluate_set[1]

    train_dataset_with_negatives = uniform_sampling(train_df, item_num=shape[1], sample_size=params['sample_size'])
    train_dataset_with_negatives = PairwiseRankingData(train_dataset_with_negatives)
    relevant_lists = test_df.groupby('user')['item'].agg(list).to_dict()

    train_loader = data.DataLoader(
        train_dataset_with_negatives,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4
    )

    model = BPRMatrixFactorization(
        user_num=shape[0],
        item_num=shape[1],
        factors=params['factors'],
        epochs=params['epochs'],
        lr=params['lr'],
        reg_1=params['user_reg'],
        reg_2=params['item_reg']
    )

    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    model.fit(train_loader, optimizer)
    model.eval()

    test_users = torch.Tensor(test_df['user'].unique().astype(int)).long()

    scores = defaultdict(list)
    for metric in metrics:
        for k in k_list:
            recommend_lists = model.predict(test_users, k)
            recommend_lists = {user: [item for item, _ in recommend_lists[user]] for user in recommend_lists}
            score = get_score_by_metric(recommend_lists,
                                        relevant_lists,
                                        metric,
                                        k,
                                        distancies,
                                        item_long_tails,
                                        n_items)

            scores[metric].append(score)

    scores = pd.DataFrame(scores)
    scores['k'] = k_list

    return scores


def cv_pairwise(folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                shape: Tuple[int, int],
                params: Dict[str, Union[str, int, float]],
                metric: str,
                k: int = 10,
                distancies: sp.lil.lil_matrix = None,
                item_long_tails: Dict[int, float] = None,
                n_items: int = None):

    scores = list()
    for fold_name, sets in folds.items():
        if fold_name == 'train':
            continue

        train_df = sets[0]
        test_df = sets[1]

        train_dataset_with_negatives = uniform_sampling(train_df, item_num=shape[1], sample_size=params['sample_size'])
        train_dataset_with_negatives = PairwiseRankingData(train_dataset_with_negatives)
        relevant_lists = test_df.groupby('user')['item'].agg(list).to_dict()

        train_loader = data.DataLoader(
            train_dataset_with_negatives,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4
        )

        model = BPRMatrixFactorization(
            user_num=shape[0],
            item_num=shape[1],
            factors=params['factors'],
            epochs=params['epochs'],
            lr=params['lr'],
            reg_1=params['user_reg'],
            reg_2=params['item_reg']
        )

        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        model.fit(train_loader, optimizer)
        model.eval()

        test_users = torch.Tensor(test_df['user'].unique().astype(int)).long()
        recommend_lists = model.predict(test_users, k)
        recommend_lists = {user: [item for item, _ in recommend_lists[user]] for user in recommend_lists}

        score = get_score_by_metric(recommend_lists,
                                    relevant_lists,
                                    metric,
                                    k,
                                    distancies,
                                    item_long_tails,
                                    n_items)

        scores.append(score)

    return np.mean(scores)


def optimize_pairwise(folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                      shape: Tuple[int, int],
                      param_space: Dict[str, Iterable],
                      max_evals: int,
                      metric: str,
                      k: int = 10,
                      distancies: sp.lil.lil_matrix = None,
                      item_long_tails: Dict[int, float] = None,
                      n_items: int = None) -> Tuple[Dict[str, Union[int, float]], pd.DataFrame]:

    def objective(params: Dict[str, Union[str, int, float]],
                  folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                  shape: Tuple[int, int],
                  metric: str,
                  k: int = 10,
                  distancies: sp.lil.lil_matrix = None,
                  item_long_tails: Dict[int, float] = None,
                  n_items: int = None):

        loss = -cv_pairwise(folds, shape, params, metric, k, distancies, item_long_tails, n_items)

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=partial(objective,
                           folds=folds,
                           shape=shape,
                           metric=metric,
                           k=k,
                           distancies=distancies,
                           item_long_tails=item_long_tails,
                           n_items=n_items),
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(1),
                show_progressbar=True)

    results = pd.DataFrame([{**x, **x['params']} for x in trials.results])
    results.drop(labels=['status', 'params'], axis=1, inplace=True)
    results.sort_values(by=['loss'], ascending=False, inplace=True)

    return best, results
