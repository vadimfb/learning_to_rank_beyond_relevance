from typing import List, Dict
from math import log2
from scipy.sparse.lil import lil_matrix


def precision_at_k(relevant_lists: Dict[int, List[int]],
                   recommend_lists: Dict[int, List[int]],
                   k: int = 10) -> float:

    hits = 0
    n_recommendations = 0

    for user, relevant_items in relevant_lists.items():

        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        intersection_items = set(relevant_items) & set(recommend_items)
        hits += len(intersection_items)
        n_recommendations += k

    precision = hits / n_recommendations

    return precision


def recall_at_k(relevant_lists: Dict[int, List[int]],
                recommend_lists: Dict[int, List[int]],
                k: int = 10) -> float:
    hits = 0
    n_relevants = 0

    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        intersection_items = set(relevant_items) & set(recommend_items)
        hits += len(intersection_items)
        n_relevants += len(relevant_items)

    recall = hits / n_relevants

    return recall


def average_precision_at_k(relevant_items: List[int], recommend_items: List[int], k: int = 10) -> float:

    relevant_items = set(relevant_items)

    score = 0
    for rank, item in enumerate(recommend_items[: k]):
        if item in relevant_items:
            intersection_items = relevant_items.intersection(recommend_items[: rank + 1])
            hits = len(intersection_items)
            p_at_rank = hits / (rank + 1)
            score += p_at_rank

    return score


def mean_average_precision_at_k(relevant_lists: Dict[int, List[int]],
                                recommend_lists: Dict[int, List[int]],
                                k: int = 10) -> float:

    ap_sum = 0
    n_users = 0
    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        ap_sum += average_precision_at_k(relevant_items, recommend_items, k)
        n_users += 1

    mean_average_precision = ap_sum / n_users

    return mean_average_precision


def normalized_discounted_cumulative_gain_at_k(relevant_lists: Dict[int, List[int]],
                                               recommend_lists: Dict[int, List[int]],
                                               k: int = 10) -> float:

    ndcg = 0
    n_users = 0

    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        dcg = 0
        idcg = 0
        for rank, item in enumerate(recommend_items):
            if item in relevant_items:
                dcg += 1 / log2(rank + 1 + 1)
            idcg += 1 / log2(rank + 1 + 1)

        ndcg += dcg / idcg
        n_users += 1

    ndcg = ndcg / n_users

    return ndcg


def diversity_at_k(items: List[int],
                   distancies: lil_matrix,
                   k: int = 10) -> float:

    diversity = 0

    items = items[: k]
    for item_i in items:
        for item_j in items:
            if item_i != item_j:
                diversity += distancies[item_i, item_j]

    diversity = diversity / (k * (k - 1))

    return diversity


def mean_diversity_at_k(item_lists: Dict[int, List[int]], distancies: lil_matrix, k: int = 10) -> float:

    mean_diversity = 0
    n_users = 0

    for _, items in item_lists.items():
        mean_diversity += diversity_at_k(items, distancies, k)
        n_users += 1

    mean_diversity = mean_diversity / n_users

    return mean_diversity


def novelty_at_k(items: List[int], item_long_tails: Dict[int, float], k: int = 10) -> float:

    items = items[: k]

    novelty = 0
    for item in items:
        novelty += item_long_tails[item]

    novelty = novelty / k

    return novelty


def mean_novelty_at_k(item_lists: Dict[int, List[int]],
                      item_long_tails: Dict[int, float],
                      k: int = 10) -> float:

    mean_novelty = 0
    n_users = 0

    for _, items in item_lists.items():
        mean_novelty += novelty_at_k(items, item_long_tails, k)
        n_users += 1

    mean_novelty = mean_novelty / n_users

    return mean_novelty


def coverage_at_k(item_lists: Dict[int, List[int]], n_items: int, k: int = 10) -> float:

    items_set = set()

    for _, items in item_lists.items():
        items_set.update(items[: k])

    coverage = len(items_set) / n_items

    return coverage


def serendipity_at_k(items: List[int], **kwargs) -> float:
    """TO DO"""
    ...


def mean_serendipity_at_k(items: List[int], **kwargs) -> float:
    """TO DO"""
    ...
