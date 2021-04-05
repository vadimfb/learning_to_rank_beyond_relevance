import pandas as pd
import numpy as np
import os

from typing import Tuple, List, Dict
from sklearn.model_selection import KFold
from math import ceil


def reindex_user_and_item(df: pd.DataFrame) -> None:
    _reindex(df, 'user')
    _reindex(df, 'item')


def _reindex(df: pd.DataFrame, col: str) -> None:
    uniq_ids = df[col].unique()

    def get_map(ids):
        mapping = pd.DataFrame(
            {
                'old': ids,
                'new': np.arange(ids.shape[0], dtype='uint32')
            }
        ).set_index('old')['new'].to_dict()

        return mapping

    df[col] = df[col].map(get_map(uniq_ids)).astype('uint32')


def read_ml_1m(dir_path: str = '../data/ml-1m') -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[int, int]]:

    ratings_path = os.path.join(dir_path, 'ratings.dat')
    movies_path = os.path.join(dir_path, 'movies.dat')
    ratings = pd.read_csv(ratings_path,
                          sep='::',
                          header=None,
                          names=['user', 'item', 'rating', 'timestamp'], engine='python')

    movies = pd.read_csv(movies_path,
                         sep='::',
                         header=None,
                         names=['item', 'title', 'genre'],
                         engine='python')

    reindex_user_and_item(ratings)
    _reindex(movies, 'item')

    shape = (ratings['user'].max() + 1, ratings['item'].max() + 1)

    return ratings, movies, shape


def split_recsys_data(df: pd.DataFrame, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(ceil(df.shape[0] * (1 - test_size)))
    train_set, test_set = df.iloc[:split_idx, :], df.iloc[split_idx:, :]

    train_users = set(train_set['user'].unique())
    test_set = test_set[test_set['user'].isin(train_users)].reset_index(drop=True)

    return train_set, test_set


def split_by_user_folds(df: pd.DataFrame, n_folds: int = 3, seed: int = 0, shuffle: bool = True) -> List[pd.DataFrame]:

    uniq_users = df['user'].unique()
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    folds = [
        df[df['user'].isin(uniq_users[test_ind])].reset_index(drop=True)
        for _, test_ind in kf.split(uniq_users)
    ]

    return folds


def get_folds_by_time(df: pd.DataFrame,
                      n_folds: int = 3,
                      test_size: float = 0.1) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    folds = dict()
    train, test = split_recsys_data(df, test_size)

    folds['train'] = (train, test)
    for i in range(n_folds):
        train_fold, test_fold = split_recsys_data(train, test_size)
        folds[f'val_{i}'] = (train_fold, test_fold)

    return folds
