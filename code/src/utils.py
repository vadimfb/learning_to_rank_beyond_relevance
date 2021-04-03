import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold

def reindex_user_and_item(df):
    _reindex(df, 'user')
    _reindex(df, 'item')

def _reindex(df, col):
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

def read_ml_1m(dir_path='../data/ml-1m'):

    ratings_path = os.path.join(dir_path, 'ratings.dat')
    movies_path = os.path.join(dir_path, 'movies.dat')
    ratings = pd.read_csv(ratings_path, sep='::', header=None,
                     names=['user', 'item', 'rating', 'timestamp'], engine='python')

    movies = pd.read_csv(movies_path, sep='::', header=None,
                         names=['item', 'title', 'genre'], engine='python')

    reindex_user_and_item(ratings)
    _reindex(movies, 'item')

    shape = (ratings['user'].max() + 1, ratings['item'].max() + 1)

    return ratings, movies, shape

def split_recsys_data(df, test_size=0.1):

    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(np.ceil(df.shape[0] * (1 - test_size)))
    train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    train_users = set(train_set['user'].unique())
    test_set = test_set.query('user in @train_users').reset_index(drop=True)

    return train_set, test_set

def split_by_user_folds(df, n_folds=3, seed=0, shuffle=True):

    uniq_users = df['user'].unique()
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    folds = [
        df[df['user'].isin(uniq_users[test_ind])].reset_index(drop=True)
        for _, test_ind in kf.split(uniq_users)
    ]

    return folds
