import pandas as pd
import os

def read_ml_1m(dir_path='../data/ml-1m'):

    ratings_path = os.path.join(dir_path, 'ratings.dat')
    movies_path = os.path.join(dir_path, 'movies.dat')
    ratings = pd.read_csv(ratings_path, sep='::', header=None,
                     names=['user', 'item', 'rating', 'timestamp'], engine='python')

    movies = pd.read_csv(movies_path, sep='::', header=None,
                         names=['item', 'title', 'genre'], engine='python')

    return {'ratings': ratings, 'item_data': movies}
