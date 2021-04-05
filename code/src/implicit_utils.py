import scipy.sparse as sp
import pandas as pd
import numpy as np

from typing import Dict, Union, Tuple, List
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
from implicit.recommender_base import MatrixFactorizationBase


def create_implicit_train_matrix(df: pd.DataFrame, shape: Tuple[int, int]) -> sp.csr.csr_matrix:
    inverse_shape = (shape[1], shape[0])

    train_matrix = sp.csr_matrix(
        (
            np.ones(df.shape[0]), (df['item'], df['user'])
        ),
        shape=inverse_shape
    )

    return train_matrix


def train_implicit_bpr(train_df: pd.DataFrame,
                       params: Dict[str, Union[str, float, int]],
                       shape: [int, int]) -> Tuple[BayesianPersonalizedRanking, sp.csr.csr_matrix]:

    train_matrix = create_implicit_train_matrix(train_df, shape)

    model = BayesianPersonalizedRanking(**params)
    model.fit(train_matrix)

    return model, train_matrix


def train_implicit_als(train_df: pd.DataFrame,
                       params: Dict[str, Union[str, float, int]],
                       shape: [int, int]) -> Tuple[AlternatingLeastSquares, sp.csr.csr_matrix]:

    train_matrix = create_implicit_train_matrix(train_df, shape)
    alpha = params.get('alpha', 40)
    train_matrix.data = alpha * train_matrix.data + 1

    model_params = dict()
    for param, param_val in params:
        if param != 'alpha':
            model_params[param] = param_val

    model = AlternatingLeastSquares(model_params)
    model.fit(train_matrix)

    return model, train_matrix

def implicit_predict(users: List[int],
                     user_item_matrix: sp.csr.csr_matrix,
                     model: MatrixFactorizationBase,
                     N: int=100) -> Dict[int, List[int]]:
    """TO DO"""
    ...
