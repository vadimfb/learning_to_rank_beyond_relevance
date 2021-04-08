import scipy.sparse as sp
import pandas as pd
import numpy as np
import inspect


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

    model_params = dict()
    args = inspect.getfullargspec(BayesianPersonalizedRanking.__init__)[0]
    for param, param_val in params.items():
        if param in args:
            model_params[param] = param_val

    model = BayesianPersonalizedRanking(**model_params)
    model.fit(train_matrix, show_progress=False)

    return model, train_matrix


def train_implicit_als(train_df: pd.DataFrame,
                       params: Dict[str, Union[str, float, int]],
                       shape: [int, int]) -> Tuple[AlternatingLeastSquares, sp.csr.csr_matrix]:

    train_matrix = create_implicit_train_matrix(train_df, shape)
    alpha = params.get('alpha', 40)
    train_matrix.data = alpha * train_matrix.data + 1

    model_params = dict()
    args = inspect.getfullargspec(AlternatingLeastSquares.__init__)[0]
    for param, param_val in params.items():
        if param != 'alpha' and param in args:
            model_params[param] = param_val

    model = AlternatingLeastSquares(**model_params)
    model.fit(train_matrix, show_progress=False)

    return model, train_matrix


def train_implicit_model(train_df: pd.DataFrame,
                         params: Dict[str, Union[str, float, int]],
                         shape: [int, int]) -> Tuple[MatrixFactorizationBase, sp.csr.csr_matrix]:

    algo = params.get('algo', 'bpr')
    model, train_matrix = None, None
    if algo == 'bpr':
        model, train_matrix = train_implicit_bpr(train_df, params, shape)
    elif algo == 'als':
        model, train_matrix = train_implicit_als(train_df, params, shape)

    return model, train_matrix


def implicit_predict(users: List[int],
                     train_matrix: sp.csr.csr_matrix,
                     model: MatrixFactorizationBase,
                     recommend_len: int = 100) -> Dict[int, List[int]]:

    recommend_lists = dict()

    user_item_matrix = train_matrix.T.tocsr()

    for user in users:
        recommendations = model.recommend(userid=user,
                                          user_items=user_item_matrix,
                                          N=recommend_len)
        recommendations = [item for item, _ in recommendations]
        recommend_lists[user] = recommendations

    return recommend_lists
