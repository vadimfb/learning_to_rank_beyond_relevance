from src.opt_tools import optimize_implicit, evaluate_implicit
from src.utils import read_ml_1m, get_folds_by_time
from src.utils import calculate_movielens_distancies, caclculate_long_tail, calculate_n_items

from hyperopt import hp
import json

choice_params = {
    'algo': ['als', 'bpr'],
    'alpha': [1, 10, 40, 100],
    'factors': [16, 32, 64],
    'regularization': [0.01, 0.1, 1., 10, 100],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
}

search_space = {
    'algo': hp.choice(label='algo', options=choice_params['algo']),
    'alpha': hp.choice(label='alpha', options=choice_params['alpha']),
    'factors': hp.choice(label='factors', options=choice_params['factors']),
    'regularization': hp.choice(label='regularization', options=choice_params['regularization']),
    'learning_rate': hp.choice(label='learning_rate', options=choice_params['learning_rate'])
}

if __name__ == '__main__':

    ratings, movies, shape = read_ml_1m()
    folds = get_folds_by_time(ratings, n_folds=3, test_size=0.1)
    best, results = optimize_implicit(folds, shape, search_space, 100, 'ndcg')

    for key, val in best.items():
        if 'int' in str(type(val)):
            best[key] = int(val)
        else:
            best[key] = float(val)
        if key in choice_params:
            best[key] = choice_params[key][val]

    with open('training_info/implicit_best_params.json', 'w') as f:
        json.dump(best, f)

    results.to_csv('training_info/implicit_hyperopt_results.csv', index=False)

    distancies = calculate_movielens_distancies(item_data=movies)
    item_long_tail = caclculate_long_tail(ratings)
    n_items = calculate_n_items(ratings)
    results = evaluate_implicit(evaluate_set=folds['train'],
                                params=best,
                                shape=shape,
                                metrics=['precision', 'map', 'ndcg', 'diversity', 'novelty', 'coverage'],
                                k_list=[1, 5, 10, 20, 50],
                                distancies=distancies,
                                item_long_tails=item_long_tail,
                                n_items=n_items)

    results.to_csv('eval_results/implicit_eval_results.csv', index=False)
