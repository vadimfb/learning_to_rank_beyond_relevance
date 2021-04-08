from src.opt_tools import optimize_implicit, evaluate_implicit
from src.utils import read_ml_1m, get_folds_by_time

from hyperopt import hp
import json

search_space = {
    'algo': hp.choice(label='algo', options=['als', 'bpr']),
    'alpha': hp.uniform(label='alpha', low=1, high=1000),
    'factors': hp.randint(label='rank', low=4, high=256),
    'regularization': hp.uniform(label='reg', low=1, high=1000),
    'learning_rate': hp.uniform(label='lr', low=0, high=0.1)
}

if __name__ == '__main__':

    ratings, movies, shape = read_ml_1m()
    folds = get_folds_by_time(ratings, n_folds=3, test_size=0.1)
    best, results = optimize_implicit(folds, shape, search_space, 10, 'ndcg')

    for key, val in best.items():
        if 'int' in str(type(val)):
            best[key] = int(val)
        else:
            best[key] = float(val)

    with open('training_info/implicit_best_params.json', 'w') as f:
        json.dump(best, f)

    results.to_csv('training_info/implicit_hyperopt_results.csv', index=False)

    if best['algo'] == 0:
        best['algo'] = 'als'
    elif best['algo'] == 1:
        best['algo'] = 'bpr'

    results = evaluate_implicit(evaluate_set=folds['train'],
                                params=best,
                                shape=shape,
                                metrics=['precision', 'map', 'ndcg'],
                                k_list=[1, 5, 10, 20, 50])

    results.to_csv('eval_results/implicit_eval_results.csv', index=False)
