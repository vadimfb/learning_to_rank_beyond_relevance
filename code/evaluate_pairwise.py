from src.opt_tools import optimize_pairwise, evaluate_pairwise
from src.utils import read_ml_1m, get_folds_by_time

from hyperopt import hp
import json

choice_params = {
    'sample_size': [4],
    'batch_size': [10000],
    'epochs': [50],
    'lr': [0.01],
    'factors': [32],
    'user_reg': [0.0],
    'item_reg': [0.01]
}

search_space = {
    'sample_size': hp.choice(label='sample_size', options=choice_params['sample_size']),
    'batch_size': hp.choice(label='batch_size', options=choice_params['batch_size']),
    'epochs': hp.choice(label='epochs', options=choice_params['epochs']),
    'lr': hp.choice(label='lr', options=choice_params['lr']),
    'factors': hp.choice(label='factors', options=choice_params['factors']),
    'user_reg': hp.choice(label='user_reg', options=choice_params['user_reg']),
    'item_reg': hp.choice(label='item_reg', options=choice_params['item_reg'])
}

if __name__ == '__main__':

    ratings, movies, shape = read_ml_1m()
    folds = get_folds_by_time(ratings, n_folds=3, test_size=0.1)
    best, results = optimize_pairwise(folds, shape, search_space, 1, 'ndcg')

    for key, val in best.items():
        if 'int' in str(type(val)):
            best[key] = int(val)
        else:
            best[key] = float(val)
        if key in choice_params:
            best[key] = choice_params[key][val]

    with open('training_info/pairwise_best_params.json', 'w') as f:
        json.dump(best, f)

    results.to_csv('training_info/pairwise_hyperopt_results.csv', index=False)

    results = evaluate_pairwise(evaluate_set=folds['train'],
                                params=best,
                                shape=shape,
                                metrics=['precision', 'map', 'ndcg'],
                                k_list=[1, 5, 10, 20, 50])

    results.to_csv('eval_results/pairwise_eval_results.csv', index=False)