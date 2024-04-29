import os
import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool, MetricVisualizer
from copy import deepcopy
from sklearn.metrics import ndcg_score
import argparse


def train_test_split(train_size, df):
    train_size = int(train_size*len(df))
    shuffle_df = df.sample(frac=1)

    train_df = shuffle_df[:train_size].reset_index(drop=True)
    test_df = shuffle_df[train_size:].reset_index(drop=True)

    train_df = train_df.sort_values(by=['query_id']).reset_index(drop=True)
    test_df = test_df.sort_values(by=['query_id']).reset_index(drop=True)
    
    X_train = train_df.drop(columns=['rank', 'query_id'], axis=1)
    y_train = train_df['rank']
    queries_train = train_df['query_id']

    X_test = test_df.drop(columns=['rank', 'query_id'], axis=1)
    y_test = test_df['rank']
    queries_test = test_df['query_id']

    return X_train, X_test, y_train, y_test, queries_train, queries_test


def fit_model(parameters, train_pool, test_pool, additional_params=None):
    parameters = deepcopy(parameters)
    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model

def create_weights(queries):
    query_set = np.unique(queries)
    query_weights = np.random.uniform(size=query_set.shape[0])
    weights = np.zeros(shape=queries.shape)

    for i, query_id in enumerate(query_set):
        weights[queries == query_id] = query_weights[i]

    return weights    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ranker")

    parser.add_argument('--input-file', '-i', type=str, default='intern_task.csv', help="input file with things one per line")
    parser.add_argument('--work-dir', '-w', type=str, default='out', help="output working directory")
    parser.add_argument('--out-file', '-o', type=str, default='pair-logit', help="output working directory")
    parser.add_argument('--resume', action='store_true', default=False, help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--train-size', type=float, default=0.8, help="train size")
    parser.add_argument('--max-steps', type=int, default=2000, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='CPU', help="device to use for compute, examples: CPU|GPU")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--top-k', type=int, default=5, help="top-k")
    parser.add_argument('--loss', type=str, default='PairLogit', help="model, RMSE|QueryRMSE|PairLogit|PairLogitPairwise|YetiRankPairwise")    
    parser.add_argument('--weight', action='store_true', help="when this flag is used, we will use weighted queries")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--verbose', '-v', type=int, default=500, help="verbose")
    args = parser.parse_args()


    df = pd.read_csv(args.input_file)
    X_train, X_test, y_train, y_test, queries_train, queries_test = train_test_split(args.train_size, df)

    max_relevance = np.max(y_train)

    y_train /= max_relevance
    y_test /= max_relevance

    train = Pool(
        data=X_train,
        label=y_train.values,
        group_id=queries_train
    )

    test = Pool(
        data=X_test,
        label=y_test,
        group_id=queries_test
    )

    if args.weight:
        train.set_group_weight(create_weights(queries_train))
        test.swet_group_weight(create_weights(queries_test))

    default_parameters = {
    'loss_function': args.loss,
    'iterations': args.max_steps,
    'custom_metric': [f'NDCG:top={args.top_k}', 'PFound', f'AverageGain:top={args.top_k}', f'MAP:top={args.top_k}', f'RecallAt:top={args.top_k}', f'PrecisionAt:top={args.top_k}'],
    'verbose': args.verbose if args.verbose > 0 else False,
    'random_seed': args.seed,
    'train_dir': args.work_dir,
    'learning_rate': args.learning_rate,
    'task_type': args.device,
    }

    if args.resume:
        model = CatBoostRanker()
        model.load_model(args.out_file)
    else:
        model = fit_model(default_parameters, train, test)    

    model.save_model(args.out_file)
    widget = MetricVisualizer(args.work_dir)
    widget.start()

    y_pred = model.predict(X_test)
    ndcg_score([y_pred], [y_test], k=5)
    print(ndcg_score)