import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

sys.path.append('..')
import settings
import preprocessing
import metrics


def objective(trial):

    """
    Objective function to minimize

    Parameters
    ----------
    trial: optuna.trial.Trial
        Optuna Trial

    Returns
    -------
    score: float

    """

    parameters = {
        'objective': 'l2',
        'boosting_type': 'gbdt',
        'data_sample_strategy': 'bagging',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.005),
        'num_leaves': trial.suggest_categorical('num_leaves', [8, 12, 16, 24, 32, 48, 64, 96, 128,]),
        'tree_learner': 'serial',
        'num_threads': -1,
        'device_type': 'cpu',
        'seed': None,
        'bagging_seed': None,
        'feature_fraction_seed': None,
        'extra_seed': None,
        'data_random_seed': None,
        'deterministic': False,
        'max_depth': -1,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100, step=5),
        'min_sum_hessian_in_leaf': 0.,
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0, step=0.05),
        'bagging_freq': 1,
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0, step=0.05),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.2, 1.0, step=0.05),
        'extra_trees': False,
        'lambda_l1': 0.,
        'lambda_l2': trial.suggest_categorical('lambda_l2', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'linear_lambda': 0.,
        'min_gain_to_split': 0.,
        'min_data_per_group': trial.suggest_int('min_data_per_group', 5, 100, step=5),
        'max_cat_threshold': trial.suggest_categorical('max_cat_threshold', [4, 8, 16, 32, 64, 128]),
        'cat_l2': trial.suggest_categorical('cat_l2', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'cat_smooth': trial.suggest_categorical('cat_smooth', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'max_cat_to_onehot': trial.suggest_categorical('max_cat_to_onehot', [4, 8, 16, 32, 64, 128]),
        'path_smooth': 0.,
        'max_bin': trial.suggest_categorical('max_bin', [255, 384, 512]),
        'min_data_in_bin': 3,
        'bin_construct_sample_cnt': 200000,
        'use_missing': True,
        'zero_as_missing': False,
        'verbose': -1
    }

    df['prediction'] = 0.

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        if config['training']['two_stage']:
            training_mask = training_mask & (df['efs'] == 1)

        for seed in seeds:

            training_dataset = lgb.Dataset(
                df.loc[training_mask, features],
                label=df.loc[training_mask, target],
                categorical_feature=categorical_features,
                weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None
            )
            validation_dataset = lgb.Dataset(
                df.loc[validation_mask, features],
                label=df.loc[validation_mask, target],
                categorical_feature=categorical_features,
                weight=df.loc[validation_mask, 'weight'] if config['training']['sample_weight'] else None
            )

            parameters['seed'] = seed
            parameters['feature_fraction_seed'] = seed
            parameters['bagging_seed'] = seed
            parameters['drop_seed'] = seed
            parameters['data_random_seed'] = seed

            model = lgb.train(
                params=parameters,
                train_set=training_dataset,
                valid_sets=[training_dataset, validation_dataset],
                num_boost_round=1500,
                callbacks=[
                    lgb.log_evaluation(0)
                ]
            )

            validation_predictions = model.predict(df.loc[validation_mask, features], num_iteration=1500)

            if config['training']['two_stage']:
                if config['training']['target'] == 'log_efs_time':
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)
                elif config['training']['target'] == 'log_km_survival_probability':
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] * np.exp(validation_predictions)

            if config['training']['rank_transform']:
                validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

            df.loc[validation_mask, 'prediction'] += validation_predictions / len(seeds)

    oof_mask = df['prediction'].notna()
    if task == 'ranking':
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
        score = oof_scores['stratified_concordance_index']
    elif task == 'classification':
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column='prediction'
        )
        score = oof_scores['log_loss']
    elif task == 'regression':
        oof_scores = metrics.regression_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column=target,
            prediction_column='prediction'
        )
        score = oof_scores['mean_squared_error']
    else:
        raise ValueError(f'Invalid task type {task}')

    return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    categorical_columns = config['dataset']['categorical_columns']

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=config['dataset']['categorical_columns'], categorical_dtype='category',
        kaplan_meier_targets_path=config['dataset']['kaplan_meier_targets_path'],
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        efs_weight=config['training']['efs_weight']
    )

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']
    categorical_features = config['training']['categorical_features']
    seeds = config['training']['seeds']

    try:
        storage = f'sqlite:///{model_directory}/study.db'
        study = optuna.create_study(
            study_name=f'{model_directory.name}_study',
            storage=storage,
            load_if_exists=True,
            direction='maximize'
        )
        study.optimize(objective, n_trials=500)
    except KeyboardInterrupt:
        settings.logger.info('Interrupted')
    finally:
        df_study = study.trials_dataframe().dropna()
        df_study = df_study.sort_values(by='value', ascending=False).drop_duplicates(subset='value', keep='first').reset_index(drop=True)
        df_study.to_csv(model_directory / 'study.csv', index=False)
        best_parameters = study.best_params
        with open(model_directory / 'best_parameters.json', mode='w') as f:
            json.dump(best_parameters, f, indent=2, ensure_ascii=False)
        settings.logger.info(f'Saved best_parameters.json to {model_directory}')
