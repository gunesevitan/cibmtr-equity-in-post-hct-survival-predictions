import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import catboost as cb
import optuna

sys.path.append('..')
import settings
import preprocessing
import metrics


def objective(trial):

    parameters = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 1200,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.005),
        'random_seed': None,
        'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'bootstrap_type': 'Bernoulli',
        'subsample': trial.suggest_float('subsample', 0.2, 1.0, step=0.05),
        'random_strength': trial.suggest_categorical('random_strength', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'use_best_model': False,
        'depth': trial.suggest_int('depth', 1, 8),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100, step=5),
        'has_time': False,
        'rsm': trial.suggest_float('rsm', 0.2, 1.0, step=0.05),
        'boosting_type': 'Plain',
        'boost_from_average': True,
        'langevin': trial.suggest_categorical('langevin', [True, False]),
        'metric_period': None,
        'silent': True,
        'thread_count': 16,
        'task_type': 'CPU',
        'border_count': trial.suggest_categorical('border_count', [255, 384, 512]),
    }

    df['prediction'] = 0.

    for fold in folds:

        for seed in seeds:

            training_mask = df[f'fold{fold}'] == 0
            validation_mask = df[f'fold{fold}'] == 1

            if config['training']['two_stage']:
                training_mask = training_mask & (df['efs'] == 1)

            training_dataset = cb.Pool(
                df.loc[training_mask, features],
                label=df.loc[training_mask, target],
                cat_features=categorical_features,
                weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None
            )
            validation_dataset = cb.Pool(
                df.loc[validation_mask, features],
                label=df.loc[validation_mask, target],
                cat_features=categorical_features,
                weight=df.loc[validation_mask, 'weight'] if config['training']['sample_weight'] else None
            )
            parameters['random_seed'] = seed

            model = cb.train(
                params=parameters,
                dtrain=training_dataset,
                evals=[validation_dataset]
            )

            if task == 'classification':
                validation_predictions = model.predict(validation_dataset, prediction_type='Probability')[:, 1]
            else:
                validation_predictions = model.predict(validation_dataset)

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
        categorical_columns=config['dataset']['categorical_columns'], categorical_dtype=str,
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
        study.optimize(objective, n_trials=100)
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
