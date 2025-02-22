import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna

sys.path.append('..')
import settings
import preprocessing
import metrics


def objective(trial):

    parameters = {
        'loss': 'log_loss',
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.005),
        'max_iter': trial.suggest_int('max_iter', 100, 500, step=10),
        'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
        'l2_regularization': trial.suggest_categorical('l2_regularization', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'max_features': trial.suggest_float('max_features', 0.2, 1.0, step=0.05),
        'max_bins': trial.suggest_categorical('max_bins', [127, 255]),
        'interaction_cst': trial.suggest_categorical('interaction_cst', ['pairwise', 'no_interactions', None]),
        'warm_start': False,
        'early_stopping': False,
        'scoring': 'loss',
        'validation_fraction': None,
        'random_state': None,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
    }

    df['prediction'] = 0.

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        for seed in seeds:

            parameters['random_state'] = seed

            model = HistGradientBoostingClassifier(**parameters)
            model.fit(
                X=df.loc[training_mask, features],
                y=df.loc[training_mask, target],
                sample_weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None
            )

            if task == 'classification':
                validation_predictions = model.predict_proba(df.loc[validation_mask, features])[:, 1]
            else:
                validation_predictions = model.predict(df.loc[validation_mask, features])

            if task == 'ranking':
                validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)

            if config['training']['rank_transform']:
                validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

            df.loc[validation_mask, 'prediction'] = validation_predictions

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
        categorical_columns=config['dataset']['categorical_columns'],
        continuous_columns=config['dataset']['continuous_columns'],
        transformer_directory=settings.DATA / 'linear_model_transformers',
        load_transformers=False,
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        efs_weight=config['training']['efs_weight']
    )

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']
    seeds = config['training']['seeds']

    try:
        storage = f'sqlite:///{model_directory}/study.db'
        study = optuna.create_study(
            study_name=f'{model_directory.name}_study',
            storage=storage,
            load_if_exists=True,
            direction='minimize'
        )
        study.optimize(objective, n_trials=300)
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
