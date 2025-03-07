import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

sys.path.append('..')
import settings
import preprocessing
import metrics


def objective(trial):

    parameters = {
        'booster': 'gbtree',
        'device': 'cpu',
        'nthread': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.005),
        'gamma': trial.suggest_categorical('gamma', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 100, step=5),
        'max_delta_step': trial.suggest_float('max_delta_step', 0., 1.0, step=0.05),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0, step=0.05),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.2, 1.0, step=0.05),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.2, 1.0, step=0.05),
        'lambda': trial.suggest_categorical('lambda', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'alpha': 0,
        'tree_method': 'hist',
        'grow_policy': 'depthwise',
        'max_bin': trial.suggest_categorical('max_bin', [255, 384, 512]),
        'objective': 'reg:squarederror',
        #'huber_slope': trial.suggest_float('huber_slope', 0.2, 1.0, step=0.05),
        'eval_metric': None,
        'seed': None
    }

    df['prediction'] = 0.

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        if config['training']['two_stage']:
            training_mask = training_mask & (df['efs'] == 1)

        for seed in seeds:

            training_dataset = xgb.DMatrix(
                df.loc[training_mask, features],
                label=df.loc[training_mask, target],
                weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None,
                enable_categorical=True
            )
            validation_dataset = xgb.DMatrix(
                df.loc[validation_mask, features],
                label=df.loc[validation_mask, target],
                weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None,
                enable_categorical=True
            )

            parameters['seed'] = seed

            model = xgb.train(
                params=parameters,
                dtrain=training_dataset,
                evals=[(validation_dataset, 'val')],
                num_boost_round=1000,
                early_stopping_rounds=None,
                verbose_eval=0,
            )

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
