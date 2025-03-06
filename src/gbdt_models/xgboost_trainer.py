import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


def load_model(model_directory):

    """
    Load trained XGBoost models from given path

    Parameters
    ----------
    model_directory: str or pathlib.Path
        Path-like string of the model directory

    Returns
    -------
    config: dict
        Dictionary of model configurations

    models: dict
        Dictionary of model file names as keys and model objects as values
    """

    models = {}

    for model_path in sorted(list(model_directory.glob('model*'))):
        model_path = str(model_path)
        model = xgb.Booster()
        model.load_model(model_path)
        model_file_name = model_path.split('/')[-1].split('.')[0]
        models[model_file_name] = model
        settings.logger.info(f'Loaded XGBoost model from {model_path}')

    config_path = model_directory / 'config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    settings.logger.info(f'Loaded config from {config_path}')

    return config, models


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
        nelson_aalen_targets_path=config['dataset']['nelson_aalen_targets_path'],
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        efs_weight=config['training']['efs_weight']
    )

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']
    categorical_features = config['training']['categorical_features']
    seeds = config['training']['seeds']

    settings.logger.info(
        f'''
        Running XGBoost trainer for {task} task
        Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Categorical Features: {json.dumps(categorical_features, indent=2)}
        Target: {target}
        '''
    )

    df_feature_importance_gain = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    df_feature_importance_weight = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    df_feature_importance_cover = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    scores = []
    if task == 'classification':
        curves = []
    else:
        curves = None

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        if config['training']['two_stage']:
            training_mask = training_mask & (df['efs'] == 1)

        df.loc[validation_mask, 'prediction'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({np.sum(training_mask)}) - Target Mean: {df.loc[training_mask, target].mean():.4f}
            Validation: ({np.sum(validation_mask)}) - Target Mean: {df.loc[validation_mask, target].mean():.4f}
            '''
        )

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
                weight=df.loc[validation_mask, 'weight'] if config['training']['sample_weight'] else None,
                enable_categorical=True
            )

            config['model_parameters']['seed'] = seed

            model = xgb.train(
                params=config['model_parameters'],
                dtrain=training_dataset,
                evals=[(validation_dataset, 'val')],
                num_boost_round=config['fit_parameters']['boosting_rounds'],
                early_stopping_rounds=None,
                verbose_eval=config['fit_parameters']['verbose_eval']
            )
            model_file_name = f'model_fold_{fold}_seed_{seed}.json'
            model.save_model(model_directory / model_file_name)
            settings.logger.info(f'{model_file_name} is saved to {model_directory}')

            df_feature_importance_gain[fold] += pd.Series(model.get_score(importance_type='gain')).fillna(0) / len(seeds)
            df_feature_importance_weight[fold] += pd.Series(model.get_score(importance_type='weight')).fillna(0) / len(seeds)
            df_feature_importance_cover[fold] += pd.Series(model.get_score(importance_type='cover')).fillna(0) / len(seeds)

            validation_predictions = model.predict(validation_dataset)

            if config['training']['two_stage']:
                if config['training']['target'] == 'log_efs_time':
                    df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)
                elif config['training']['target'] == 'log_km_survival_probability':
                    df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] * np.exp(validation_predictions)

            if config['training']['rank_transform']:
                validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

            df.loc[validation_mask, 'prediction'] += validation_predictions / len(seeds)

        if task == 'ranking':
            validation_scores = metrics.ranking_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                time_column='efs_time',
                event_column='efs',
                prediction_column='prediction'
            )
        elif task == 'classification':
            validation_scores = metrics.classification_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                event_column='efs',
                prediction_column='prediction'
            )
            validation_curves = metrics.classification_curves(
                df=df.loc[validation_mask],
                event_column='efs',
                prediction_column='prediction',
            )
            curves.append(validation_curves)
        elif task == 'regression':
            validation_scores = metrics.regression_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                time_column=target,
                prediction_column='prediction'
            )
        else:
            raise ValueError(f'Invalid task type {task}')

        settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(validation_scores, indent=2)}')
        scores.append(validation_scores)

    scores = pd.DataFrame(scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        ----------------------
        {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}

        Standard Deviations
        -------------------
        ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
        '''
    )

    oof_mask = df['prediction'].notna()
    if task == 'ranking':
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
    elif task == 'classification':
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column='prediction'
        )
    elif task == 'regression':
        oof_scores = metrics.regression_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column=target,
            prediction_column='prediction'
        )
    else:
        raise ValueError(f'Invalid task type {task}')

    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    scores = pd.concat((
        scores,
        pd.DataFrame([oof_scores])
    )).reset_index(drop=True)
    scores['fold'] = folds + ['OOF']
    scores = scores[scores.columns.tolist()[::-1]]
    scores.to_csv(model_directory / 'scores.csv', index=False)
    settings.logger.info(f'scores.csv is saved to {model_directory}')

    visualization.visualize_scores(
        scores=scores,
        title=f'XGBoost Model Scores of {len(folds)} Fold(s)',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'Saved scores.png to {model_directory}')

    if task == 'classification':

        visualization.visualize_roc_curves(
            roc_curves=[curve['roc'] for curve in curves],
            title=f'XGBoost Model Validation ROC Curves',
            path=model_directory / 'roc_curves.png'
        )
        settings.logger.info(f'Saved roc_curves.png to {model_directory}')

        visualization.visualize_pr_curves(
            pr_curves=[curve['pr'] for curve in curves],
            title=f'XGBoost Model Validation PR Curves',
            path=model_directory / 'pr_curves.png'
        )
        settings.logger.info(f'Saved pr_curves.png to {model_directory}')

    for importance_type, df_feature_importance in zip(['gain', 'weight', 'cover'], [df_feature_importance_gain, df_feature_importance_weight, df_feature_importance_cover]):
        df_feature_importance['mean'] = df_feature_importance[config['training']['folds']].mean(axis=1)
        df_feature_importance['std'] = df_feature_importance[config['training']['folds']].std(axis=1).fillna(0)
        df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title=f'XGBoost Feature Importance ({importance_type.capitalize()})',
            path=model_directory / f'feature_importance_{importance_type}.png'
        )
        settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

    columns_to_save = ['prediction', 'reg_1_prediction'] if config['training']['two_stage'] else ['prediction']
    df.loc[:, columns_to_save].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'Saved oof_predictions.csv to {model_directory}')
