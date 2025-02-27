import sys
import argparse
from pathlib import Path
import pickle
import yaml
import json
import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.ensemble

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=config['dataset']['categorical_columns'],
        continuous_columns=config['dataset']['continuous_columns'],
        transformer_directory=settings.DATA / 'linear_model_transformers',
        load_transformers=False,
        kaplan_meier_targets_path=config['dataset']['kaplan_meier_targets_path'],
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        efs_weight=config['training']['efs_weight']
    )

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']

    settings.logger.info(
        f'''
        Running Trainer for {config['model_class']}
        Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Target: {target}
        '''
    )

    df_coefficients = pd.DataFrame(
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

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({np.sum(training_mask)})
            Validation: ({np.sum(validation_mask)})
            '''
        )

        if 'tree' in config['model_class'].lower():
            model = getattr(sklearn.tree, config['model_class'])(**config['model_parameters'])
        else:
            model = getattr(sklearn.ensemble, config['model_class'])(**config['model_parameters'])

        model.fit(
            X=df.loc[training_mask, features],
            y=df.loc[training_mask, target],
            sample_weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None
        )

        model_file_name = f'model_fold_{fold}.pickle'
        with open(model_directory / model_file_name, mode='wb') as f:
            pickle.dump(model, f)
        settings.logger.info(f'{model_file_name} is saved to {model_directory}')

        try:
            df_coefficients.loc[:, fold] = model.feature_importances_
        except AttributeError:
            df_coefficients.loc[:, fold] = 0

        if task == 'classification':
            validation_predictions = model.predict_proba(df.loc[validation_mask, features])[:, 1]
        else:
            validation_predictions = model.predict(df.loc[validation_mask, features])

        if config['training']['two_stage']:
            if config['training']['target'] == 'log_efs_time':
                df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)
            elif config['training']['target'] == 'log_km_survival_probability':
                df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                validation_predictions = df.loc[validation_mask, 'efs_prediction'] * np.exp(validation_predictions)

        if config['training']['rank_transform']:
            validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

        df.loc[validation_mask, 'prediction'] = validation_predictions

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
                prediction_column='prediction',
                weight_column=None
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
            prediction_column='prediction',
            weight_column=None
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
        title=f'{config["model_class"]} Model Scores of {len(folds)} Fold(s)',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'Saved scores.png to {model_directory}')

    if task == 'classification':

        visualization.visualize_roc_curves(
            roc_curves=[curve['roc'] for curve in curves],
            title=f'{config["model_class"]} Model Validation ROC Curves',
            path=model_directory / 'roc_curves.png'
        )
        settings.logger.info(f'Saved roc_curves.png to {model_directory}')

        visualization.visualize_pr_curves(
            pr_curves=[curve['pr'] for curve in curves],
            title=f'{config["model_class"]} Model Validation PR Curves',
            path=model_directory / 'pr_curves.png'
        )
        settings.logger.info(f'Saved pr_curves.png to {model_directory}')

    df_coefficients['mean'] = df_coefficients[config['training']['folds']].mean(axis=1)
    df_coefficients['std'] = df_coefficients[config['training']['folds']].std(axis=1).fillna(0)
    df_coefficients.sort_values(by='mean', ascending=False, inplace=True)
    visualization.visualize_feature_importance(
        df_feature_importance=df_coefficients,
        title=f'{config["model_class"]} Model Feature Importance',
        path=model_directory / f'feature_importance.png'
    )
    settings.logger.info(f'Saved feature_importance.png to {model_directory}')

    columns_to_save = ['prediction', 'reg_1_prediction'] if config['training']['two_stage'] else ['prediction']
    df.loc[:, columns_to_save].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
